from dataclasses import dataclass
import os
import logging
import pickle
import itertools
import custom_modeling_utils as custom_modeling_utils
import dictionary_definition as dictionary_definition
import re
import torch
import time
import sys
from collections import Counter
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from typing import NamedTuple, List, Optional

logger = logging.getLogger(__name__)

oed_to_upos = {
    "exclamation": ("NOUN", "PROPN", "VERB", "NUM", "SYM", "X"),
    "abbreviation": ("X", "NOUN", "PROPN", "NUM", "SYM"),
    "noun": ("NOUN", "PROPN"),
    "adjective adverb": ("ADJ", "ADVERB"),
    "adjective": ("ADJ",),
    "verb": ("VERB"),
    "adverb": ("ADVERB"),
    "prefix": ("ADJ", "ADVERB"),
    "conjunction": ("AUX"),
    "pronoun": ("INTJ", "NOUN"),
}


@dataclass
class GeneratedWord:
    word: str
    definition: str
    decoded: Optional[str]
    decoded_tokens: Optional[List[int]]

    @classmethod
    def print_words(cls, words, f=sys.stdout):
        for word in words:
            print(word)
            word_str = [word.word]
            print(" ".join(word_str), file=f)
            print(f"\t{word.definition}{' |n| '}", file=f)
            print("----------------", file=f)


@dataclass
class GeneratedWordCandidate:
    score: float
    candidate: GeneratedWord


def _len_range_overlap(x, y):
    start = max(x[0], y[0])
    end = min(x[-1], y[-1]) + 1
    return max(0, end - start)


def _split_range(splits, split_idx):
    splits_tensor = torch.tensor(splits)
    sum_splits = torch.cumsum(splits_tensor, 0)

    if sum_splits[-1] != 1.0:
        raise RuntimeError(f"Splits must sum to 1 (actual: {sum_splits[-1]})")
    elif split_idx >= len(sum_splits):
        raise RuntimeError(f"Invalid split index {split_idx} (must be less than {len(sum_splits)})")

    if split_idx == 0:
        start_range = 0.0
    else:
        start_range = sum_splits[split_idx - 1]

    end_range = sum_splits[split_idx]

    return (start_range, end_range)


def _cache_path(class_name, base_directory, filename, **keys):
    path = [class_name]
    for k, v in keys.items():
        if isinstance(v, str):
            path.append(f"{k}-{v}")
            continue

        try:
            path.append(f"{k}-{'-'.join(str(e) for e in iter(v))}")
            continue
        except TypeError:
            pass

        path.append(f"{k}-{str(v)}")

    path.append(filename)
    return os.path.join(base_directory, "__".join(path))


class TokenGroup(NamedTuple):
    separator: List[int] = []
    payload: List[int] = []
    remove_if_truncated: bool = False


def _join_and_truncate(
    max_len: int, begin_tokens: List[int], token_groups: List[TokenGroup], end_tokens: List[int], min_append_size=5,
):
    if len(begin_tokens) + len(end_tokens) > max_len:
        raise RuntimeError("Length is too small for required tokens")

    running_max_len = max_len - len(begin_tokens) - len(end_tokens)

    ret = [begin_tokens]

    for token_group in token_groups:
        if len(token_group.separator) + len(token_group.payload) > running_max_len:
            if token_group.remove_if_truncated:
                break

            if running_max_len - len(token_group.separator) - len(token_group.payload) < min_append_size:
                break

            ret.append(token_group.separator)
            running_max_len -= len(token_group.separator)
            ret.append(token_group.payload[:running_max_len])
            running_max_len = 0
            break
        else:
            ret.append(token_group.separator)
            ret.append(token_group.payload)
            running_max_len -= len(token_group.separator) + len(token_group.payload)

    ret.append(end_tokens)
    return list(itertools.chain.from_iterable(ret))


class SpecialTokens:
    BOS_TOKEN = "<|bod|>"
    EOS_TOKEN = "<|eod|>"
    PAD = "<|pad|>"

    DEFINITION_SEP = "<|bd|>"

    @classmethod
    def special_tokens_dict(cls):
        return {
            "bos_token": cls.BOS_TOKEN,
            "eos_token": cls.EOS_TOKEN,
            "pad_token": cls.PAD,
            "additional_special_tokens": [cls.DEFINITION_SEP,],
        }


@dataclass
class GenerationStats:
    num_iterations: int = 0

    num_items_considered: int = 0
    num_failed_match: int = 0
    num_seen_filtered: int = 0
    num_proper_noun_filtered: int = 0

    num_example_missing: int = 0

    num_user_filtered: int = 0
    num_returned: int = 0
    num_example_pos_match_failed: int = 0
    num_example_missing_title: int = 0
    num_short_definitions: int = 0
    wall_time: float = 0.0
    wall_stanza_time: float = 0.0

    def __str__(self):
        return f"iterations={self.num_iterations} time={self.wall_time} stanza_time={self.wall_stanza_time} | " + ", ".join(
            f"{k} {v / self.num_items_considered:.2f}@{v}"
            for k, v in (
                ("items_considered", self.num_items_considered),
                ("failed_match", self.num_failed_match),
                ("seen_filtered", self.num_seen_filtered),
                ("proper_noun_filtered", self.num_proper_noun_filtered),
                ("example_missing", self.num_example_missing),
                ("short_definitions", self.num_short_definitions),
                ("example_missing_title", self.num_example_missing_title),
                ("example_pos_match_failed", self.num_example_pos_match_failed),
                ("user_filtered", self.num_user_filtered),
                ("returned", self.num_returned),
            )
        )


class ParsedDictionaryDefinitionDataset(Dataset):
    @classmethod
    def _split_re(cls):
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<title>.+?)"
            f"{re.escape(SpecialTokens.DEFINITION_SEP)}(?P<definition>.+?)"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    @classmethod
    def approx_pos(cls, nlp, sentence, lookup_idx, lookup_len):
        start_end_re = re.compile(r"start_char=(\d+)\|end_char=(\d+)")
        doc = nlp(sentence)
        uposes = Counter()

        for sentence in doc.sentences:
            for word in sentence.words:
                m = start_end_re.match(word.misc)
                if not m:
                    raise RuntimeError("Unable to extract start and end positions!")
                start_char = int(m.group(1))
                end_char = int(m.group(2))
                uposes[word.upos] += _len_range_overlap((lookup_idx, lookup_idx + lookup_len - 1), (start_char, end_char - 1),)

        ((tag, _),) = uposes.most_common(1)
        return tag

    @classmethod
    def generate_words(
        cls,
        tokenizer,
        model,
        prefix=SpecialTokens.BOS_TOKEN,
        num=100,
        max_iterations=10,
        generation_args={},
        example_title_match=True,
        example_match_pos_pipeline=None,
        dedupe_titles=True,
        user_filter=None,
        filter_proper_nouns=False,
        use_custom_generate=True,
        min_definition_words=1,
    ):
        start = time.time()
        viable_candidates = []
        ret = []
        num_iteration = 0
        if isinstance(prefix, str):
            input = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
        else:
            input = torch.tensor([prefix], dtype=torch.long).to(model.device)

        split_re = cls._split_re()
        seen_titles = set()
        stats = GenerationStats()
        t = tqdm(total=num)
        while len(ret) < num and num_iteration < max_iterations:
            num_iteration += 1
            stats.num_iterations += 1
            if not use_custom_generate:
                generated = model.generate(
                    input, pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, **generation_args,
                )
            else:

                def partial_generation_transform(input_ids, tokens_to_add):
                    return tokens_to_add

                generated = custom_modeling_utils.custom_generate(
                    model,
                    input,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    partial_generation_transform=partial_generation_transform,
                    **generation_args,
                )

            for i in range(generated.size()[0]):
                if len(ret) >= num:
                    break
                viable_candidates = viable_candidates[:1000]

                stats.num_items_considered += 1
                sentence_tokens = generated[i, :].tolist()
                decoded = tokenizer.decode(sentence_tokens)

                m = split_re.match(decoded)
                if not m:
                    stats.num_failed_match += 1
                    continue

                title = m.group("title")
                definition = m.group("definition")

                generated_word = GeneratedWord(
                    word=title and title.strip(), definition=definition and definition.strip(), decoded=decoded, decoded_tokens=sentence_tokens,
                )

                if dedupe_titles and title.strip().lower() in seen_titles:
                    stats.num_seen_filtered += 1
                    continue

                if len(definition.split()) < min_definition_words:
                    stats.num_short_definitions += 1
                    viable_candidates.append(GeneratedWordCandidate(0.2, generated_word))
                    continue

                if user_filter and not user_filter(generated_word):
                    stats.num_user_filtered += 1
                    continue
                else:
                    t.update()
                    ret.append(generated_word)
                    seen_titles.add(generated_word.word.lower())

        stats.num_returned = len(ret)
        stats.viable_candidates = viable_candidates
        stats.wall_time = time.time() - start
        return ret[:num], stats

    def _make_examples(self, tokenizer, entry: dictionary_definition.Entry):
        examples = []
        token_groups = []
        token_groups.append(TokenGroup(separator=[], payload=tokenizer.encode(entry.word)))

        token_groups.append(TokenGroup(separator=self.definition_sep_ids, payload=tokenizer.encode(entry.text_body.rstrip(". ")),))

        example = _join_and_truncate(max_len=self.max_len, begin_tokens=self.bos_token_ids, end_tokens=self.eos_token_ids, token_groups=token_groups,)

        assert len(example) <= self.max_len, f"Example should be less than max length: {len(example)} Vs. {self.max_len}"

        examples.append(example)
        return examples

    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0,
    ):
        self.max_len = min(tokenizer.max_len_single_sentence, args.block_size)
        self.bos_token_ids = tokenizer.encode(SpecialTokens.BOS_TOKEN)
        self.eos_token_ids = tokenizer.encode(SpecialTokens.EOS_TOKEN)
        self.definition_sep_ids = tokenizer.encode(SpecialTokens.DEFINITION_SEP)

        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = _cache_path(
            self.__class__.__name__, directory, filename, model_type=args.model_type, splits=splits, split_idx=split_idx, max_len=self.max_len,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached filezzzz %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info(f"Loaded {len(self.examples)} features")
        else:
            logger.info(
                f"Cache at {cached_features_file} not found... creating features from dataset file at %s", directory,
            )

            self.examples = []
            split_range = _split_range(splits, split_idx)

            with open(file_path, "rb") as f:
                entries = pickle.load(f)

            for entry in entries:
                self.examples.extend(self._make_examples(tokenizer, entry))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
