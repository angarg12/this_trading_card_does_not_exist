import torch
import logging
import stanza
import datasets as datasets
import modeling as modeling
from transformers import AutoModelWithLMHead, AutoTokenizer

logger = logging.getLogger(__name__)


class WordGenerator:
    def __init__(
        self, forward_model_path, quantize=False, device=None,
    ):
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        stanza.download("en")
        self.stanza_pos_pipeline = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos", use_gpu=("cpu" not in self.device.type),)

        logger.info(f"Using device {self.device}")

        logger.info("Loading GPT2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
        logger.info("Loaded tokenizer")

        ml = modeling.load_model
        if quantize:
            ml = modeling.load_quantized_model
            logger.info(f"Peforming quantization on models")

        logger.info(f"Loading forward model from {forward_model_path}")
        self.forward_model = ml(AutoModelWithLMHead, forward_model_path).to(self.device)
        logger.info("Loaded forward model")

        self.approx_max_length = 250

    def generate_word(self, user_filter=None):
        expanded, stats = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.forward_model,
            num=1,
            max_iterations=5,
            generation_args=dict(top_k=300, num_return_sequences=10, max_length=self.approx_max_length, do_sample=True,),
            example_match_pos_pipeline=self.stanza_pos_pipeline,
            user_filter=user_filter,
            dedupe_titles=True,
            filter_proper_nouns=True,
            use_custom_generate=True,
        )
        print(expanded)
        print(stats)
        return expanded[0] if expanded else None
