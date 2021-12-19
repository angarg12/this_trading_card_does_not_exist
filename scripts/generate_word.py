import sys

sys.path.append("./")
sys.path.append("./title_maker_pro")
from title_maker_pro.word_generator import WordGenerator
from title_maker_pro.datasets import GeneratedWord

word_generator = WordGenerator(device="cpu", forward_model_path="./models/dict_words", quantize=False,)

# a word from scratch:
GeneratedWord.print_words([word_generator.generate_word()])
