from transliterator.transliteration import Transliterator
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model and tokenizer
MODEL_NAME = "Sameera827/sinhala-bert-dakshina_new01"
TOKENIZER_NAME = "Ransaka/sinhala-bert-medium-v2"

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
# model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# Load the tokenizer and model from the local directory
model_directory = "models"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForMaskedLM.from_pretrained(model_directory)

# dictionary
DICTIONARY_PATH = "data/dictionary.txt"

# Initialize Transliterator
transliterator = Transliterator(
    dictionary_path=DICTIONARY_PATH, tokenizer=tokenizer, model=model
)

# Transliterate input
singlish_sentence = input("Enter Singlish Sentence: ").strip()
sinhala_sentence = transliterator.generate_sinhala(singlish_sentence)
print("Sinhala Output:", sinhala_sentence)
