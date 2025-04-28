from .rule_based import RuleBasedTransliterator
from .model import MaskedLMModel
from .utils import (
    numbering_masks_sentence,
    find_mask_words,
    process_sentence,
    generate_sentences_with_one_blank,
    generate_sentences_with_all_combinations,
    calculate_product,
    numbering_masks_sentences,
    replace_masks_and_collect_candidates,
)
from .dictionary import TransliterationDictionary
import itertools
from .chunker import Chunker


class Transliterator:
    def __init__(self, dictionary_path, tokenizer, model):
        self.dictionary = TransliterationDictionary(dictionary_path)
        self.tokenizer = tokenizer
        self.model = MaskedLMModel(model, tokenizer)
        self.rule_based_transliterator = RuleBasedTransliterator()
        self.chunker = Chunker(max_bert_call=20, overlap=2)

    def get_sinhala_word(self, singlish_word):
        sinhala_word = self.dictionary.get(singlish_word)
        if sinhala_word == "Word not found":
            sinhala_word = [self.rule_based_transliterator.convert_text(singlish_word)]
        return sinhala_word

    # Split the input sentence and get the relevant native Sinhala words
    def get_sinhala_words(self, singlish_sentence):
        # Separate words by spaces and get corresponding Sinhala words
        singlish_words = singlish_sentence.split()
        sinhala_words = [self.get_sinhala_word(word) for word in singlish_words]
        return sinhala_words

    # Sinhala word suggesions for the given singlish word
    def get_sinhala_word_suggestions(self, singlish_word):
        sinhala_words = self.dictionary.get(singlish_word)
        if sinhala_words == "Word not found":
            sinhala_words = [self.rule_based_transliterator.convert_text(singlish_word)]
        else:
            sinhala_words += [
                self.rule_based_transliterator.convert_text(singlish_word)
            ]
        return sinhala_words

    # Remove words which are not in the BERT vocabulary
    def clean_words(self, sinhala_words):
        new_sinhala_words = []
        vocab = set(self.tokenizer.vocab)  # Convert vocab to a set for O(1) lookups

        for words in sinhala_words:
            if len(words) == 1:
                new_sinhala_words.append(words)
                continue
            clean_words = [word for word in words if word in vocab]

            if not clean_words:  # If no words are found in the vocab
                clean_words = [words[0]]
            new_sinhala_words.append(clean_words)
        return new_sinhala_words

    # Update the calling function
    def generate_probability_dict(self, one_blank_sentences, tokenizer):
        sentences_with_blank = list(
            one_blank_sentences.keys()
        )  # Extract all masked sentences

        # Get word probabilities in parallel
        word_probabilities = self.model.generate_probs(
            sentences_with_blank, one_blank_sentences
        )

        # Convert output into final dictionary format
        probability_dict = {}
        for (masked_sentence, word), prob in word_probabilities.items():
            full_sentence = masked_sentence.replace("[MASK]", word)
            sentence_key = f"{masked_sentence}--{word}--{full_sentence}"
            probability_dict[sentence_key] = prob
        return probability_dict

    # transliterating process
    def transliterate(self, masked_sentence, candidates):
        word_combinations = list(itertools.product(*candidates))

        print("\n\nMasked sentence: ", masked_sentence)
        print("Candidates: ", candidates)

        print("\n\nWord combinations: ", len(word_combinations))

        word_list = masked_sentence.split()
        mask_indexes = [
            index for index, word in enumerate(word_list) if word == "[MASK]"
        ]
        # generate sentences with one blanks including possible candidate words for the blank
        one_blank_sentences = generate_sentences_with_one_blank(
            word_combinations, mask_indexes, masked_sentence
        )

        print("One blank sentences: ", len(one_blank_sentences))

        word_probabilities = self.generate_probability_dict(
            one_blank_sentences, self.tokenizer
        )
        print("Word probabilities: ", len(word_probabilities))
        # print("Word probabilities: ", word_probabilities)
        full_sentences = generate_sentences_with_all_combinations(
            masked_sentence, candidates
        )

        print("Full sentences: ", len(full_sentences))
        # print("Full sentences[0]: ", full_sentences[0])
        # Find the sentence with the highest product
        max_product = None
        max_sentence = None

        for sentence in full_sentences:
            product = calculate_product(sentence, word_probabilities)
            if product is not None and (max_product is None or product > max_product):
                max_product = product
                max_sentence = sentence
        print("Max sentence: ", max_sentence)
        return max_sentence

    def generate_sinhala(self, singlish_sentence):
        # generate sinhala word suggestions for one word input
        if len(singlish_sentence.split()) == 1:
            sinhala_word_suggestions = self.get_sinhala_word_suggestions(
                singlish_sentence
            )
            return sinhala_word_suggestions

        sinhala_words = self.get_sinhala_words(singlish_sentence)
        filtered_sinhala_words = self.clean_words(sinhala_words)
        masked_sentence, candidates = process_sentence(filtered_sinhala_words)

        while True:
            if len(candidates) == 0:
                return masked_sentence
            else:
                if len(candidates) <= 3:
                    output = self.transliterate(masked_sentence, candidates)
                    return output
                else:

                    print("\n\nMasked sentence: ", masked_sentence)
                    print("Candidates: ", candidates)

                    sentences, candidates = self.chunker.chunk_sentence(
                        masked_sentence, candidates
                    )

                    print("\n\nChunked sentences: ", sentences)
                    print("Chunked candidates: ", candidates)
        

                    # Numbering masks
                    numbered_input_sentence = numbering_masks_sentence(masked_sentence)
                    numbered_sentences = numbering_masks_sentences(sentences)

                    filled_sentences = [
                        self.transliterate(sentences[i], candidates[i])
                        for i in range(len(sentences))
                    ]

                    # Find the words for each mask
                    mask_words = find_mask_words(numbered_sentences, filled_sentences)

                    # Replace the MASKs and collect candidates
                    masked_sentence, candidates = replace_masks_and_collect_candidates(
                        numbered_input_sentence, mask_words
                    )
