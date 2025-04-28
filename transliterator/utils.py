import re
from collections import defaultdict
from functools import reduce
import itertools

# Get the candidate words and mask the sentence accordingly
def process_sentence(new_sinhala_words):
    sinhala_words_processing = new_sinhala_words.copy()
    candidate_words = []

    # Use list comprehension and handle masking
    masked_sentence = []
    for words in sinhala_words_processing:
        if len(words) > 1:
            candidate_words.append(words)
            masked_sentence.append("[MASK]")
        else:
            masked_sentence.append(" ".join(words))

    # Use 'join' to build the sentence efficiently
    masked_sentence = " ".join(masked_sentence)
    return masked_sentence, candidate_words


# generate the sentence combinations with candidate words
def generate_sentences_with_all_combinations(template, candidates):
    # Split the template by spaces
    template_parts = template.split()

    # Identify the indices of blanks
    blank_indices = [i for i, word in enumerate(template_parts) if word == "[MASK]"]

    # Generate all combinations of candidate words for blanks
    all_combinations = list(itertools.product(*candidates))

    # Generate all full sentences
    all_sentences = []
    for combo in all_combinations:
        filled_sentence = template_parts.copy()  # Start with the template
        for i, word in zip(blank_indices, combo):  # Replace blanks with the combo words
            filled_sentence[i] = word
        all_sentences.append(" ".join(filled_sentence))  # Join words into a sentence
    return all_sentences


def fill_blanks(blanked_sentence, combination):
    # Split the sentence by the '[BLANK]' placeholder
    parts = blanked_sentence.split("[MASK]")
    # Rebuild the sentence by joining the parts and filling in the blanks
    filled_sentence = "".join(
        f"{part}{word}" for part, word in zip(parts, combination + ("",))
    )
    return filled_sentence


# generate the sentence combinations with candidate words
def generate_sentences_with_all_combinations(template, candidates):
    # Split the template by spaces
    template_parts = template.split()

    # Identify the indices of blanks
    blank_indices = [i for i, word in enumerate(template_parts) if word == "[MASK]"]

    # Generate all combinations of candidate words for blanks
    all_combinations = list(itertools.product(*candidates))

    # Generate all full sentences
    all_sentences = []
    for combo in all_combinations:
        filled_sentence = template_parts.copy()  # Start with the template
        for i, word in zip(blank_indices, combo):  # Replace blanks with the combo words
            filled_sentence[i] = word
        all_sentences.append(" ".join(filled_sentence))  # Join words into a sentence

    return all_sentences


# Function to calculate the product of values for a given sentence
def calculate_product(sentence, word_probabilities):
    product = 1
    found = False
    for key, value in word_probabilities.items():
        parts = key.split("--")

        if re.sub(r"\s+", " ", parts[2]).strip() == sentence:
            product *= value
            found = True
    return product if found else None


# Generate sentences with one blank
def generate_sentences_with_one_blank(word_combinations, mask_indexes, masked_sentence):
    one_blank_sentences = {}
    for comb in word_combinations:
        for index, mask_index in enumerate(mask_indexes):
            combination = list(comb)
            masking_word = combination[index]
            combination[index] = "[MASK]"
            one_blank_sentence = fill_blanks(masked_sentence, tuple(combination))

            if one_blank_sentence in one_blank_sentences.keys():
                one_blank_sentences[one_blank_sentence].append(masking_word)
            else:
                one_blank_sentences[one_blank_sentence] = [masking_word]
    return one_blank_sentences


def calculate_number_of_bert_calls(candidate_words):
    n = len(candidate_words)
    # If there's only one blank, there is only one possible sentence with that blank remaining
    if n == 1:
        return 1

    total_sentences = 0
    # Iterate over each blank
    for i in range(n):
        # Calculate the product of candidate words for all other blanks
        product = reduce(
            lambda x, y: x * y, (len(candidate_words[j]) for j in range(n) if j != i)
        )
        total_sentences += product
    return total_sentences


def get_key_by_value(dictionary, value):
    """
    This function searches a dictionary for a given value and returns the corresponding key.
    If the value is not found, it returns None.
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def numbering_masks_sentences(sentences):
    # Initialize the mask counter and a dictionary to keep track of mappings
    mask_counter = 1
    mask_map = {}
    previous_masks = []
    updated_sentences = []

    for i, sentence in enumerate(sentences):
        parts = sentence.split("[MASK]")
        labeled_sentence = parts[0]

        for j in range(1, len(parts)):
            # For the first two masks in a sentence (if not the first sentence), use previous masks
            if i > 0 and j <= 2:
                label = previous_masks[j - 1]
            else:
                # Otherwise, assign a new label if not already assigned
                mask_key = (i, j)
                if mask_key in mask_map:
                    label = mask_map[mask_key]
                else:
                    label = f"MASK{mask_counter}"
                    mask_map[mask_key] = label
                    mask_counter += 1

            labeled_sentence += f"[{label}]" + parts[j]
            # Save the last two labels to use in the next sentence for overlap
            if j >= len(parts) - 2:
                previous_masks.append(label)

        # Only retain the last two labels for overlap with the next sentence
        previous_masks = previous_masks[-2:]
        updated_sentences.append(labeled_sentence)
    return updated_sentences


# Replace [MASK] with numbered labels
def numbering_masks_sentence(sentence):
    masks = re.findall(r"\[MASK\]", sentence)
    for i, _ in enumerate(masks, 1):
        sentence = sentence.replace("[MASK]", f"[MASK{i}]", 1)
    return sentence


# Function to extract words for each mask
def find_mask_words(mask_sentences, labeled_sentences):
    mask_words = defaultdict(list)

    # Iterate over sentences with masks and corresponding labeled sentences
    for mask_sentence, labeled_sentence in zip(mask_sentences, labeled_sentences):
        mask_parts = re.findall(r"\[MASK\d+\]", mask_sentence)
        labeled_parts = labeled_sentence.split()

        # Iterate over mask parts and map them to corresponding words
        mask_index = 0
        for i, part in enumerate(mask_sentence.split()):
            if mask_index < len(mask_parts) and part == mask_parts[mask_index]:
                word = labeled_parts[i]
                mask_words[mask_parts[mask_index]].append(word)
                mask_index += 1
    # Remove duplicates and format the output
    mask_words = {mask: list(set(words)) for mask, words in mask_words.items()}
    return mask_words


# Function to replace MASKs and collect candidates
def replace_masks_and_collect_candidates(sentence, mask_words):
    candidates = []

    for mask, words in mask_words.items():
        if len(words) > 1:
            # Replace with [MASK] if there are multiple words
            sentence = sentence.replace(f"[{mask[1:-1]}]", "[MASK]")
            candidates.append(words)
        else:
            # Replace with the single word
            sentence = sentence.replace(f"[{mask[1:-1]}]", words[0])

    return sentence, candidates
