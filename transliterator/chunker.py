from .utils import get_key_by_value, calculate_number_of_bert_calls


class Chunker:
    def __init__(self, max_bert_call, overlap):
        self.max_bert_call = max_bert_call
        self.overlap = overlap

    def chunk_sentence(self, mask_sentence, candidate_words):
        mask_sentence = mask_sentence.split()
        mask_indexes = [
            index for index, word in enumerate(mask_sentence) if word == "[MASK]"
        ]

        # Map candidate_words indexes with mask_indexes
        index_map = {}
        for i in range(len(mask_indexes)):
            index_map[i] = mask_indexes[i]

        mask_count = 0
        sentences = []
        candidates = []
        sub_sentence = []
        sub_candidates = []

        for index, word in enumerate(mask_sentence):
            if word != "[MASK]":
                sub_sentence.append(word)
                if index == len(mask_sentence) - 1:
                    sentences.append(" ".join(sub_sentence))
                    candidates.append(sub_candidates)
            else:
                sub_candidates.append(
                    candidate_words[get_key_by_value(index_map, index)]
                )
                sub_sentence.append(word)
                mask_count += 1

                if len(sub_candidates) >= (self.overlap + 1):

                    if (
                        calculate_number_of_bert_calls(sub_candidates)
                    ) > self.max_bert_call:
                        try:
                            sub_sentence = (
                                sub_sentence
                                + mask_sentence[
                                    mask_indexes[mask_count - 1]
                                    + 1 : mask_indexes[mask_count]
                                ]
                            )
                        except:
                            sub_sentence = (
                                sub_sentence
                                + mask_sentence[mask_indexes[mask_count - 1] + 1 :]
                            )

                        sentences.append(" ".join(sub_sentence))
                        candidates.append(sub_candidates)

                        sub_sentence = mask_sentence[
                            (mask_indexes[(mask_count - self.overlap) - 1])
                            + 1 : index
                            + 1
                        ]
                        sub_candidates = sub_candidates[-2:]

                    elif index == len(mask_sentence) - 1:
                        sentences.append(" ".join(sub_sentence))
                        candidates.append(sub_candidates)

        return sentences, candidates
