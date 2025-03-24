import torch
import torch.nn.functional as F


class MaskedLMModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_probs(self, sentence_with_blank, candidates):
        input_ids = self.tokenizer.encode(sentence_with_blank, return_tensors="pt")
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[
            1
        ].item()
        with torch.no_grad():
            logits = self.model(input_ids).logits
        mask_token_logits = logits[0, mask_token_index, :]
        word_ids = self.tokenizer.convert_tokens_to_ids(candidates)
        word_probs = F.softmax(mask_token_logits, dim=-1)[word_ids].tolist()
        return zip(candidates, word_probs)


# import torch
# import torch.nn.functional as F


# class MaskedLMModel:
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def generate_probs(self, sentences_with_blank, candidate_dict):
#         # Tokenize all sentences in batch
#         inputs = self.tokenizer(
#             sentences_with_blank, return_tensors="pt", padding=True, truncation=True
#         )

#         print(f"inputs: {inputs}")

#         # Identify mask positions in batch
#         mask_token_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(
#             as_tuple=True
#         )

#         # Perform forward pass in parallel
#         with torch.no_grad():
#             logits = self.model(
#                 **inputs
#             ).logits  # Shape: (batch_size, seq_len, vocab_size)

#         word_probabilities = {}
#         for i, sentence in enumerate(sentences_with_blank):
#             mask_pos = mask_token_indices[1][
#                 i
#             ].item()  # Get mask index for this sentence
#             mask_logits = logits[i, mask_pos, :]  # Extract logits for mask position

#             candidates = candidate_dict[sentence]
#             word_ids = self.tokenizer.convert_tokens_to_ids(candidates)
#             word_probs = F.softmax(mask_logits, dim=-1)[word_ids].tolist()

#             # Store probabilities for each word
#             for j, word in enumerate(candidates):
#                 word_probabilities[(sentence, word)] = word_probs[j]

#         return word_probabilities
