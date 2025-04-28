import torch
import torch.nn.functional as F

class MaskedLMModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_probs(self, sentences_with_blank, candidate_dict):
        # Tokenize all masked sentences in batch
        inputs = self.tokenizer(
            sentences_with_blank, return_tensors="pt", padding=True, truncation=True
        )

        # Identify mask positions in batch
        mask_token_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )

        # Forward pass in parallel
        with torch.no_grad():
            logits = self.model(**inputs).logits

        word_probabilities = {}
        for i, sentence in enumerate(sentences_with_blank):
            mask_pos = mask_token_indices[1][i].item()  # Get mask index
            mask_logits = logits[i, mask_pos, :]  # Get logits for mask position
            candidates = candidate_dict[sentence]
            word_ids = self.tokenizer.convert_tokens_to_ids(candidates)
            word_probs = F.softmax(mask_logits, dim=-1)[word_ids].tolist()

            for j, word in enumerate(candidates):
                word_probabilities[(sentence, word)] = word_probs[j]

        return word_probabilities
