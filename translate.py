import torch
from typing import Tuple

class DecodeStrategy(object):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length):
        self.pad_id=pad_id
        self.bos_id=bos_id
        self.eos_id=eos_id
        self.min_length=min_length
        self.max_length=max_length

class BeamSearch(DecodeStrategy):
    pass


class GreedySearch(DecodeStrategy):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length):
        super(GreedySearch, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)

    def search(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # predictions: (batch_size, seq_len, vocab_size)
        max_prob, max_ids = torch.max(predictions, dim=2)

        return max_ids
