from typing import List, Optional
import torch

class PadFunction(object):
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch):
        return self._pad_fn(batch)

    # @profile
    def merge(self, sequences, pad_size:Optional[int]=None, device:str="cpu"):
        lengths = [len(seq) for seq in sequences]
        if pad_size is None:
            pad_size = max(lengths)
        padded_seqs = torch.full((len(lengths), pad_size), self.pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # @profile
    def make_mask(self, inputs, inputs_length: List[int], device:str="cpu"):
        max_len = inputs.shape[1]
        inputs_mask = torch.arange(max_len, device=device).expand(len(inputs_length), max_len) < torch.tensor(inputs_length, device=device).unsqueeze(1)
        return inputs_mask

    # @profile
    def _pad_fn(self, batch):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        # batch.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = tuple(zip(*batch))
        device = src_seqs[0].device

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        # pad_size = max([len(seq) for seq in src_seqs] + [len(seq) for seq in trg_seqs])
        pad_size = None
        src_seqs, src_lengths = self.merge(src_seqs, pad_size, device=device)
        trg_seqs, trg_lengths = self.merge(trg_seqs, pad_size, device=device)

        source_tokens = {
            'token_ids': src_seqs,
            'mask': self.make_mask(src_seqs, src_lengths, device=device)
        }

        target_tokens = {
            'token_ids': trg_seqs,
            'mask': self.make_mask(trg_seqs, trg_lengths, device=device)
        }
        return source_tokens, target_tokens

