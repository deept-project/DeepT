import torch

class PadFunction(object):
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch):
        return self._pad_fn(batch)

    # @profile
    def merge(self, sequences, pad_size=None):
        lengths = [len(seq) for seq in sequences]
        if pad_size is None:
            pad_size = max(lengths)
        padded_seqs = torch.full(
            (len(lengths), pad_size), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # @profile
    def make_mask(self, inputs, inputs_length):
        max_len = inputs.shape[1]
        inputs_mask = torch.arange(max_len).expand(len(inputs_length), max_len) < torch.tensor(inputs_length).unsqueeze(1)
        return inputs_mask

    # @profile
    def _pad_fn(self, batch):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        # batch.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = tuple(zip(*batch))

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        # pad_size = max([len(seq) for seq in src_seqs] + [len(seq) for seq in trg_seqs])
        pad_size = None
        src_seqs, src_lengths = self.merge(src_seqs, pad_size)
        trg_seqs, trg_lengths = self.merge(trg_seqs, pad_size)

        source_tokens = {
            'token_ids': src_seqs,
            'mask': self.make_mask(src_seqs, src_lengths)
        }

        target_tokens = {
            'token_ids': trg_seqs,
            'mask': self.make_mask(trg_seqs, trg_lengths)
        }
        return source_tokens, target_tokens

