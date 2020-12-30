import pytorch_lightning as pl
import torch
import transformers

from data import TranslationDataset
from model import BartForMaskedLM
import torch

class PadFunction(object):
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        return self._pad_fn(batch)

    def merge(self, sequences, pad_size):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.full((len(sequences), pad_size), self.pad_id).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def _pad_fn(self, batch):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = zip(*batch)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        pad_size = max([len(seq) for seq in src_seqs] + [len(seq) for seq in trg_seqs])
        src_seqs, src_lengths = self.merge(src_seqs, pad_size)
        trg_seqs, trg_lengths = self.merge(trg_seqs, pad_size)

        return src_seqs, src_lengths, trg_seqs, trg_lengths

def pad_fn(batch):
    def merge(sequences, pad_size):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), pad_size).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*batch)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    pad_size = max([len(seq) for seq in src_seqs] + [len(seq) for seq in trg_seqs])
    src_seqs, src_lengths = merge(src_seqs, pad_size)
    trg_seqs, trg_lengths = merge(trg_seqs, pad_size)

    return src_seqs, src_lengths, trg_seqs, trg_lengths

if __name__ == "__main__":
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    dataset = TranslationDataset('data/debug.en', 'data/debug.zh', tokenizer=tokenizer)
    pad_fn_object = PadFunction(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=pad_fn_object)

    # init model
    model = BartForMaskedLM()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(gpus=[1], max_epochs=10)
    trainer.fit(model, train_loader)

    inputs = tokenizer(["Pair of winner"], max_length=1024, return_tensors='pt') # , padding="max_length", truncation=True

    # Generate Summary
    output = model(inputs['input_ids'])[0]
    max, translation_ids = torch.max(output, dim=1)
    print(translation_ids)
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translation_ids])
