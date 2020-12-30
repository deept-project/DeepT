import pytorch_lightning as pl
import torch
import transformers

from data import TranslationDataset
from model import BartForMaskedLM
import torch

def pad_fn(batch):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*batch)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths

if __name__ == "__main__":
    dataset = TranslationDataset('data/train.en', 'data/train.zh')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=pad_fn)

    # init model
    model = BartForMaskedLM()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer()
    trainer.fit(model, train_loader)
