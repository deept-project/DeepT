import pytorch_lightning as pl
import torch
import transformers

from data import TranslationDataset
from model import BartForMaskedLM
import torch
import tqdm

from translate import GreedySearch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PadFunction(object):
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        return self._pad_fn(batch)

    def merge(self, sequences, pad_size=None):
        lengths = [len(seq) for seq in sequences]
        if pad_size is None:
            pad_size = max(lengths)
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
        # pad_size = max([len(seq) for seq in src_seqs] + [len(seq) for seq in trg_seqs])
        pad_size=None
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
    tokenizer = transformers.BertTokenizerFast('./vocab/vocab.txt')
    setattr(tokenizer, "_bos_token", '[CLS]')
    setattr(tokenizer, "_eos_token", '[SEP]')

    debug=True
    if not debug:
        dataset = TranslationDataset(
            'data/train.en', 'data/train.zh', tokenizer=tokenizer
        )
    else:
        dataset = TranslationDataset(
            'data/debug_mini.en', 'data/debug_mini.zh', tokenizer=tokenizer
        )
    pad_fn_object = PadFunction(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=2, collate_fn=pad_fn_object)

    # init model
    model = BartForMaskedLM(
        config={
            'vocab_size': tokenizer.vocab_size,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
        }
    )

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(gpus=[0], accelerator='ddp', max_epochs=3, checkpoint_callback=False)
    trainer.fit(model, train_loader)

    inputs = tokenizer(["One against 500."], max_length=1024, return_tensors='pt') # , padding="max_length", truncation=True

    # Generate Summary
    greedy_search = GreedySearch(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        min_length=1,
        max_length=1024)

    input = inputs['input_ids']
    translation_ids = []
    labels = []
    labels.append(tokenizer.bos_token_id)
    for i in tqdm.tqdm(iterable=range(len(input[0]))):
        labels_tensor = torch.tensor(labels)
        output = model((input, [len(input[0])], torch.unsqueeze(labels_tensor, 0), [len(labels)]))[0,:,:]
        cur_output = output[i,:]
        max_prob, max_ids = torch.max(cur_output, dim=0)

        # show 
        show_max_prob, show_max_ids = torch.max(output, dim=1)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in show_max_ids])

        translation_ids.append(max_ids.item())
        labels.append(max_ids.item())
    print(translation_ids)

    # translation_ids = greedy_search.search(output)
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translation_ids])
