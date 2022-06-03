
from typing import List
from model import BartForMaskedLM
import pytorch_lightning as pl
import torch
import transformers
from translate import GreedySearch, BeamSearch, BeamSearchSlow

import glob

device='cpu'

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

    def make_mask(self, inputs, inputs_length):
        inputs_mask = torch.zeros_like(inputs)
        for i in range(inputs_mask.size(0)):
            inputs_mask[i,:inputs_length[i]] = 1
        return inputs_mask

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

        source_tokens = {
            'token_ids': src_seqs.to(device),
            'mask': self.make_mask(src_seqs, src_lengths).to(device),
        }

        target_tokens = {
            'token_ids': trg_seqs.to(device),
            'mask': self.make_mask(trg_seqs, trg_lengths).to(device),
        }
        return source_tokens, target_tokens

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def vector_to_text(tokenizer, vector):
    result = ''
    tokens = tokenizer.convert_ids_to_tokens(vector, skip_special_tokens=True)
    for each_token in tokens:
        if each_token == '[SEP]':
            break
        if not each_token.startswith('##'):
            result += ' ' + each_token
        else:
            result += each_token[2:]
    return result

if __name__ == "__main__":
    ckpts = glob.glob('./tb_logs/translation/version_*/checkpoints/*.ckpt')
    ckpts = sorted(ckpts)
    checkpoint_path = ckpts[-1]
    print(f'Loading {checkpoint_path}...')
    # onnx_filepath = 'model.onnx'

    tokenizer = transformers.BertTokenizer('./vocab/vocab.txt', do_basic_tokenize=False)
    setattr(tokenizer, "_bos_token", '[CLS]')
    setattr(tokenizer, "_eos_token", '[SEP]')

    pad_fn_object = PadFunction(tokenizer.pad_token_id)

    model = BartForMaskedLM.load_from_checkpoint(
        checkpoint_path,
        config={
            'vocab_size': tokenizer.vocab_size,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
        }
        )

    model = model.to(device)
    model.eval()

    # Generate Summary
    greedy_search = BeamSearchSlow(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        min_length=1,
        max_length=512)

    def predit_fn(source_inputs: List[torch.Tensor], states: List[torch.Tensor]):
        batch_size = len(source_inputs)

        batch = pad_fn_object(list(zip(source_inputs, states)))
        output = model(source_tokens=batch[0], target_tokens=batch[1])
        return output

    while True:
        # text = input('请输入原文：')
        # print("输入是：" + text)

        # inputs = tokenizer([text.strip()], max_length=512, truncation=True, padding=True, return_tensors='pt')

        inputs = tokenizer(["hello", "print"], max_length=512, truncation=True, padding=True, return_tensors='pt')

        source_inputs = inputs['input_ids']
        batch_size = source_inputs.size(0)
        init_states = torch.full((batch_size, 1), tokenizer.bos_token_id).to(device)
        translation_ids = greedy_search.search(source_inputs, init_states, predit_fn)

        # translation_ids = greedy_search.search(output)
        
        for i in range(batch_size):
            translation = vector_to_text(tokenizer, translation_ids[i])
            print(f'{i+1}/{batch_size}:\n{translation}')
        break
