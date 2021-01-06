
from model import BartForMaskedLM
import pytorch_lightning as pl
import torch
import transformers
from translate import GreedySearch


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
            'token_ids': src_seqs.to('cuda'),
            'mask': self.make_mask(src_seqs, src_lengths).to('cuda'),
        }

        target_tokens = {
            'token_ids': trg_seqs.to('cuda'),
            'mask': self.make_mask(trg_seqs, trg_lengths).to('cuda'),
        }
        return source_tokens, target_tokens

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

if __name__ == "__main__":
    checkpoint_path = 'tb_logs/translation/version_1/checkpoints/epoch=22-step=43535.ckpt'
    # onnx_filepath = 'model.onnx'

    tokenizer = transformers.BertTokenizerFast('./vocab/vocab.txt')
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

    model = model.to('cuda')

    model.eval()

    # Generate Summary
    greedy_search = GreedySearch(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        min_length=1,
        max_length=512)


    def predit_fn(source_inputs: torch.Tensor, states: torch.Tensor):
        batch_size = source_inputs.size(0)
        source_list = [source_inputs[i,:] for i in range(batch_size)]
        state_list = [states[i,:] for i in range(batch_size)]

        batch = pad_fn_object(list(zip(source_list, state_list)))
        output = model(source_tokens=batch[0], target_tokens=batch[1])
        return output

    while True:
        text = input('请输入原文：')
        print("输入是：" + text)
        
        inputs = tokenizer([text.strip()], max_length=512, truncation=True, return_tensors='pt') # , padding="max_length", truncation=True


        source_inputs = inputs['input_ids'].to('cuda')
        batch_size = source_inputs.size(0)
        init_states = torch.full((batch_size, 1), tokenizer.bos_token_id).to('cuda')
        translation_ids = greedy_search.search(source_inputs, init_states, predit_fn)

        # translation_ids = greedy_search.search(output)
        tokens = tokenizer.convert_ids_to_tokens(translation_ids[0], skip_special_tokens=False)
        new_tokens = []
        for each_token in tokens:
            if each_token == '[SEP]':
                break
            new_tokens.append(each_token)
        
        result = ''
        for each_token in new_tokens:
            if any([is_chinese(c) for c in each_token]):
                result += each_token
            else:
                if not each_token.startswith('##'):
                    result += ' ' + each_token
                else:
                    result += each_token[2:]

        print(result)
