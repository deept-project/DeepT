
from typing import List
from deept.model.mbart import BartForMaskedLM
import pytorch_lightning as pl
import torch
import transformers
from deept.tranalate.translate import GreedySearch, BeamSearch, BeamSearchSlow
from deept.utils import PadFunction

import glob

device='cuda'

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def vector_to_text(tokenizer:transformers.tokenization_utils.PreTrainedTokenizer, vector):
    result = ''
    tokens = tokenizer.convert_ids_to_tokens(vector, skip_special_tokens=True)
    for each_token in tokens:
        if each_token == tokenizer.eos_token:
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

    tokenizer = transformers.MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # setattr(tokenizer, "_bos_token", '[CLS]')
    # setattr(tokenizer, "_eos_token", '[SEP]')

    pad_fn_object = PadFunction(tokenizer.pad_token_id)

    bos_token_id = tokenizer.lang_code_to_id["en_XX"]

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

    # tokenizer.lang_code_to_id["fr_XX"]

    # Generate Summary
    greedy_search = BeamSearchSlow(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        min_length=1,
        max_length=64,
        top_k=5
    )

    def predit_fn(source_inputs: List[torch.Tensor], states: List[torch.Tensor]):
        batch_size = len(source_inputs)

        batch = pad_fn_object(list(zip(source_inputs, states)))
        output = model(source_tokens=batch[0], target_tokens=batch[1])
        return output

    while True:
        text = input('请输入原文：')
        print("输入是：" + text)

        inputs = tokenizer([text.strip()], max_length=512, truncation=True, padding=True, return_tensors='pt')

        # texts = ["hello world", "Also, note that the copy mechanism is only applied to the raw dataset of source code tokens."]
        # inputs = tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors='pt')

        source_inputs = inputs['input_ids'].to(device)
        batch_size = source_inputs.size(0)
        init_states = torch.full((batch_size, 1), bos_token_id).to(device)
        translation_ids = greedy_search.search(source_inputs, init_states, predit_fn)

        # translation_ids = greedy_search.search(output)
        
        for i in range(batch_size):
            translation = vector_to_text(tokenizer, translation_ids[i])
            print(f'{i+1}/{batch_size}:\n{translation}')
