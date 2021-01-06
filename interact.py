
from model import BartForMaskedLM
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers
from allennlp.nn.beam_search import BeamSearch

from typing import Dict, Tuple, List


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
            'token_ids': src_seqs,
            'mask': self.make_mask(src_seqs, src_lengths),
            'length': src_lengths
        }

        target_tokens = {
            'token_ids': trg_seqs,
            'mask': self.make_mask(trg_seqs, trg_lengths),
            'length': trg_lengths
        }
        return source_tokens, target_tokens

if __name__ == "__main__":
    checkpoint_path = 'tb_logs/translation/version_1/checkpoints/epoch=29-step=57080.ckpt'

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

    model.eval()

    # Generate Summary
    beam_search = BeamSearch(
        end_index=tokenizer.eos_token_id,
        max_steps=512,
        beam_size=10,

        )

    def _dict_to_decoder_cache(cache_dict):
        decoder_cache = []
        for key, cache_value in cache_dict.items():
            # Split key and extract index and dict keys
            layer_idx, attention_name, tensor_name = key
            # Extend decoder_cache to fit layer_idx + 1 layers
            decoder_cache = decoder_cache + [{} for _ in range(layer_idx + 1 - len(decoder_cache))]
            cache = decoder_cache[layer_idx]
            if attention_name not in cache:
                cache[attention_name] = {}
            assert tensor_name not in cache[attention_name]
            cache[attention_name][tensor_name] = cache_value
        return decoder_cache

    def take_step(last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)

        batch_size = last_predictions.size(0)

        # Only the last predictions are needed for the decoder, but we need to pad the decoder ids
        # to not mess up the positional embeddings in the decoder.
        padding_size = 0
        if step > 0:
            padding_size = step + 1
            padding = torch.full(
                (batch_size, padding_size),
                tokenizer.pad_token_id,
                dtype=last_predictions.dtype,
                device=last_predictions.device,
            )
            last_predictions = torch.cat([padding, last_predictions], dim=-1)

        decoder_cache = None
        decoder_cache_dict = {
            k: (state[k].contiguous() if state[k] is not None else None)
            for k in state
            if k not in {"input_ids", "input_mask", "encoder_states"}
        }
        if len(decoder_cache_dict) != 0:
            decoder_cache = _dict_to_decoder_cache(decoder_cache_dict)

        log_probabilities = None
        for i in range(padding_size, last_predictions.shape[1]):
            encoder_outputs = (
                (state["encoder_states"],) if state["encoder_states"] is not None else None
            )
            outputs = model(
                source_tokens = {
                    'token_ids': state["input_ids"],
                    'mask': state["input_mask"],
                },
                target_tokens = {
                    'token_ids': last_predictions[:, : i + 1],
                    'mask': state["input_mask"],
                }
            )

            decoder_log_probabilities = F.log_softmax(outputs[:, 0], dim=-1)

            if log_probabilities is None:
                log_probabilities = decoder_log_probabilities
            else:
                idx = last_predictions[:, i].view(-1, 1)
                log_probabilities = decoder_log_probabilities + log_probabilities.gather(
                    dim=-1, index=idx
                )

            decoder_cache = outputs[1]

            state["encoder_states"] = outputs[2]

        if decoder_cache is not None:
            decoder_cache_dict = _decoder_cache_to_dict(decoder_cache)
            state.update(decoder_cache_dict)

        return log_probabilities, state


        batch_size = source_inputs.size(0)
        source_list = [source_inputs[i,:] for i in range(batch_size)]
        state_list = [states[i,:] for i in range(batch_size)]

        batch = pad_fn_object(list(zip(source_list, state_list)))
        output = model(batch)
        return output

    while True:
        text = input('请输入原文：')
        print("输入是：" + text)
        
        inputs = tokenizer([text.strip()], max_length=512, truncation=True, return_tensors='pt') # , padding="max_length", truncation=True


        source_inputs = inputs['input_ids']
        batch_size = source_inputs.size(0)
        
        initial_decoder_ids = torch.tensor(
                [[tokenizer.bos_token_id]],
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            ).repeat(batch_size, 1)

        inital_state = {
            "input_ids": source_inputs,
            "input_mask": torch.ones(1, source_inputs.size(1)),
            "encoder_states": None,
        }


        translation_ids = beam_search.search(initial_decoder_ids, inital_state, take_step)

        # translation_ids = greedy_search.search(output)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translation_ids[0]])
