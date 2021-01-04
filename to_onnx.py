
from model import BartForMaskedLM
import pytorch_lightning as pl
import torch
import transformers

if __name__ == "__main__":
    checkpoint_path = 'lightning_logs/version_0/checkpoints/epoch=1-step=148291.ckpt'
    onnx_filepath = 'model.onnx'

    tokenizer = transformers.BertTokenizerFast('./vocab/vocab.txt')
    setattr(tokenizer, "_bos_token", '[CLS]')
    setattr(tokenizer, "_eos_token", '[SEP]')

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
    encoder_input_sample = {
        'token_ids': torch.zeros(1, 512, dtype=torch.int64),
        'mask': torch.ones(1, 512, dtype=torch.int64)
    }
    decoder_input_sample = {
        'token_ids': torch.zeros(1, 512, dtype=torch.int64),
        'mask': torch.ones(1, 512, dtype=torch.int64)
    }

    input = [encoder_input_sample, decoder_input_sample]

    # model.to_onnx(onnx_filepath, input, export_params=True)
    torch.onnx.export(model, [encoder_input_sample, decoder_input_sample], 'translate.onnx', verbose=True)