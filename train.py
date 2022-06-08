from typing import List
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import transformers

from deept.model.mbart import BartForMaskedLM, PadFunction
import torch
import tqdm
import numpy as np

from deept.decode.greedy_strategy import GreedySearch

import random
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # params
    random_seed = 0

    # debug_dataset = TranslationDataset('data/debug_mini.en', 'data/debug_mini.zh', tokenizer=tokenizer)

    # dataset = debug_dataset

    # init model
    model = BartForMaskedLM()

    train_dataloader = model.train_dataloader()
    valid_dataloader = model.val_dataloader()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    gpu_stats = GPUStatsMonitor()
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    logger = TensorBoardLogger('tb_logs', name='translation')
    trainer = pl.Trainer(
        gpus=[0],
        # num_nodes=1,
        max_epochs=500,
        # accelerator='ddp',
        # plugins='ddp_sharded',
        amp_backend='native',
        precision=16,
        auto_scale_batch_size=None,
        log_gpu_memory=None,
        log_every_n_steps=10,
        flush_logs_every_n_steps=100,
        progress_bar_refresh_rate=1,
        val_check_interval=1.0,
        accumulate_grad_batches=4,
        sync_batchnorm=True,
        checkpoint_callback=True,
        resume_from_checkpoint="tb_logs/translation/version_1/checkpoints/epoch=18-step=325013.ckpt",
        logger=logger,
        callbacks=[early_stop_callback],
        # profiler="simple",
    )
    # find the batch size
    # trainer.tune(model)
    trainer.fit(model, train_dataloader, valid_dataloader)


    # infer
    tokenizer = transformers.MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # setattr(tokenizer, "_bos_token", '[CLS]')
    # setattr(tokenizer, "_eos_token", '[SEP]')

    inputs = tokenizer(["XiaHuan is a C++ expert."], max_length=512, truncation=True, return_tensors='pt') # , padding="max_length", truncation=True

    # Generate Summary
    greedy_search = GreedySearch(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        min_length=1,
        max_length=512)

    pad_fn_object = PadFunction(tokenizer.pad_token_id)

    def predit_fn(source_inputs: List[torch.Tensor], states: List[torch.Tensor]):
        batch_size = len(source_inputs)

        batch = pad_fn_object(list(zip(source_inputs, states)))
        output = model(source_tokens=batch[0], target_tokens=batch[1])
        return output

    source_inputs = inputs['input_ids']
    batch_size = source_inputs.size(0)
    init_states = torch.full((batch_size, 1), tokenizer.bos_token_id, device=model.device)
    translation_ids = greedy_search.search(source_inputs, init_states, predit_fn)


    # translation_ids = greedy_search.search(output)
    result = ''
    tokens = tokenizer.convert_ids_to_tokens(translation_ids[0], skip_special_tokens=True)
    for each_token in tokens:
        if each_token == '[SEP]':
            break
        if not each_token.startswith('##'):
            result += ' ' + each_token
        else:
            result += each_token[2:]

    print(result)
