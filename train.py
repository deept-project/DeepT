import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor
import torch
import transformers

from model import BartForMaskedLM, PadFunction
import torch
import tqdm
import numpy as np

from translate import GreedySearch

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
    logger = TensorBoardLogger('tb_logs', name='translation')
    trainer = pl.Trainer(
        gpus=[0],
        # num_nodes=1,
        max_epochs=10,
        # accelerator='ddp',
        # plugins='ddp_sharded',
        amp_backend='native',
        amp_level='O2',
        precision=16,
        auto_scale_batch_size=None,
        log_gpu_memory=None,
        log_every_n_steps=10,
        flush_logs_every_n_steps=100,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        accumulate_grad_batches=16,
        sync_batchnorm=True,
        checkpoint_callback=True,
        resume_from_checkpoint=None,
        logger=logger,
        callbacks=[],
        # profiler="simple",
    )
    # find the batch size
    # trainer.tune(model)
    trainer.fit(model, train_dataloader, valid_dataloader)


    # infer
    tokenizer = transformers.BertTokenizerFast('./vocab/vocab.txt')
    setattr(tokenizer, "_bos_token", '[CLS]')
    setattr(tokenizer, "_eos_token", '[SEP]')

    inputs = tokenizer(["XiaHuan is a C++ expert."], max_length=512, truncation=True, return_tensors='pt') # , padding="max_length", truncation=True

    # Generate Summary
    greedy_search = GreedySearch(
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        min_length=1,
        max_length=512)

    pad_fn_object = PadFunction(tokenizer.pad_token_id)

    def predit_fn(source_inputs: torch.Tensor, states: torch.Tensor):
        batch_size = source_inputs.size(0)
        source_list = [source_inputs[i,:] for i in range(batch_size)]
        state_list = [states[i,:] for i in range(batch_size)]

        batch = pad_fn_object(list(zip(source_list, state_list)))
        output = model(*batch)
        return output

    source_inputs = inputs['input_ids']
    batch_size = source_inputs.size(0)
    init_states = torch.full((batch_size, 1), tokenizer.bos_token_id)
    translation_ids = greedy_search.search(source_inputs, init_states, predit_fn)


    # translation_ids = greedy_search.search(output)
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translation_ids[0]])
