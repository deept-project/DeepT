import pytorch_lightning as pl
import torch
import transformers
from deept.dataset.dataset import TranslationDataset, TranslationLazyDataset

from ..utils import PadFunction

class BartForMaskedLM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.batch_size = 4
        self.learning_rate = 3e-5
        self.d_model = 1024

        self.tokenizer = transformers.MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        # setattr(self.tokenizer, "_bos_token", '[CLS]')
        # setattr(self.tokenizer, "_eos_token", '[SEP]')

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.vocab_size = self.tokenizer.vocab_size

        self.transformer = transformers.MBartModel.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.lm_head = torch.nn.Linear(self.d_model, self.vocab_size, bias=False)

    # @profile
    def forward(self, source_tokens, target_tokens):
        inputs, labels = source_tokens, target_tokens

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        # in lightning, forward defines the prediction/inference actions
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=label_ids,
            decoder_attention_mask=label_mask,
            use_cache=False,
        )
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

    # @profile
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs, labels = batch

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        lm_logits = self.forward(
            source_tokens={
                'token_ids': input_ids,
                'mask': input_mask
            },
            target_tokens={
                'token_ids': label_ids[..., :-1],
                'mask': label_mask[..., :-1]
            }
        )

        shift_label_ids = label_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size),
                        shift_label_ids.view(-1))

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        lm_logits = self.forward(
            source_tokens={
                'token_ids': input_ids,
                'mask': input_mask
            },
            target_tokens={
                'token_ids': label_ids[..., :-1],
                'mask': label_mask[..., :-1]
            }
        )

        shift_label_ids = label_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size),
                        shift_label_ids.view(-1))
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        ai_challenger_2017_dataset = TranslationLazyDataset(
            'data/ai_challenger_2017_train.en', 'data/ai_challenger_2017_train.zh', tokenizer=self.tokenizer)
        minecraft_dataset = TranslationLazyDataset(
            'data/minecraft.en', 'data/minecraft.zh', tokenizer=self.tokenizer)
        translation2019zh_dataset = TranslationLazyDataset(
            'data/translation2019zh_train.en', 'data/translation2019zh_train.zh', tokenizer=self.tokenizer)
        MultiUN_en_zh_dataset = TranslationLazyDataset(
            'data/MultiUN.en-zh.en', 'data/MultiUN.en-zh.zh', tokenizer=self.tokenizer)
        umcorpus_dataset = TranslationLazyDataset(
            'data/umcorpus.en', 'data/umcorpus.zh', tokenizer=self.tokenizer)
        news_commentary_dataset = TranslationLazyDataset(
            'data/news-commentary-v12.zh-en.en', 'data/news-commentary-v12.zh-en.zh', tokenizer=self.tokenizer)
        ted_dataset = TranslationLazyDataset(
            'data/ted_train_en-zh.raw.en', 'data/ted_train_en-zh.raw.zh', tokenizer=self.tokenizer)

        dataset = torch.utils.data.ConcatDataset(
            [
                ai_challenger_2017_dataset,
                minecraft_dataset,
                translation2019zh_dataset,
                MultiUN_en_zh_dataset,
                umcorpus_dataset,
                news_commentary_dataset,
                ted_dataset,
            ]
        )
        train_sampler = torch.utils.data.RandomSampler(
            dataset, num_samples=len(dataset)//100, replacement=True)

        pad_fn_object = PadFunction(self.tokenizer.pad_token_id)
        train_loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=self.batch_size, collate_fn=pad_fn_object, sampler=train_sampler, pin_memory=True)

        return train_loader

    def val_dataloader(self):
        translation2019zh_valid_dataset = TranslationLazyDataset('data/translation2019zh_valid.en', 'data/translation2019zh_valid.zh', tokenizer=self.tokenizer)

        valid_dataset = translation2019zh_valid_dataset

        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)

        pad_fn_object = PadFunction(self.tokenizer.pad_token_id)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, num_workers=4, batch_size=self.batch_size, collate_fn=pad_fn_object, sampler=valid_sampler, pin_memory=True)

        return valid_loader

    def test_dataloader(self):
        pass
