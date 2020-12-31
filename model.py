import pytorch_lightning as pl
import torch
import transformers

class BartForMaskedLM(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.bos_token_id = config['bos_token_id']
        self.eos_token_id = config['eos_token_id']
        self.pad_token_id = config['pad_token_id']

        self.vocab_size = config['vocab_size']
        self.config = transformers.BartConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=1024,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
        )
        self.transformer = transformers.BartModel(self.config)
        self.lm_head = torch.nn.Linear(1024, self.vocab_size, bias=False)

    def forward(self, batch):
        inputs, inputs_length, labels, labels_length = batch
        batch_size = inputs.shape[0]

        inputs_mask = torch.zeros_like(inputs)
        for i in range(batch_size):
            inputs_mask[i,:inputs_length[i]] = 1
        labels_mask = torch.zeros_like(labels)
        for i in range(batch_size):
            labels_mask[i,:labels_length[i]] = 1

        # in lightning, forward defines the prediction/inference actions
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=inputs_mask,
            decoder_input_ids=labels,
            decoder_attention_mask=labels_mask,
            )
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs, inputs_length, labels, labels_length = batch
        batch_size = inputs.shape[0]

        inputs_mask = torch.zeros_like(inputs)
        for i in range(batch_size):
            inputs_mask[i,:inputs_length[i]] = 1
        labels_mask = torch.zeros_like(labels)
        for i in range(batch_size):
            labels_mask[i,:labels_length[i]] = 1

        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=inputs_mask,
            decoder_input_ids=labels[..., :-1],
            decoder_attention_mask=labels_mask[..., :-1],
            )
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        # Logging to TensorBoard by default
        self.log('train_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
