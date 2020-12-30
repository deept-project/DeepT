import pytorch_lightning as pl
import torch
import transformers

class BartForMaskedLM(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.vocab_size = 119547
        self.config = transformers.BartConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=1024
        )
        self.transformer = transformers.BartModel(self.config)
        self.lm_head = torch.nn.Linear(1024, self.vocab_size, bias=False)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        transformer_outputs = self.transformer(x)
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, input_ids_length, labels, labels_length = batch

        transformer_outputs = self.transformer(input_ids)
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        loss = outputs[0]  # (loss), lm_logits, (all hidden states), (all attentions)

        # Logging to TensorBoard by default
        self.log('train_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer