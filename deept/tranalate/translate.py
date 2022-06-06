import torch
import torch.nn.functional as F

from typing import Tuple, List, Callable

import tqdm
import math

class DecodeStrategy(object):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length):
        self.pad_id=pad_id
        self.bos_id=bos_id
        self.eos_id=eos_id
        self.min_length=min_length
        self.max_length=max_length

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

class BeamScorer(object):
    def __init__(self, alpha=0.8, beta=0.6):
        self.alpha=alpha
        self.beta=beta
    
    def score(self, beam_entry):
        return beam_entry.score * self.alpha + beam_entry.length * self.beta

class BeamNode(object):
    def __init__(self, batch_idx: int, seq: torch.Tensor, eos_id: int, max_length: int):
        self.batch_idx = batch_idx
        self.max_length = max_length
        self.seq = seq
        self.log_prob = 0.0
        self.finished = False
        self.eos_id = eos_id
        self.length_penalty = 0.0

    def add_token(self, token: int, prob: float):
        if self.finished:
            return
        if self.seq.size(-1) >= self.max_length:
            token = self.eos_id
        self.seq = torch.cat([self.seq, torch.full((1,), token, dtype=torch.long, device=self.seq.device)])
        self.log_prob += math.log(prob)
        if token == self.eos_id:
            self.finished = True
    
    def clone(self):
        node = BeamNode(self.batch_idx, self.seq.clone(), self.eos_id, self.max_length)
        node.log_prob = self.log_prob
        node.finished = self.finished
        return node
    
    def length(self):
        return self.seq.size(-1)

    def mean_log_prob(self):
        return self.log_prob / self.length()

class BeamSearchSlow(DecodeStrategy):
    def __init__(self, pad_id: int, bos_id: int, eos_id: int,
                min_length: int, max_length: int,
                num_beams: int = 3, top_k: int = 5, top_p: float = 0.9
        ):
        super(BeamSearchSlow, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p

    def search_one_batch(self, source_inputs: torch.Tensor, init_state: torch.Tensor, predit_fn: Callable):
        ret = [None] * self.num_beams
        seq_len = 0
        k_states = [BeamNode(0, init_state.clone(), self.eos_id, self.max_length)]
        while any([not item.finished for item in k_states]):
            input_seqs, input_states = zip(*[(s.seq, s) for s in k_states if not s.finished])
            source_seqs = [source_inputs.clone() for i in range(len(input_states))]
            output = predit_fn(source_seqs, input_seqs)
            output = output.detach()

            # output: (batch_size, seq_len, vocab_size)
            # top_k_top_p_filtering: (batch_size, seq_len, vocab_size)
            output[:, -1, :] = top_k_top_p_filtering(output[:, -1, :].squeeze(1), top_k = self.top_k, top_p = self.top_p, filter_value = 1)
            
            # top-k
            top_prob, top_ids = torch.topk(output, self.num_beams, dim=2)

            new_states = []
            for i in range(len(input_states)):
                for k in range(self.num_beams):
                    new_token = top_ids[i, seq_len, k]
                    new_prob = top_prob[i, seq_len, k]
                    input_state = input_states[i].clone()
                    input_state.add_token(new_token, new_prob)
                    new_states.append(input_state)

            new_states = sorted(new_states, key=lambda x: x.mean_log_prob(), reverse=True)
            k_states = new_states[:self.num_beams]

            seq_len += 1
        ret = k_states
        ret = sorted(ret, key=lambda x: x.mean_log_prob(), reverse=True)
        return ret[0].seq

    def search(self, source_inputs: torch.Tensor, init_states: torch.Tensor, predit_fn: Callable): #  -> Tuple[torch.Tensor, torch.Tensor]:
        # source_inputs: (batch_size, seq_len)
        batch_size = source_inputs.size(0)
        ret = []

        for batch_i in range(batch_size):
            out = self.search_one_batch(source_inputs[batch_i,:].detach(), init_states[batch_i,:].detach(), predit_fn)
            ret.append(out)

        return ret


class BeamSearch(DecodeStrategy):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length, top_k=10):
        super(BeamSearch, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)
        self.top_k = top_k

    def search(self, source_inputs: torch.Tensor, init_states: torch.Tensor, predit_fn: Callable): #  -> Tuple[torch.Tensor, torch.Tensor]:
        # source_inputs: (batch_size, seq_len)
        batch_size = source_inputs.size(0)
        max_seq_len = source_inputs.size(1) * 2
        seq_len = 16 if max_seq_len > 16 else max_seq_len
        translation_ids = torch.full((batch_size, seq_len), self.pad_id, device=source_inputs.device)

        # init_states: (batch_size, seq_len)
        states = init_states

        for i in tqdm.tqdm(iterable=range(seq_len)):
            output = predit_fn(source_inputs, states)
            # output: (batch_size, seq_len, vocab_size)
            top_prob, top_ids = torch.topk(output, self.top_k, dim=2)

            # top_prob, top_ids: (batch_size, seq_len, top_k)
            # states = torch.cat((states, torch.unsqueeze(max_ids[:, i], dim=1)), dim=1)
            # translation_ids[:, i] = max_ids[:, i]
        print(translation_ids)

        return translation_ids


class GreedySearch(DecodeStrategy):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length):
        super(GreedySearch, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)

    def search(self, source_inputs: torch.Tensor, init_states: torch.Tensor, predit_fn: Callable): #  -> Tuple[torch.Tensor, torch.Tensor]:
        # source_inputs: (batch_size, seq_len)
        batch_size = source_inputs.size(0)

        seq_len = 0
        states = [init_states[i,:] for i in range(batch_size)]
        ret = [None] * batch_size

        while any([item is None for item in ret]):
            input_states = []
            index_map = []
            for batch_i, state in enumerate(states):
                if ret[batch_i] is not None:
                    continue
                input_states.append(state)
                index_map.append(batch_i)

            output = predit_fn(source_inputs, input_states)
            output = output.detach()
            # output: (batch_size, seq_len, vocab_size)
            max_prob, max_ids = torch.max(output, dim=2)

            for i, s in enumerate(input_states):
                batch_i = index_map[i]
                s = torch.cat((s, max_ids[i, seq_len].expand(1)), dim=0)
                if max_ids[i, seq_len] == self.eos_id:
                    ret[batch_i] = s
                    states[batch_i] = None
                else:
                    states[batch_i] = s

            seq_len += 1

        return ret
