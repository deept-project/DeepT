import torch
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

class BeamNode(object):
    def __init__(self, batch_idx: int, seq: torch.Tensor, eos_id: int, max_length: int):
        self.batch_idx = batch_idx
        self.max_length = max_length
        self.seq = seq
        self.log_prob = 0.0
        self.finished = False
        self.eos_id = eos_id

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
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length, top_k=3):
        super(BeamSearchSlow, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)
        self.top_k = top_k

    def search_one_batch(self, source_inputs: torch.Tensor, init_state: torch.Tensor, predit_fn: Callable):
        ret = [None] * self.top_k
        seq_len = 0
        k_states = [BeamNode(0, init_state.clone(), self.eos_id, self.max_length)]
        while any([not item.finished for item in k_states]):
            input_seqs, input_states = zip(*[(s.seq, s) for s in k_states if not s.finished])
            source_seqs = [source_inputs.clone() for i in range(len(input_states))]
            output = predit_fn(source_seqs, input_seqs)
            output = output.detach()
            # output: (batch_size, seq_len, vocab_size)
            top_prob, top_ids = torch.topk(output, self.top_k, dim=2)

            new_states = []
            for i in range(len(input_states)):
                for k in range(self.top_k):
                    new_token = top_ids[i, seq_len, k]
                    new_prob = top_prob[i, seq_len, k]
                    input_state = input_states[i].clone()
                    input_state.add_token(new_token, new_prob)
                    new_states.append(input_state)

            new_states = sorted(new_states, key=lambda x: x.mean_log_prob(), reverse=True)
            k_states = new_states[:self.top_k]

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
