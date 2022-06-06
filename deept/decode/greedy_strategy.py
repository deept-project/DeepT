
from typing import Callable
import torch

from .decode_strategy import DecodeStrategy


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
