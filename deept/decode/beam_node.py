import math
import torch
from .beam_scorer import MeanLogProbScorer, MeanLogProbWithLengthPenaltyScorer, MeanLogProbWithRepetitionPenaltyScorer

class BeamNode(object):
    def __init__(self, batch_idx: int, seq: torch.Tensor, eos_id: int, max_length: int):
        self.batch_idx = batch_idx
        self.seq = seq
        self.eos_id = eos_id
        self.max_length = max_length
        self.finished = False
        self.scorer = MeanLogProbWithRepetitionPenaltyScorer()

    def add_token(self, token: int, prob: float):
        if self.finished:
            return
        if self.seq.size(-1) >= self.max_length:
            token = self.eos_id
        self.seq = torch.cat([self.seq, torch.full((1,), token, dtype=torch.long, device=self.seq.device)])
        self.scorer.process(token, prob)
        if token == self.eos_id:
            self.finished = True
    
    def clone(self):
        node = BeamNode(self.batch_idx, self.seq.clone(), self.eos_id, self.max_length)
        node.finished = self.finished
        node.scorer = self.scorer.clone()
        return node

    def score(self):
        return self.scorer.finalize()
