import math
import torch
class BeamScorer(object):
    def __init__(self):
        pass
    
    def process(self, token: int, prob: float):
        pass

    def finalize(self):
        pass

    def reset(self):
        pass
    
    def clone(self):
        return BeamScorer()


class MeanLogProbScorer(BeamScorer):
    def __init__(self):
        super(MeanLogProbScorer, self).__init__()
        self.log_prob = 0.0
        self.length = 0

    def process(self, token: int, prob: float):
        self.log_prob += math.log(prob)
        self.length += 1

    def finalize(self) -> float:
        return self.log_prob / self.length

    def clone(self):
        scorer = MeanLogProbScorer()
        scorer.log_prob = self.log_prob
        scorer.length = self.length
        return scorer

class MeanLogProbWithLengthPenaltyScorer(BeamScorer):
    def __init__(self):
        super(MeanLogProbWithLengthPenaltyScorer, self).__init__()
        self.length_penalty = 0.9

        self.log_prob = 0.0
        self.length = 0

    def process(self, token: int, prob: float):
        self.log_prob += math.log(prob)
        self.length += 1

    def finalize(self) -> float:
        return self.log_prob / self.length ** self.length_penalty

    def reset(self):
        self.log_prob = 0.0
        self.length = 0

    def clone(self):
        scorer = MeanLogProbWithLengthPenaltyScorer()

        scorer.length_penalty = self.length_penalty

        scorer.log_prob = self.log_prob
        scorer.length = self.length
        return scorer

class MeanLogProbWithRepetitionPenaltyScorer(BeamScorer):
    def __init__(self):
        super(MeanLogProbWithRepetitionPenaltyScorer, self).__init__()
        self.length_penalty = 0.99
        self.repetition_penalty = 0.95

        self.previous_tokens = set()
        self.log_prob = 0.0
        self.length = 0
        self.repetition_counter = 0

    def process(self, token: int, prob: float):
        self.log_prob += math.log(prob)
        self.length += 1
        if token in self.previous_tokens:
            self.repetition_counter += 1
        self.previous_tokens.add(token)

    def finalize(self) -> float:
        return self.log_prob / (self.length ** self.length_penalty) * (self.repetition_penalty ** self.repetition_counter)

    def reset(self):
        self.previous_tokens = set()
        self.log_prob = 0.0
        self.length = 0
        self.repetition_counter = 0

    def clone(self):
        scorer = MeanLogProbWithRepetitionPenaltyScorer()
        scorer.length_penalty = self.length_penalty
        scorer.repetition_penalty = self.repetition_penalty

        scorer.previous_tokens = self.previous_tokens.copy()
        scorer.log_prob = self.log_prob
        scorer.length = self.length
        scorer.repetition_counter = self.repetition_counter

        return scorer