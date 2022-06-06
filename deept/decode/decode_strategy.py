

class DecodeStrategy(object):
    def __init__(self, pad_id: int, bos_id: int, eos_id: int, min_length: int, max_length: int):
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.min_length = min_length
        self.max_length = max_length