
import queue
import uuid
import torch
import time
from typing import Dict, List
from enum import Enum
import threading

import transformers
from translate import GreedySearch

from flask import Flask

from model import BartForMaskedLM

app = Flask(__name__)


class PadFunction(object):
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        return self._pad_fn(batch)

    def merge(self, sequences, pad_size=None):
        lengths = [len(seq) for seq in sequences]
        if pad_size is None:
            pad_size = max(lengths)
        padded_seqs = torch.full((len(sequences), pad_size), self.pad_id).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def make_mask(self, inputs, inputs_length):
        inputs_mask = torch.zeros_like(inputs)
        for i in range(inputs_mask.size(0)):
            inputs_mask[i,:inputs_length[i]] = 1
        return inputs_mask

    def _pad_fn(self, batch):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = zip(*batch)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        # pad_size = max([len(seq) for seq in src_seqs] + [len(seq) for seq in trg_seqs])
        pad_size=None
        src_seqs, src_lengths = self.merge(src_seqs, pad_size)
        trg_seqs, trg_lengths = self.merge(trg_seqs, pad_size)

        source_tokens = {
            'token_ids': src_seqs.to('cuda'),
            'mask': self.make_mask(src_seqs, src_lengths).to('cuda'),
        }

        target_tokens = {
            'token_ids': trg_seqs.to('cuda'),
            'mask': self.make_mask(trg_seqs, trg_lengths).to('cuda'),
        }
        return source_tokens, target_tokens

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

class TranslationType(Enum):
    EN2ZH = 1
    ZH2EN = 2

class TranslationStatus(Enum):
    Empty = 1
    Waiting = 2
    Translating = 3
    Completed = 4
    Outdated = 5


class TranslationTask(object):
    def __init__(self, task_type:TranslationType, text:str):
        self.uuid = uuid.uuid4()
        self.task_type = task_type
        self.text = text
        self.result = None
        self.status = TranslationStatus.Empty

    def waiting(self):
        self.status = TranslationStatus.Waiting

    def translating(self):
        self.status = TranslationStatus.Translating
    
    def completed(self):
        self.status = TranslationStatus.Completed

    def outdated(self):
        self.status = TranslationStatus.Outdated

class TranslationService(object):
    def __init__(self):
        self.tokenizer = transformers.BertTokenizerFast('./vocab/vocab.txt')
        setattr(self.tokenizer, "_bos_token", '[CLS]')
        setattr(self.tokenizer, "_eos_token", '[SEP]')

        self.model = BartForMaskedLM.load_from_checkpoint('tb_logs/translation/version_1/checkpoints/epoch=24-step=48372.ckpt')
        self.model = self.model.to('cuda')
        self.model.eval()

        self.q = queue.Queue()
        self.tasks = {}

        self.pad_fn_object = PadFunction(self.tokenizer.pad_token_id)

        self.timer = threading.Timer(1, self.process_tasks,())
        self.timer.start()

    def create_task(self, task_type:TranslationType, text:str) -> uuid.UUID:
        task = TranslationTask(task_type, text)
        self.tasks[task.uuid] = task

        return task.uuid

    def push_task_to_queue(self, uuid: uuid.UUID):
        if not uuid in self.tasks:
            return
        self.tasks[uuid].waiting()
        self.q.put(uuid)

    def submit(self, task_type:TranslationType, text:str) -> int:
        id = self.create_task(task_type, text)
        self.push_task_to_queue(id)
        return id

    def status(self, uuid: uuid.UUID) -> TranslationStatus:
        if not uuid in self.tasks:
            return TranslationStatus.Empty
        else:
            return self.tasks[uuid].status

    def get_result(self, uuid: uuid.UUID) -> str:
        if not uuid in self.tasks:
            return ''
        else:
            return self.tasks[uuid].result

    def process_tasks(self):
        while True:
            if self.q.qsize() > 16:
                pop_num = 16
            elif self.q.qsize() == 0:
                continue
            else:
                pop_num = self.q.qsize()

            task_ids = []
            for i in range(pop_num):
                each_id = self.q.get_nowait()
                task_ids.append(each_id)

            tasks = []
            for each_id in task_ids:
                each_task = self.tasks[each_id]
                each_task.status = TranslationStatus.Translating
                tasks.append(each_task)
        
            result = self.translate([each_task.text for each_task in tasks])

            for i, each_id in enumerate(task_ids):
                each_task = self.tasks[each_id]
                each_task.result = result[i]
                each_task.status = TranslationStatus.Completed
            time.sleep(1)

    def predit_fn(self, source_inputs: torch.Tensor, states: torch.Tensor):
        batch_size = source_inputs.size(0)
        source_list = [source_inputs[i,:] for i in range(batch_size)]
        state_list = [states[i,:] for i in range(batch_size)]

        batch = self.pad_fn_object(list(zip(source_list, state_list)))
        output = self.model(source_tokens=batch[0], target_tokens=batch[1])
        return output

    def translate(self, text_list:List[str]) -> List[str]:
        greedy_search = GreedySearch(
            pad_id=self.tokenizer.pad_token_id,
            bos_id=self.tokenizer.bos_token_id,
            eos_id=self.tokenizer.eos_token_id,
            min_length=1,
            max_length=512)

        inputs = self.tokenizer(text_list, max_length=512, padding="longest", truncation=True, return_tensors='pt') # , truncation=True

        source_inputs = inputs['input_ids'].to('cuda')
        batch_size = source_inputs.size(0)
        init_states = torch.full((batch_size, 1), self.tokenizer.bos_token_id).to('cuda')
        translation_ids = greedy_search.search(source_inputs, init_states, self.predit_fn)

        # translation_ids = greedy_search.search(output)
        result = []
        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(translation_ids[i, :], skip_special_tokens=False)
            new_tokens = []
            for each_token in tokens:
                if each_token == '[SEP]':
                    break
                new_tokens.append(each_token)
            
            result_str = ''
            for each_token in new_tokens:
                if any([is_chinese(c) for c in each_token]):
                    result_str += each_token
                else:
                    if not each_token.startswith('##'):
                        result_str += ' ' + each_token
                    else:
                        result_str += each_token[2:]

            result.append(result_str)

        print(result)
        return result

        

service = TranslationService()

@app.route('/en2zh/<text>', methods=['GET'])
def submit_en2zh_task(text):
    id = service.submit(TranslationType.EN2ZH, text)
    return str(id)


@app.route('/status/<id>', methods=['GET'])
def check_status(id):
    return str(service.status(uuid.UUID(id)))

@app.route('/result/<id>', methods=['GET'])
def get_result(id):
    return service.get_result(uuid.UUID(id))

if __name__ == "__main__":
    app.run()