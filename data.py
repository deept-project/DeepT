import pytorch_lightning as pl
import torch
import transformers
import tqdm
import os
import numpy as np
import pickle

def save_bin(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_bin(path):
    with open(path, 'rb') as handle:
        ret = pickle.load(handle)
    return ret

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, tokenizer=None):
        self.source_path = source_path
        self.target_path = target_path
        self.source_ids_path = source_path + '.npy'
        self.target_ids_path = target_path + '.npy'
        # tokenizer
        if tokenizer is None:
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        else:
            self.tokenizer = tokenizer

        preprocess_batch_size = 20480
        print('tokenize source texts')
        self.input_ids = []
        if not os.path.exists(self.source_ids_path):
            with open(self.source_path, 'r', encoding='utf-8') as f:
                self.source = f.read().splitlines()
            for each_lines in tqdm.tqdm(iterable=batch(self.source, preprocess_batch_size), total=len(self.source)//preprocess_batch_size):
                ids = self.tokenizer(each_lines, max_length=1024, truncation=True, is_split_into_words=False)['input_ids']
                for each_id in ids:
                    self.input_ids.append(torch.tensor(each_id, dtype=torch.int32))
            save_bin(self.source_ids_path, self.input_ids)
            del self.source
        else:
            self.input_ids=load_bin(self.source_ids_path)
        
        

        print('tokenize target texts')
        self.labels = []
        if not os.path.exists(self.target_ids_path):
            with open(self.target_path, 'r', encoding='utf-8') as f:
                self.target = f.read().splitlines()
            for each_lines in tqdm.tqdm(iterable=batch(self.target, preprocess_batch_size), total=len(self.target)//preprocess_batch_size):
                ids = self.tokenizer(each_lines, max_length=1024, truncation=True, is_split_into_words=False)['input_ids']
                for each_id in ids:
                    self.labels.append(torch.tensor(each_id, dtype=torch.int32))
            save_bin(self.target_ids_path, self.labels)
            del self.target
        else:
            self.labels=load_bin(self.target_ids_path)

    def __getitem__(self, i):
        return (self.input_ids[i], self.labels[i])

    def __len__(self):
        return len(self.input_ids)