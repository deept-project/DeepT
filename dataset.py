import pytorch_lightning as pl
import torch
import transformers
import tqdm
import os
import numpy as np
import pickle

def save_bin(path, obj, obj_length):
    # with open(path, 'wb') as handle:
    #     pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(path, data=obj, length=obj_length)

def load_bin(path):
    # with open(path, 'rb') as handle:
    #     ret = pickle.load(handle)
    loaded = np.load(path)
    return loaded['data'], loaded['length']

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class TranslationDataset(torch.utils.data.Dataset):
    # @profile
    def __init__(self, source_path, target_path, tokenizer=None):
        self.source_path = source_path
        self.target_path = target_path
        self.source_ids_path = source_path + '.npz'
        self.target_ids_path = target_path + '.npz'
        # tokenizer
        if tokenizer is None:
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        else:
            self.tokenizer = tokenizer
        self.max_length = 512

        preprocess_batch_size = 20480
        print(f'tokenize source texts from {self.source_path}')
        
        if not os.path.exists(self.source_ids_path):
            with open(self.source_path, 'r', encoding='utf-8') as f:
                self.source = f.read().split('\n')
            self.input_ids = np.full((len(self.source), self.max_length), self.tokenizer.pad_token_id, dtype=np.int32)
            self.input_length = np.zeros((len(self.source),), dtype=np.int32)
            i = 0
            for each_lines in tqdm.tqdm(iterable=batch(self.source, preprocess_batch_size), total=len(self.source)//preprocess_batch_size):
                ids = self.tokenizer(each_lines, max_length=self.max_length, truncation=True, padding="do_not_pad", is_split_into_words=False)['input_ids']
                for each_id in ids:
                    self.input_ids[i, :len(each_id)] = each_id
                    self.input_length[i] = len(each_id)
                    i += 1
            save_bin(self.source_ids_path, self.input_ids, self.input_length)
            del self.source
        else:
            self.input_ids, self.input_length=load_bin(self.source_ids_path)
        
        print(f'tokenize target texts from {self.target_path}')
        if not os.path.exists(self.target_ids_path):
            with open(self.target_path, 'r', encoding='utf-8') as f:
                self.target = f.read().split('\n')
            self.labels = np.full((len(self.target), self.max_length), self.tokenizer.pad_token_id, dtype=np.int32)
            self.labels_length = np.zeros((len(self.target),), dtype=np.int32)
            i = 0
            for each_lines in tqdm.tqdm(iterable=batch(self.target, preprocess_batch_size), total=len(self.target)//preprocess_batch_size):
                ids = self.tokenizer(each_lines, max_length=self.max_length, truncation=True, padding="do_not_pad", is_split_into_words=False)['input_ids']
                for each_id in ids:
                    self.labels[i, :len(each_id)] = each_id
                    self.labels_length[i] = len(each_id)
                    i += 1
            save_bin(self.target_ids_path, self.labels, self.labels_length)
            del self.target
        else:
            self.labels, self.labels_length=load_bin(self.target_ids_path)

    def __getitem__(self, i):
        return (
            torch.tensor(self.input_ids[i,:self.input_length[i]]),
            torch.tensor(self.labels[i,:self.labels_length[i]])
            )

    def __len__(self):
        return self.input_ids.shape[0]


class TranslationLazyDataset(torch.utils.data.Dataset):
    # @profile
    def __init__(self, source_path, target_path, tokenizer=None):
        self.source_path = source_path
        self.target_path = target_path

        # tokenizer
        if tokenizer is None:
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        else:
            self.tokenizer = tokenizer
        self.max_length = 512

        print(f'tokenize source texts from {self.source_path}')
        with open(self.source_path, 'r', encoding='utf-8') as f:
            self.source = f.read().split('\n')

        print(f'tokenize target texts from {self.target_path}')
        with open(self.target_path, 'r', encoding='utf-8') as f:
            self.target = f.read().split('\n')

        print(len(self.source), len(self.target))

    def __getitem__(self, i):
        source_ids = self.tokenizer(self.source[i], max_length=self.max_length, truncation=True, padding="do_not_pad", is_split_into_words=False)['input_ids']
        target_ids = self.tokenizer(self.target[i], max_length=self.max_length, truncation=True, padding="do_not_pad", is_split_into_words=False)['input_ids']
        return (
            torch.tensor(source_ids),
            torch.tensor(target_ids)
            )

    def __len__(self):
        return len(self.source)
