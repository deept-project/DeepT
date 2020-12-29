import pytorch_lightning as pl
import torch
import transformers
import tqdm
import os
import numpy as np

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        self.source_ids_path = source_path + '.npy'
        self.target_ids_path = target_path + '.npy'
        # tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

        print('tokenize source texts')
        self.source = []
        if not os.path.exists(self.source_ids_path):
            with open(self.source_path, 'r', encoding='utf-8') as f:
                for each_line in tqdm.tqdm(iterable=f):
                    if each_line.endswith('\n'):
                        each_line = each_line[:-1]
                    self.source.append(self.tokenizer(each_line))
            np.save(self.source_ids_path)
        else:
            self.source = np.load(self.source_ids_path)

        print('tokenize target texts')
        self.target = []
        if not os.path.exists(self.target_ids_path):
            with open(self.target_path, 'r', encoding='utf-8') as f:
                for each_line in tqdm.tqdm(iterable=f):
                    if each_line.endswith('\n'):
                        each_line = each_line[:-1]
                    self.target.append(self.tokenizer(each_line))
            np.save(self.target)
        else:
            self.target = np.load(self.target_ids_path)

    def __getitem__(self, i):
        return tuple(self.source[i], self.target[i])

    def __len__(self):
        return len(self.source)