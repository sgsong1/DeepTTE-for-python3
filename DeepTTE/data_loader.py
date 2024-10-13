import time
import utils

import torch
import torch.multiprocessing as multiprocessing
# CUDA 설정 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

class MySet(Dataset):
    def __init__(self, input_file):
        with open('./data/' + input_file, 'r') as f:
            self.content = [json.loads(line) for line in f]  # 리스트로 변환
        self.lengths = [len(item['lngs']) for item in self.content]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            # 만약 idx가 리스트라면, 해당 리스트에 포함된 인덱스 데이터들을 반환
            return [self.content[i] for i in idx]
        return self.content[idx]  # 단일 인덱스일 경우 개별 데이터 반환

    def __len__(self):
        return len(self.content)



def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        # pad to the max length
        seqs = [item[key] for item in data]
        
        # Find the max sequence length for the current batch
        max_len = max([len(seq) for seq in seqs])
        
        # Initialize padded sequence with zeros
        padded = np.zeros((len(seqs), max_len), dtype=np.float32)
        
        # Mask and fill the padded sequences
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq

        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        padded = torch.from_numpy(padded).float()
        traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = list(dataset.lengths)  # 리스트로 변환
        self.indices = list(range(self.count))  # 리스트로 변환

    def __iter__(self):
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100
        chunks = (self.count + chunk_size - 1) // chunk_size

        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]  # iterable한 리스트 반환

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, batch_size):
    dataset = MySet(input_file=input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset=dataset, 
                             collate_fn=collate_fn, 
                             num_workers=4,
                             batch_sampler=batch_sampler,
                             pin_memory=True)

    return data_loader