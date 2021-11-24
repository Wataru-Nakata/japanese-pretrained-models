# coding=utf-8
# Copyright 2021 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import torch
import numpy as np
import pandas as pd
import os


def collate_fn(batch_data):
    return batch_data



class DataSource(torch.utils.data.Dataset):

    def __init__(self, config, docs, stage=None):
        # Attributes
        self.max_seq_len = config.max_seq_len
        # Other attributes
        self.stage = stage
        self.statistics = {"n_docs": 0, "n_sents": 0, "n_tokens": 0}
        self.num_sentence_embedding_dim = config.num_sentence_embedding_dim

        # Load dataset
        self.docs = docs
        self.dfs = []

        # Calculate basic statistics
        self.statistics["n_docs"] = len(self.docs)
        for doc in self.docs:
            self.statistics["n_sents"] += len(doc)
            for sent in doc:
                self.statistics["n_tokens"] += len(sent)
            self.dfs.append(pd.read_feather(os.path.join('../data/sentence-book/doc_data',doc))['embeds'])

    def __len__(self):
        return len(self.docs)
    def cls_token(self):
        return [0]*self.num_sentence_embedding_dim
    def pad_token(self):
        return [1]*self.num_sentence_embedding_dim
    def mask_token(self):
        return [-1]*self.num_sentence_embedding_dim

    def __getitem__(self, idx):
        df = self.dfs[idx]
        seq = df.to_list()
        
        if len(seq) > self.max_seq_len:
            start_pos = torch.randint(0,len(seq) - self.max_seq_len,(1,))
        else:
            start_pos = 0
        seq = [self.cls_token()] + seq[start_pos:start_pos+self.max_seq_len - 1]

        return seq
