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
import h5py





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
        self.mask_prob = 0.15
        self.hdf5 =  h5py.File(os.environ['SGE_LOCALDIR'] + '/dataset.hdf5', 'r') 

        # Calculate basic statistics
        self.statistics["n_docs"] = len(self.docs)
        for doc in self.docs:
            self.statistics["n_sents"] += len(doc)
            for sent in doc:
                self.statistics["n_tokens"] += len(sent)

    def __len__(self):
        return self.statistics["n_sents"]
    def cls_token(self):
        return [0]*self.num_sentence_embedding_dim
    def pad_token(self):
        return [1]*self.num_sentence_embedding_dim
    def mask_token(self):
        return [-1]*self.num_sentence_embedding_dim

    def __getitem__(self, idx):
        seq = self.hdf5[self.stage + '/' + self.docs[idx%len(self.docs)]]
        if seq.shape[0] > self.max_seq_len:
            start_pos = torch.randint(0,seq.shape[0] - self.max_seq_len,(1,))
        else:
            start_pos = 0
        seq = [self.cls_token()] + seq[start_pos:start_pos+self.max_seq_len - 1].tolist()

        return seq
    def collate_fn(self,seqs):
        batch_size = len(seqs)
        max_seq_len = max([len(seq) for seq in seqs])

        # padding input sequences
        seqs = [seq + [self.pad_token()]*(max_seq_len-len(seq)) for seq in seqs]

        # convert to tensors
        seqs = torch.FloatTensor(np.array(seqs))

        # get mask token masks
        special_token_masks = torch.zeros(seqs[:,:,0].size()).bool()
        for special_token_id in [self.pad_token(), self.cls_token()]:
            special_token_masks = special_token_masks | (seqs == special_token_id)
        # sample mask token masks
        mask_token_probs = torch.FloatTensor([self.mask_prob]).expand_as(seqs[:,:,0])  # [batch_size, max_seq_len]
        mask_token_probs = mask_token_probs.masked_fill(special_token_masks, 0.0)
        while True:  # prevent that there is not any mask token
            mask_token_masks = torch.bernoulli(mask_token_probs).bool()
            if (mask_token_masks.long().sum(1) == 0).sum() == 0:
                break

        # input ids
        input_ids = seqs.clone()
        input_ids[mask_token_masks] = torch.FloatTensor(self.mask_token())

        # position ids
        position_ids = [list(range(max_seq_len))] * batch_size
        position_ids = torch.LongTensor(position_ids)

        attn_masks = input_ids.sum(dim=2) == torch.FloatTensor(self.pad_token()).sum()
        attn_masks = (~attn_masks).float()

        return {
            "input_embeds": input_ids,
            "position_ids": position_ids,
            "attn_masks": attn_masks
        }, seqs,mask_token_masks
