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

import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize
from konoha import SentenceTokenizer

from corpus.sentence_book.LaBSEExtractor import LabseExtractor
from corpus.sentence_book.config import Config
import argparse

config = Config()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract labse embedding for sentence_roberta')
    parser.add_argument('num', metavar='N', type=int)
    args = parser.parse_args()
    extractor = LabseExtractor()
    files = []
    languages = []
    print('args',args.num)
    japanese_tokenizer = SentenceTokenizer()
    os.makedirs(config.doc_data_dir,exist_ok=True)
    for raw_data_dir,lang in zip(config.raw_data_dirs,config.languages):
        files += list((Path(raw_data_dir)).glob('*.txt'))
        languages += [lang]*len(list((Path(raw_data_dir)).glob('*.txt')))
    print("process {} files".format(len(files)))
    for file,lang in tqdm(zip(files,languages)):
        if (Path(config.doc_data_dir)/file.with_suffix('.feather').name).exists():
            continue
        with open(file,mode='r') as f:
            lines = f.readlines()
        lines = [line for line in lines if line.strip()] 
        sents = []
        if lang == 'en':
            for line in lines:
                sents += sent_tokenize(line,language='english')
        elif lang == 'ja':
            for line in lines:
                sents += japanese_tokenizer.tokenize(line)
        else:
            raise ValueError
        sentence_embeds = extractor.extract_embeds(sents)
        df = pd.DataFrame()
        df['lines'] = sents
        df['embeds'] = sentence_embeds.tolist()
        df.to_feather(Path(config.doc_data_dir)/file.with_suffix('.feather').name)
