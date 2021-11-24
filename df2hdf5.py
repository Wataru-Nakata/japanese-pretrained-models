from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import tqdm
import random
data_dir = Path('data/sentence-book/doc_data/')
feathers = data_dir.glob('*.feather')

f = h5py.File(data_dir.parent/'dataset.hdf5', "w")
f.create_group('train')
f.create_group('val')
feathers = list(feathers)
random.shuffle(feathers)
for idx, feather in enumerate(tqdm.tqdm(feathers)):
    df = pd.read_feather(feather)
    if idx > 1000:
        grp = f.create_dataset('/train/' +feather.stem,dtype=np.float32,shape=(len(df), 768),data=np.stack(df['embeds'].values))
    else:
        grp = f.create_dataset('/val/' +feather.stem,dtype=np.float32,shape=(len(df), 768),data=np.stack(df['embeds'].values))
f.close()
