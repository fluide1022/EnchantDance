import codecs as cs
from os.path import join as pjoin

import numpy as np
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# import spacy
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


"""For use of training music2dance generative model"""

class Music2DanceDatasetNew(data.Dataset):

    def __init__(
        self,
        split_file,
        motion_dir,
        music_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):

        self.pointer = 0
        self.motion_dir = motion_dir
        self.music_dir = music_dir

        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading ChoreoSpectrum3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        name_list = []
        length_list = []
        for i, name in enumerator:
            name_list.append(name)
            # For ChoreoSpectrum3D
            length_list.append(200)

            # For aist
            # length_list.append(240)

        self.length_arr = np.array(length_list)
        self.nfeats = 216
        self.name_list = name_list

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        name = self.name_list[item]

        # Train diffusion
        motion = np.load(pjoin(self.motion_dir, name + ".npy"))[:, 3:]
        m_length = 200
        music = np.load(pjoin(self.music_dir, name + ".npy"))

        return (motion, m_length, music)




