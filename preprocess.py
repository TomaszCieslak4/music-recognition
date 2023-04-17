import csv
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import torch


class MusicDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        mlb = MultiLabelBinarizer()
        self.root_dir = root_dir

        with open(csv_file, "r") as fp:
            reader = csv.reader(fp, delimiter="\t")
            header = next(reader)

            meta_list = []
            tags_list = []
            for row in reader:
                meta_list.append(row[:5])
                tags_list.append(row[5:])

            self.meta_df = pd.DataFrame(meta_list, columns=header[:5])
            tags_series = pd.Series(tags_list)
            self.tags_df = pd.DataFrame(
                mlb.fit_transform(tags_series),
                columns=mlb.classes_,
                index=tags_series.index
            )

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(
            self.root_dir,
            self.meta_df.iloc[idx, 3]
        )[:-3] + "npy"

        return np.load(path), self.tags_df.iloc[idx].to_numpy()


# dataset = MusicDataset("./autotagging_moodtheme.tsv", "dump-spec-trimmed/")
# print(len(dataset))
# print(dataset[0][0].shape)
