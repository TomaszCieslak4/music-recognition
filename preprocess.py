import csv
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import torch

CLASSES = ["electronic", "classical", "pop", "soundtrack", "rock", "reggae", "hiphop", "ambient", "jazz", "metal", "poprock"]

class MusicDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir

        with open(csv_file, "r") as fp:
            reader = csv.reader(fp, delimiter="\t")
            header = next(reader)

            meta_list = []
            tags_list = []
            for row in reader:
                tags = [tag.removeprefix("genre---") for tag in row[5:]]
                if len(tags) != 1:
                    continue

                tag = tags[0]
                if tag not in CLASSES:
                    continue

                meta_list.append(row[:5])
                tags_list.append(tag)

            self.meta_df = pd.DataFrame(meta_list, columns=header[:5])
            unfiltered_tags_df = pd.get_dummies(pd.Series(tags_list))

            min_tag_count = unfiltered_tags_df.sum().min()
            self.tags_df = pd.concat([unfiltered_tags_df[unfiltered_tags_df[class_]==1].iloc[:min_tag_count] for class_ in CLASSES])

            self.meta_df = self.meta_df.iloc[self.tags_df.index].reset_index(drop=True)
            self.tags_df = self.tags_df.reset_index(drop=True)

            print("Loading dataset...")
            data1 = []
            data2 = []
            for idx in range(len(self.meta_df)):
                path = os.path.join(self.root_dir, self.meta_df.iloc[idx, 3])[:-3] + "npy"

                data1.append(np.load(path))
                data2.append(self.tags_df.iloc[idx].to_numpy())

            self.data1 = torch.tensor(np.array(data1)).to("cuda")
            self.data2 = torch.tensor(np.array(data2)).to("cuda").float()
            print(f"Loaded {len(self.meta_df)} data points")

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]
    
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # path = os.path.join(self.root_dir, self.meta_df.iloc[idx, 3])[:-3] + "npy"

        # return np.load(path), self.tags_df.iloc[idx].to_numpy()


dataset = MusicDataset("./autotagging_genre.tsv", "dump-spec-trimmed/")
# print(dataset[0][0].shape)
