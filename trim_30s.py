import csv
import os

import numpy as np

with open("./data/raw_30s_cleantags_50artists.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        folder, filename = row[3][:-4].split("/")
        print(f"Trimming {folder}/{filename}")
        array = np.load(f"./dump-spec/{folder}/{filename}.npy")[:, :1400]
        os.makedirs(f"./dump-spec-trimmed/{folder}", exist_ok=True)
        np.save(f"./dump-spec-trimmed/{folder}/{filename}.npy", array)
