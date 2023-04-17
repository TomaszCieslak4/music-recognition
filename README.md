# Music-Recognition | RNN Model

Music has been historically categorized by humans into several genres, such as rock, blues, classical, and hip-hop.

Our model located in `model.py` analyzes segments of digital music recordings to predict the mood, and theme <!-- Add more? --> of the sample. The model will rely on pattern recognition based on the sequence of features extracted from different time sequences of the musical input.

## Data

We will be using the [MTG-Jamendo dataset](https://zenodo.org/record/3826813) which is a relatively new music dataset created in 2019 and contains more than 55,000 tracks with 195 tags. All of the tracks in the dataset can be found on Freesound.org, a creative-commons licensed music library. Custom subsets of the data set are available on the [MGT github](https://github.com/MTG/mtg-jamendo-dataset/tree/master/data).

<p align="center">
    <img src="./images/top20.png">
</p>

The 195 tags in the MTG-Jamendo dataset consist of 95 genres, 41 instruments, and 59 mood/themes. The 5 most popular mood/themes are happy, melodic, dark, relaxing, and energetic. All tags in the dataset are guaranteed to have at least 100 tracks assigned to each.

### Obtaining and Processing the Data

> :warning: You should have at least 300GB of free disk space to download and process the data.

Clone the MTG-Jamendo dataset repo and setup an virtual environment

```bash
git clone https://github.com/MTG/mtg-jamendo-dataset
cd mtg-jamendo-dataset
python -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt
```

Download the melspectrogram files

```bash
python scripts/download/download.py dump-spec/ --type melspecs
```

Copy the `trim_30s.py` file into the mtg-jamendo-dataset repo and run the script to trim melspectrogram files to 30s

```bash
python trim_30s.py
```

### Data Transformation

The data loading code is located in `preprocess.py`. Any `.tsv` dataset can be loaded in the same manner.

Here is an example of loading a dataset of songs tagged by mood/theme.

```py
from preprocess import MusicDataset

dataset = MusicDataset("./autotagging_moodtheme.tsv", "dump-spec-trimmed/")
```

Each dataset element is a tuple containing the audio sample and the one hot vector containing tags:

```py
sample, one_hot = dataset[0]
assert sample.shape == (96, 1400)
assert one_hot.tolist() == [0, 0, 1, ..., 0]
```

The `MusicDataset` class implements PyTorch's dataset API so it can be used in PyTorch's dataloader:

```py
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
```

### Data Split

We randomly split the data into test, validation, and training sets. The training set is 80% of the data, then we evenly split the remaining 20% between the validation and test sets. The dataset used is fairly large, hence it is acceptable to use a smaller validation and test set ratio in respect to the training set.


## Model

Our model starts with two convolutional layers on the input sequence, followed by a recurrent LSTM layer, and then finally a dense layer to produce our prediction.

<!-- Get draw.io diagram here: https://drive.google.com/file/d/1bzp8DmXyVA3tZMGHvwD5Hm7ej_323Axe/view?usp=sharing -->
<p align="center">
    <img src="./images/Music-Recognition.drawio.png">
</p>

<!-- Count the number of parameters in the model, and a description of where the parameters come from

Examples of how the model performs on two actual examples from the test set: one successful and one
unsuccessful 

TODO: ? utilize various attention-based mechanisms to optimize the performance and predictive accuracy.  -->

## Training


## Hyperparamter Tuning 

Following hyperparameters were set to the following values: 

hidden_channel=16, linear_transform=32, output_channel=8
learning_rate=0.01, weight_decay=0.001

Hidden channel, the number of features extracted at each layer in our RNN, was chosen to be 16, as there are 96 input channels associated with each spectrogram signal, if set too high, runtime would be affected, and if set too low, some information loss would occur given the high number of input channels. For similar reasons, linear transform, the dimension of the hidden dense layer, was set to 32, as to avoid information loss. 

We start 

## Results

<!-- A justification that your implemented method performed reasonably, given the difficulty of the
problem—or a hypothesis for why it doesn’t. -->

We hypothesize that our implemented method to train and validate the model is somewhat functional but is not stable with either our dataset or our architecture.

We adjusted our model in various ways including and reducing/increasing some layers and adding/reducing their complexity. Our hyper parameters above resulted in what we determine to be the most realistic to our music recognition problem.

We have troubles with determining a stable and compatible loss function as our dataset 

<!-- We are looking for an interpretation of the result. -->

<!-- You may want to refer to your data summary and hyperparameter choices to make your argument. -->


## Ethical Consideration

All the music in the MTG-Jamendo dataset is available under a Creative Commons Attribution Non-Commercial Share Alike license.

There are several limitations of our model.

1. Can only make predictions on 30s audio samples, meaning that we can't effectively classify longer pieces of music, without first dividing it into sections, furthermore if there is a sample shorter than 30s, then it has to be padded for a prediction to be made, and it will likely not be very high quality. Shorter music samples may arise in certain commercial applications like the advertising industry, or video game development, where this model could be of use.
2. Model is trained on data that is "subjectively" labelled, and not everyone may agree on some of the labels.

We see our model as helpful, as it can be used by other audio/music hosting applications when attempting to sort their data by categorizing their music libraries. Moreover, it could highlight similarities between genres, and may uncover certain patterns between them, which could augment research in the musical space.

## Authors 
- Tomasz Cieslak
- Mina Makar
- Daren Liang
- George Lewis


Data subset breakdown
https://github.com/MTG/mtg-jamendo-dataset/tree/master/data