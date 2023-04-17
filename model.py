import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.classification import (
    multilabel_accuracy,
    multilabel_f1_score,
)

from preprocess import MusicDataset

dataset = MusicDataset("./autotagging_genre.tsv", "dump-spec-trimmed/")
data_len = len(dataset)
train_size = math.floor(data_len * 0.8)
leftover = data_len - train_size
val_size = leftover // 2
test_size = leftover - val_size

train_data, val_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

print("Number of images in the training set: ", len(train_data))
print("Number of images in the val set: ", len(val_data))
print("Number of images in the test set: ", len(test_data))


class RNN1(nn.Module):
    def __init__(
        self,
        input_channel,
        hidden_channel,
        linear_transform,
        output_channel,
        sequence_length,
        classes,
    ):
        super(RNN1, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, output_channel, 3, padding=1)

        self.conv2 = nn.Conv1d(
            output_channel, output_channel * 2, 3, padding=1
        )  # TODO: Adjust channels

        self.lstm1 = nn.LSTM(
            sequence_length // 4, hidden_channel, num_layers=3, bias=True
        )  # TODO: Adjust lstm

        self.linear1 = nn.Linear(hidden_channel, linear_transform, bias=True)

        self.linear2 = nn.Linear(linear_transform, classes, bias=True)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # Sigmoid is not needed because we are using BCEWithLogitsLoss
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inp, hidden=None):
        x = self.pool(self.relu(self.conv1(inp)))
        x = self.pool(self.relu(self.conv2(x)))

        # RESHAPING?
        # x = x.view(-1, self.out_channels * 4 * 28 * 28)

        # hidden = (torch.zeros(num_layers, 256, hidden_size), torch.zeros(num_layers, batch_size, hidden_size))

        x, hidden = self.lstm1(x, hidden)
        x = x[:, -1]
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x, hidden


def get_f1_score(model, dataset, num_classes):
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    model.eval()

    score = 0
    count = 0
    for data, target in loader:
        count += data.shape[0]
        res, _ = model(data)
        res_binary = torch.where(res > 0, 1, 0)
        f1_score = multilabel_f1_score(res_binary, target, num_classes)
        score += f1_score * data.shape[0]

    return score / count


def get_accuracy(model, dataset, num_classes):
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    model.eval()

    score = 0
    count = 0
    for data, target in loader:
        count += data.shape[0]
        res, _ = model(data)
        res_binary = torch.where(res > 0, 1, 0)
        f1_score = multilabel_accuracy(res_binary, target, num_classes)
        score += f1_score * data.shape[0]

    return score / count


# def weighted_cross_entropy_with_logits(logits, targets, pos_weight): # https://stackoverflow.com/a/49104640
#     return targets * -logits.log() * pos_weight + (1 - targets) * -(1 - logits).log()


def train(
    model, train_data, batch_size, num_epochs, learning_rate, weight_decay, classes
):
    loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    lossBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([35] * classes))

    all_epoch, all_val_f1_score, all_train_f1_score, all_train_cost = [], [], [], []

    iteration = 0

    for epoch in range(num_epochs):
        sum_loss_per_song = 0
        num_songs = 0

        for data, target in loader:
            if data.shape[0] != batch_size:
                continue

            model.train()
            res, _ = model(data)
            loss = lossBCE(res, target.float())
            #    loss = weighted_cross_entropy_with_logits(res, target.float(), 1)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            sum_loss_per_song += loss
            num_songs += batch_size
            iteration += 1

        all_epoch.append(epoch)

        train_cost = sum_loss_per_song / num_songs
        all_train_cost.append(train_cost.item())

        val_f1_score = get_f1_score(model, val_data, classes)
        all_val_f1_score.append(val_f1_score)

        val_acc = get_accuracy(model, val_data, classes)

        train_f1_score = get_f1_score(model, train_data, classes)
        all_train_f1_score.append(train_f1_score)

        train_acc = get_accuracy(model, train_data, classes)

        # TODO: Checkpointing

        # if model_name != "":
        #     PATH = "checkpoints/" + model_name + "/" + \
        #     "epoch" + str(epoch) + ".pt"
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'loss': train_cost,
        #         'iteration': iteration,
        #         'val_acc': val_acc * 100,
        #         'train_acc':  train_acc * 100
        #         }, PATH)

        print(
            "Epoch %d. Iteration %d. [Val F1 %.4f%%, Val Acc %.4f%%] \
              [Train F1 %.4f%%, Train Acc %.4f%%, Loss %f]"
            % (
                epoch,
                iteration,
                val_f1_score * 100,
                val_acc * 100,
                train_f1_score * 100,
                train_acc * 100,
                train_cost,
            )
        )

    """
    Plot the learning curve.
    """


model1 = RNN1(
    input_channel=96,
    hidden_channel=16,
    linear_transform=32,
    output_channel=8,
    sequence_length=1400,
    classes=95,
)
# print(get_accuracy(model1, train_data))
train(
    model1,
    train_data,
    batch_size=256,
    num_epochs=400,
    learning_rate=0.1,
    weight_decay=0.0,
    classes=95,
)
