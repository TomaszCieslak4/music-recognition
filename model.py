import torch
import torch.nn as nn
import math
import torch.optim as optim
import matplotlib.pyplot as plt

from preprocess import MusicDataset

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using device {device}")
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

dataset = MusicDataset("./autotagging_moodtheme.tsv", "dump-spec-trimmed/")
data_len = len(dataset)
train_size = math.floor(data_len*0.8)
leftover = data_len - train_size
val_size = leftover//2
test_size = leftover - val_size

train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print("Number of images in the training set: ", len(train_data))
print("Number of images in the val set: ", len(val_data))
print("Number of images in the test set: ", len(test_data))



class RNN1(nn.Module):
    def __init__(self, input_channel, hidden_channel, linear_transform, output_channel, classes):
        super(RNN1, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, output_channel,  3, padding=1)
        
        self.conv2 = nn.Conv1d(output_channel, output_channel * 2, 3, padding=1)

        # RESHAPING?
        
        self.lstm1 = nn.LSTM(350, hidden_channel, num_layers=3, bias=True)

        self.linear1 = nn.Linear(hidden_channel, linear_transform, bias=True)

        self.linear2 = nn.Linear(linear_transform, classes, bias=True)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
       
        
    def forward(self, inp, hidden=None):
        x = self.pool(self.relu(self.conv1(inp)))
        x = self.pool(self.relu(self.conv2(x)))

        # RESHAPING?
        # x = x.view(-1, self.out_channels * 4 * 28 * 28)
        
        # hidden = (torch.zeros(num_layers, 256, hidden_size), torch.zeros(num_layers, batch_size, hidden_size))

        x, hidden = self.lstm1(x, hidden) #
        x = x[:, -1]
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x, hidden
    
def get_accuracy(model, dataset):

    loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    model.eval().to(device)

    incorrect = 0
    total = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        total += target.shape[1]*target.shape[0]
        res, _ = model(data)
        res.to(device)
        incorrect += torch.count_nonzero((torch.where(res > 0, 1, 0).to(device) - target)).to(device)

    return (total - incorrect)/total

def train(model, train_data, batch_size, num_epochs, learning_rate, weight_decay, classes):

    loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay)
    
    lossBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([29.5]*classes).to(device)).to(device)

    all_epoch, all_val_acc, all_train_acc, all_train_cost = [], [], [], []

    iteration = 0

    for epoch in range(num_epochs):

        sum_loss_per_song = 0
        num_songs = 0

        for data, target in loader:
           data = data.to(device)
           target = target.to(device)
           if data.shape[0] != batch_size:
               continue
           
           model.train().to(device)
           res, _ = model(data)
           loss = lossBCE(res, target.float())
              
           loss.backward()

           optimizer.step()
           optimizer.zero_grad()

           sum_loss_per_song += loss
           num_songs += batch_size
           iteration += 1

        all_epoch.append(epoch)

        train_cost = sum_loss_per_song /  num_songs
        all_train_cost.append(train_cost.item())

        val_acc = get_accuracy(model, val_data)
        all_val_acc.append(val_acc.item())

        train_acc = get_accuracy(model, train_data)
        all_train_acc.append(train_acc.item())

        print("Epoch %d. Iteration %d. [Val Acc %.4f%%] \
              [Train Acc %.4f%%, Loss %f]" % (
        epoch, iteration, val_acc * 100, train_acc * 100,
        train_cost))

    """
    Plot the learning curve.
    """
    plt.title("Learning Curve: Loss per Epoch")
    plt.plot(all_epoch, all_train_cost, label="Train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve: Accuracy per Epoch")
    plt.plot(all_epoch, all_train_acc, label="Train")
    plt.plot(all_epoch, all_val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

# Train model
model1 = RNN1(input_channel=96, hidden_channel=16, linear_transform=32, output_channel=8, classes=59)
model1.to(device)
train(model1, train_data, batch_size=256, num_epochs=100, learning_rate=0.01, weight_decay=0.001, classes=59)

# Obtain test acc
acc = get_accuracy(model1, test_data)
print("RNN model provided a test accuracy of: ", acc.item())