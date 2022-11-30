import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import random

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OUTPUT_FILE_NAME = "test-o-hat_1.6.txt"
MODEL_NAME = './mymodel_1.6.pth'

# set seeds for reproducibility of results
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

data = pd.read_csv('train-io.txt', sep=" ", header=None)
data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "target"]

y = data["target"].to_numpy()
X = data.drop(["target"], axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42) # 0.1 x 0.9 = 0.18

# clear files
with open("test-o-hat-expected.txt", 'w') as file, open(OUTPUT_FILE_NAME, 'w') as file2:
    pass


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        # batch_size x 1 => batch_size dimensions
        return torch.sigmoid(x).squeeze(-1)


class Data(Dataset):
    def __init__(self, X_train, y_train=None):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':

    # 1. Load training data
    traindata = Data(X_train, y_train)
    trainloader = DataLoader(traindata, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)

    # 2. Define classifier
    clf = Network(input_dim=10, output_dim=1)
    print(clf)

    # 3. Define loss function, optimizer and learning rate scheduler
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    loss_vals, val_loss_vals = [], []
    epoch_plot = []

    # 4. Training loop
    for epoch in range(EPOCHS):
        running_loss = 0.0
        i = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        running_loss /= i+1
        epoch_plot.append(epoch)
        loss_vals.append(running_loss)

        # load validation data
        valdata = Data(X_val, y_val)
        valdataloader = DataLoader(valdata, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)

        correct, total = 0, 0
        # no need to calculate gradients during inference
        with torch.no_grad():
            running_val_loss = 0.0
            i = 0
            for i, data in enumerate(valdataloader, 0):
                inputs, labels = data
                # calculate output by running through the network
                outputs = clf(inputs)
                # get the predictions
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                # update results
                total += labels.size(0)
                outputs = outputs >= 0.5
                correct += (outputs == labels).sum().item()

        print(f'Accuracy of the network on the {len(valdata)} validation data: {100 * correct / total} %')
        running_val_loss /= i + 1
        val_loss_vals.append(running_val_loss)

        #scheduler.step()

        print(f'[{epoch + 1}/{EPOCHS} epochs] loss: {running_loss:.5f} val loss: {running_val_loss:.5f} LR: {scheduler.get_last_lr()}')

    # 5. Saving the trained model
    PATH = MODEL_NAME
    torch.save(clf.state_dict(), PATH)

    plt.plot(epoch_plot, loss_vals, 'bo', label='Training loss')
    plt.plot(epoch_plot, val_loss_vals, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 6. Load held out test data
    test1data = Data(X_test, y_test)
    test1dataloader = DataLoader(test1data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    correct, total = 0, 0

    with torch.no_grad():
        running_val_loss = 0.0
        i = 0
        for i, data in enumerate(test1dataloader, 0):
            inputs, labels = data
            outputs = clf(inputs)

            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            total += labels.size(0)

            outputs = outputs >= 0.5
            correct += (outputs == labels).sum().item()

    print(f'Accuracy of the network on the {len(test1data)} test data: {100 * correct / total} %')

    # 7. final assignment test set
    data = pd.read_csv('test-i.txt', sep=" ", header=None)
    data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]

    X = data.to_numpy()

    # load model
    clf = Network(input_dim=10, output_dim=1)
    clf.load_state_dict(torch.load(PATH))

    # the second parameter doesn't matter - it needs to by numpy array
    testdata = Data(X, y_train)
    testloader = DataLoader(testdata, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2
                            )

    for data in testloader:
        inputs, _ = data
        outputs = clf(inputs)

        with open(OUTPUT_FILE_NAME, "a") as f:
            for i in outputs:
                if i >= 0.5:
                    f.write(str(1) + "\n")
                else:
                    f.write(str(0) + "\n")
