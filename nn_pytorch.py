import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import random

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.1
OUTPUT_FILE_NAME = "test-o-hat_1.3.txt"

# for reproducibility of results
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

# convert txt file to pandas dataframe
data = pd.read_csv('train-io.txt', sep=" ", header=None)
data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "target"]

y = data["target"].to_numpy()
X = data.drop(["target"], axis=1).to_numpy()


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# clear files
with open("test-o-hat-expected.txt", 'w') as file, open(OUTPUT_FILE_NAME, 'w') as file2:
    pass


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        # batch_size x 1 => batch_size dimensions
        return torch.sigmoid(x).squeeze(-1)


class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))

        self.y = torch.from_numpy(y_train.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device}")

    traindata = Data(X_train, Y_train)

    trainloader = DataLoader(traindata, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)

    clf = Network(input_dim=10, output_dim=1)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(clf.parameters(), lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    loss_vals, val_loss_vals = [], []
    epoch_plot = []

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

            #todo sprwdzic czy labelki zgadzaja sie z featurami
            #print(f"{inputs[0]} {labels[0]}")

        running_loss /= i+1
        epoch_plot.append(epoch)
        loss_vals.append(running_loss)


        # load val data
        testdata = Data(X_test, Y_test)
        testloader = DataLoader(testdata, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2
                                )
        # testing and assessing data
        correct, total = 0, 0

        # no need to calculate gradients during inference
        with torch.no_grad():
            running_val_loss = 0.0
            i = 0
            for i, data in enumerate(testloader, 0):
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

        print(f'Accuracy of the network on the {len(testdata)} validation data: {100 * correct / total} %')
        running_val_loss /= i + 1
        val_loss_vals.append(running_val_loss)
        scheduler.step()

        print(f'[{epoch + 1}/{EPOCHS} epochs] loss: {running_loss:.5f} val loss: {running_val_loss:.5f} LR: {scheduler.get_last_lr()}')

    # save the trained model
    PATH = './mymodel_1.2.pth'
    torch.save(clf.state_dict(), PATH)


    plt.plot(epoch_plot, loss_vals, 'bo', label='Training loss')
    plt.plot(epoch_plot, val_loss_vals, 'orange', label='Validation loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # test set

    # convert txt file to pandas dataframe
    data = pd.read_csv('test-i.txt', sep=" ", header=None)
    data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]

    X = data.to_numpy()

    # load model
    clf = Network(input_dim=10, output_dim=1)
    clf.load_state_dict(torch.load(PATH))

    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2
                            )

    for data in testloader:
        inputs, labels = data
        # calculate output by running through the network
        outputs = clf(inputs)

        with open(OUTPUT_FILE_NAME, "a") as f:
            for i in outputs:
                if i.item() >= 0.5:
                    f.write(str(1) + "\n")
                else:
                    f.write(str(0) + "\n")
    # with open("test-o-hat-expected.txt", "a") as f:
    #     for i in labels:
    #         f.write(str(i.item()) + "\n")