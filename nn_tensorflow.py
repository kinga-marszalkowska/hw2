import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# convert txt file to pandas dataframe
data = pd.read_csv('train-io.txt', sep=" ", header=None)
data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "target"]

y = data["target"].to_numpy()
X = data.drop(["target"], axis=1).to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# clear files
with open("test-o-hat-expected.txt", 'w') as file, open("test-o-hat.txt", 'w') as file2:
    pass


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class Data(Dataset):
    def __init__(self, X_train, y_train):
        # need to convert float64 to float32 else
        # will get the following error
        # RuntimeError: expected scalar type Double but found Float
        self.X = torch.from_numpy(X_train.astype(np.float32))
        # need to convert float64 to Long else
        # will get the following error
        # RuntimeError: expected scalar type Long but found Float
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    traindata = Data(X_train, Y_train)

    batch_size = 32
    trainloader = DataLoader(traindata, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    # number of features (len of X cols)
    input_dim = 10
    # number of classes (unique of y)
    output_dim = 2

    clf = Network()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

    loss_vals = []
    epoch_plot = []

    epochs = 170
    for epoch in range(epochs):
        running_loss = 0.0
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

            loss_vals.append(running_loss)
            epoch_plot.append(epoch)

        # display statistics
        print(f'[{epoch + 1}/{epochs} epochs] loss: {running_loss / 2000:.5f}')

    # save the trained model
    PATH = './mymodel1.pth'
    torch.save(clf.state_dict(), PATH)

    # load model
    clf = Network()
    clf.load_state_dict(torch.load(PATH))

    # load test data
    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=batch_size,
                            shuffle=True, num_workers=2
                            )
    # testing and assessing data
    correct, total = 0, 0

    # no need to calculate gradients during inference
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # calculate output by running through the network
            outputs = clf(inputs)
            # get the predictions
            __, predicted = torch.max(outputs.data, 1)
            with open("test-o-hat.txt", "a") as f:
                for i in predicted:
                    f.write(str(i.item()) + "\n")
            with open("test-o-hat-expected.txt", "a") as f:
                for i in labels:
                    f.write(str(i.item()) + "\n")

            # update results
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {len(testdata)} test data: {100 * correct // total} %')

    # print(loss_vals)
    print(len(loss_vals))
    # plt.plot(epoch_plot, loss_vals, 'bo', label='Training loss')
    # # plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
    # plt.title('Training loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
