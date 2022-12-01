import pandas as pd
from nn_pytorch import Network, Data, DataLoader, y_train
import torch
import zipfile

BATCH_SIZE = 32
OUTPUT_FILE_NAME = "test-o-hat.txt"
PATH = MODEL_NAME = './mymodel_1.8.pth'

data = pd.read_csv('test-i.txt', sep=" ", header=None)
data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]

X = data.to_numpy()

with open(OUTPUT_FILE_NAME, 'w') as file2:
    pass

# load model
clf = Network(input_dim=10, output_dim=1)
clf.load_state_dict(torch.load(PATH))

testdata = Data(X, y_train)
testloader = DataLoader(testdata, batch_size=BATCH_SIZE,
                        shuffle=True
                        )

file = open(OUTPUT_FILE_NAME, "a")
results = []

for data in testloader:
    inputs, _ = data
    outputs = clf(inputs)

    for i in outputs:
        if i >= 0.5:
            results.append(str(1))
        else:
            results.append(str(0))

file.write("\n".join(results))
with zipfile.ZipFile("labels.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(OUTPUT_FILE_NAME)


