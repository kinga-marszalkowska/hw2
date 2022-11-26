import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

# convert txt file to pandas dataframe
data = pd.read_csv('train-io.txt', sep=" ", header=None)
data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "target"]


print(data["target"].value_counts())

y = data["target"]
X = data.drop(["target"], axis=1)

for i in data.columns:
    plt.hist(data[i], bins=10)
    plt.show()
    plt.title(i)

b_plot = X.boxplot()
b_plot.plot()
plt.show()
plt.title("Boxplots of all variables")

