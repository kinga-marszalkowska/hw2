import pandas as pd
import numpy as np
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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# convert txt file to pandas dataframe
data = pd.read_csv('train-io.txt', sep=" ", header=None)
data.columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "target"]

# print(data.head())
print(data["target"].value_counts())

y = data["target"]
X = data.drop(["target"], axis=1)

# splitting the dataset into training validation and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

# preprocess with standard scaler
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_val = StandardScaler()
X_val = ss_val.fit_transform(X_val)

# prepare many models for experiment
models = {'Logistic Regression': LogisticRegression(), 'Support Vector Machines': LinearSVC(),
          'Decision Trees': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(),
          'Naive Bayes': GaussianNB(), 'K-Nearest Neighbor': KNeighborsClassifier()}

accuracy, precision, recall = {}, {}, {}

# hyperparameter tuning
# grid = {"C": np.array([0.001, 0.01, 0.1, 1, 10]), "penalty": ["l1", "l2"]}
# logreg_cv = GridSearchCV(logreg, grid, cv=10)
# logreg_cv.fit(X_val, y_val)
# print('The Best Penalty:', logreg_cv.best_estimator_.get_params()['penalty'])
# print('The Best C:', logreg_cv.best_estimator_.get_params()['C'])

pipe = Pipeline([('classifier', RandomForestClassifier())])

param_grid = [
    {'classifier': [LogisticRegression()],
     'classifier__penalty': ['l1', 'l2'],
     'classifier__C': np.logspace(-4, 4, 20),
     'classifier__solver': ['liblinear']},
    {'classifier': [RandomForestClassifier()],
     'classifier__n_estimators': list(range(10, 101, 10)),
     'classifier__max_features': list(range(2, 6, 10))}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

# Fit on data

best_clf = clf.fit(X_train, y_train)

# Make predictions
predictions = best_clf.predict(X_val)

# Calculate metrics
print(best_clf.best_estimator_)
print(f"Accuracy: {accuracy_score(predictions, y_val)}")

for key in models.keys():
    # Fit the classifier
    models[key].fit(X_train, y_train)

    # Make predictions
    predictions = models[key].predict(X_val)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_val)
    precision[key] = precision_score(predictions, y_val)
    recall[key] = recall_score(predictions, y_val)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)

ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()),
    bbox_to_anchor=(0, 1),
    loc='lower left',
    prop={'size': 14}
)
plt.tight_layout()
