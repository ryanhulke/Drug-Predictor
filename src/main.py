import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("input/drug200.csv")


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# changes "M" and "F" to 1 and 0, same with BP and cholestorol
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = df["Drug"].values

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print("X train set: " + str(X_trainset.shape) + ", X test set: " + str(X_testset.shape))
print("y train set: " + str(y_trainset.shape) + ", y test set: " + str(y_trainset.shape) + "\n")

# choose model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

# train model
drugTree.fit(X_trainset,y_trainset)

# test trained model on the test set
predTree = drugTree.predict(X_testset)

# see a few predictions to check accuracy at a glance
print (predTree [0:5])
print (y_testset [0:5])

# measure accuracy
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# see the created tree
tree.plot_tree(drugTree)
plt.show()