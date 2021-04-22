
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

fish=pd.read_csv('fish.csv')

#visualization
plt.figure(figsize=(12,8))
sns.countplot(fish['Species'])
plt.show()

# We can look at an individual feature in Seaborn through a

plt.figure(figsize=(12,8))
sns.boxplot(x="Species", y="Weight", data=fish)
plt.show()

#we will split our data to dependent and independent
#first dependent data
X=fish.iloc[:,1:]

#second independent
# we add more [] to make it 2d array
y=fish[["Species"]]

#split our data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 42)
classifier.fit(X_train, y_train)
print(classifier.score(X_test,y_test))