import pandas as pd
import plotly.graph_objects as go 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
# Exploratory

# statistical parameters
def stat_param(data):
    print(data.describe())

# scatter plot
def scatter(data):

    layout = go.Layout(
        title="Height x Width",
        xaxis_title="Height",
        yaxis_title="Width"
        )

    fig = go.Figure(
        data=go.Scatter(x=data['Height'], y=data['Width'],mode='markers'), 
        layout=layout)
         
    fig.show()


# line plot
def visualize_data(data):  
     
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data['Length1'],
                        mode='lines',
                        name='Length1'))
    fig.add_trace(go.Scatter(y=data['Length2'],
                        mode='lines',
                        name='Length2'))
    fig.add_trace(go.Scatter(y=data['Weight'],
                        mode='lines',
                        name='Weight'))
    fig.show()
    
# ALTERATE BRANCH 
def alterate_branch(data):
    
    c1 = pd.DataFrame(data[['Length1','Height','Width','Species']])
    d1 = pd.DataFrame(data[['Weight']])
    w1 = pd.get_dummies(c1, columns=["Species"])

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(c1,d1,random_state=1,test_size=0.2)

if __name__ == "__main__":
    fish_df = pd.read_csv('./fish.csv')
    print(fish_df.head(5))
    
    stat_param(fish_df)

    scatter(fish_df)
    visualize_data(fish_df)

    decision_tree(fish_df)
    
    
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
