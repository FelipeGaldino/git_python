import pandas as pd
import plotly.graph_objects as go 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn import tree

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
    fig.add_trace(go.Scatter(y=data['Length3'],
                        mode='lines',
                        name='lines'))
    fig.add_trace(go.Scatter(y=data['Length2'],
                        mode='lines',
                        name='lines'))
    fig.add_trace(go.Scatter(y=data['Height'],
                        mode='lines',
                        name='lines'))
    fig.show()
    

# Model: decision tree classifier
def decision_tree(data):

    X1 = pd.DataFrame(data[['Length1','Height','Width','Species']])
    y1 = pd.DataFrame(data[['Weight']])
    X1 = pd.get_dummies(X1, columns=["Species"])

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X1,y1,random_state=1,test_size=0.2)

    mms = MinMaxScaler()
    X_train_norm_1 = mms.fit_transform(X_train_1)
    X_test_norm_1 = mms.transform(X_test_1)

    scaler = StandardScaler()
    scaler.fit(X_train_1)
    X_train_std_1 = scaler.transform(X_train_1)
    X_test_std_1 = scaler.transform(X_test_1)

    xx = np.arange(len(X_train_std_1))
    yy1 = X_train_norm_1[:,0]
    yy2 = X_train_std_1[:,0]

    dtr_model = dtr(splitter='random')
    dtr_model.fit(X_train_norm_1,y_train_1)

    print("DecisionTreeClassifier Train Score: ", dtr_model.score(X_train_norm_1, y_train_1))
    print("DecisionTreeClassifier Test Score: ", dtr_model.score(X_test_norm_1, y_test_1))

    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(15,15),dpi=300)
    tree.plot_tree(dtr_model,filled=True)
    plt.show()


if __name__ == "__main__":
    fish_df = pd.read_csv('./fish.csv')
    print(fish_df.head(5))
    
    stat_param(fish_df)

    scatter(fish_df)
    visualize_data(fish_df)

    decision_tree(fish_df)


 

