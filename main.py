import pandas as pd
import plotly.graph_objects as go 

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
    
if __name__ == "__main__":
    fish_df = pd.read_csv('fish.csv')
    print(fish_df.head(5))
    
    visualize_data(fish_df)