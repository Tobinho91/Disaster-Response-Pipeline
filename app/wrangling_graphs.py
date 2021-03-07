import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from sqlalchemy import create_engine
from plotly.graph_objs import Bar
from sklearn.externals import joblib

def return_graphs():

    """Creates three plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the three plotly visualizations

    """
    # load data
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table("messages_disaster", engine)
    
    
    #data for vizualizations:
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Show distribution of different category
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))

    # extract data exclude related
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories_mean.index)
    
    
    
    #first chart plots bar chart 
    
    graph_one = []
    
    graph_one.append(
      go.Bar(
              x=genre_names,
              y=genre_counts,
              #color="genre_names"
              marker=dict(
            color='LightSkyBlue')
      )
    )
    
    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Count'),
                yaxis = dict(title = 'Genre'),
                )

    #second chart plots bar chart 
    
    graph_two = []
    
    graph_two.append(
      go.Bar(
              x=categories_names,
              y=categories_mean,
              marker=dict(
            color='indianred')
      )
    )
    
    
    layout_two = dict(title = 'Top 10 of Message Categories',
                xaxis = dict(title = 'Percent'),
                yaxis = dict(title = 'Categories'),
                )
   
    #third chart plots bar chart 
    
    graph_three = []
    
    graph_three.append(
      go.Bar(
             x=category,
             y=category_counts
      )
    )
    
    
    layout_three = dict(title = 'Distribution of Message Categories',
                xaxis = dict(title = 'Count'),
                yaxis = dict(title = 'Category'),
                )
    
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    
    return graphs
