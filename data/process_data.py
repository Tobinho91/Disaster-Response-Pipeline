import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    output:
        df: The merged dataset
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    
    # merge datasets
    df = pd.merge(messages,categories, how='outer', on = 'id')
    #df.head()
    return df


def clean_data(df):
    '''
    input:
        df: The merged dataset in previous step.
    output:
        df: Dataset after cleaning.
    '''
    # select the first row of the categories dataframe
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    #category_colnames 

    category_colnames = []
    for names in row:
        #new_name = names.split("/")
        new_name = names[0:-2]
        category_colnames.append(new_name)
   
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head(10)
    
    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    #df.head()
    
    # concatenate the original dataframe with the new `categories` dataframe
    frames = [df, categories]
    df = pd.concat (frames, axis = 1, sort = False)
    #df.head()
    
    #remove duplicates:
    # check number of duplicates
    print('The amount of duplicates is ', sum(df.duplicated()), 'in the dataset')
    
    #drop duplicates
    df.drop_duplicates(inplace = True)
    
    #check number of duplicates
    print('The amount of duplicates is ', sum(df.duplicated()), 'in the dataset')
    
    #Handling of values of 2 in column related:
    # Remove values with a 2
    df = df[df['related'] != 2]
    
    #remove nan
    df = df.fillna(0)
    
    return df


def save_data(df, database_filepath):
    
    '''
    input:
        df: The merged and cleaned dataset in previous step.
    output:
        database_filename: a filename in a database
    '''
            
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('messages_disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()