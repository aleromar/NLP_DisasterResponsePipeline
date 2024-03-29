import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(messages_filepath, categories_filepath):
"""
Creates a new dataframe by combining information
from messages and categories

Args:
    messages_filepath: Path where the message file is located,
                       including filename
    categories_filepath: Path where the categories file is located,
                         including filename

Returns:
    A dataframe that contains information from both input files.
"""
    messages = pd.read_csv(messages_filepath)
    try:
        messages.drop(columns='original',inplace=True)
    except KeyError:
        print('Could not remove column original')
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id',how='outer')
    return df

def clean_data(df):
"""
Performs data cleaning on a dataframe

Args:
    df: dataframe to be cleaned

Returns:
    A dataframe that results on cleaning the input dataframe
"""
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = [x.split('-')[0] for x in row] 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.columns = category_colnames
    categories.related = categories.related.replace(2,1)
    df.drop(columns='categories',inplace=True)
    df = pd.concat([df, categories],axis=1)
    df.drop_duplicates(inplace=True)
    df.drop(columns='id',inplace=True)
    return df

def save_data(df, database_filename):
"""
Saves data stored in a dataframe into an sql database

Args:
    df: dataframe that includes the information to be saved
    database_filename: Path where the database needs to be created.

"""
    engine = create_engine('sqlite:///'+database_filename) 
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')


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