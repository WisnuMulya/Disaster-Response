import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads disaster messages and disaster categories CSVs and returns merged
    dataframe.
    INPUT:
    messages_filepath (str) - disaster messages file path
    categories_filepath (str) - disaster categories file path
    OUTPUT:
    df (Pandas DataFrame) - merged dataframe from messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Transforms categories and clean duplicates and inconsistent data.
    INPUT:
    df (Pandas Dataframe) - disaster response dataframe
    OUTPUT:
    clean_df (Pandas DataFrame) - cleaned disaster response dataframe
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # Name columns by extracting the first row in 'categories'
    row = categories.iloc[0]
    category_colnames = row.str[:-2]
    categories.columns = category_colnames

    # Convert categories values to just numbers 0 and 1
    categories = categories.applymap(lambda value: int(value[-1]))
    ## Change values to either 0 or 1 for column 'related' which has a value of 2
    categories['related'] = categories['related'].apply(lambda value: value%2)

    # Replace 'categories' column in df with new category columns
    clean_df = df.drop('categories', axis=1)
    clean_df = pd.concat([clean_df, categories], axis=1)

    # Remove duplicates
    clean_df = clean_df.loc[~clean_df.duplicated()]

    return clean_df


def save_data(df, database_filename):
    """
    Save dataframe in a SQL database.
    INPUT:
    df (Pandas DataFrame) - disaster response dataframe
    database_filename (str) - the name of the database file
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """
    Run the main function of the program: clean and transform CSV data and save
    to SQL.
    """
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
