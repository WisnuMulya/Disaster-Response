import sys
import numpy as np
import pandas as pd
import sqlite3
import pickle

# NLTKs
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Scikit Learn Modules
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Loads disaster dataframe from a SQL database.
    INPUT:
    database_filepath (str) - path to the SQL database
    OUTPUT:
    X (numpy array) - numpy array of messages dataset
    Y (numpy array) - numpy array of labels dataset
    category_names (list of str) - list of category/label names
    """
    # Load data from database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM messages', conn)

    category_names = df.columns[4:].tolist()
    X = np.array(df['message'])
    Y = np.array(df[category_names])

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes, lemmatizes, and normalizes text.
    INPUT:
    text (str) - text to be procesed
    OUTPUT:
    clean_tokens (list of str) - list of clean tokens
    """
    # Replace URLs with 'urlplaceholder' string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Filter to include only alphabets or numbers and normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)

    # Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Clean tokens: lemmatize, filter stopwords
    stop_words = stopwords.words("english")
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    An estimator to evaluate whether there is a verb starting
    a sentence in the message.
    """
    def starting_verb(self, text):
        """
        Returns True if there is a verb starting a sentence.
        and False otherwise.
        INPUT:
        text (str) - a string of message
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # If nltk.pos_tag() finds words
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
                # If the first word is a verb or a retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        """
        A placeholder self.fit() required as an estimator.
        """
        return self

    def transform(self, X):
        """
        Running the self.starting_verb() function on messages.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Returns a model which has been pipelined with multiple estimators.
    """
    # Pipeline estimators
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ], verbose=1)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, verbose=1)))
    ], verbose=True)
    
    # Search for the best parameters of the pipeline
    # NOTE: uncomment parameters you'd like to optimize on
    # NOTE: depending on computing capacity, the GridSearchCV might take a long time
    parameters = {
#        'vect__ngram_range': ((1, 1), (1, 2)),
#        'vect__max_df': (0.5, 0.75, 1.0),
#        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
#        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [100, 200],
#        'clf__estimator__min_samples_split': [2, 3, 4],
    }
    model = GridSearchCV(pipeline, parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print out accuracy of the model and F1 score analysis.
    INPUT:
    model (model object) - a model to be evaluated
    X_test (numpy array) - an array containing messages dataset
    Y_test (numpy array) - an array containing multi-labels dataset
    category_names (list) - a list of category names
    """
    # Print accuracy and best parameters
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()

    print('-----------------------------------------------------')
    print('Accuracy: {}'.format(accuracy))
    print('Best Parameters: {}'.format(model.best_params_))
    print('-----------------------------------------------------')

    # Print F1 scores
    for idx, category in enumerate(category_names):
        print('Category: {}'.format(category))
        print(classification_report(Y_test[:,idx], Y_pred[:,idx]))
        print('-----------------------------------------------------')
    

def save_model(model, model_filepath):
    """
    Saves model to a pickle file.
    INPUT:
    model (model object) - the model to be saved
    model_filepath (str) - saving path for the model
    """
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    file.close()


def main():
    """
    Run the main function of the program: create and save a model trained
    on the SQL disaster response database.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
