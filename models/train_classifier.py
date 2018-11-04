import sys
import pickle
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import nltk
import numpy as np

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def load_data(database_filepath):
    """
    Loads and process the sql database from filepath
    returns Feature set(X) labels(y) and category_names from database.
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('DisasterData', engine)
    df = df[df['related'] != 2]
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Buils string tokens from text sentences.

    returns a liist of tokens.
    """

    tokenizer = nltk.tokenize.casual.TweetTokenizer(preserve_case=False)
    return tokenizer.tokenize(text)


def build_model():
    """
    Builds the ML pipeline model.

    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(XGBClassifier(n_jobs=-1)))
    ])
    parameters = {
        "clf__estimator__n_estimators": [200,300]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv
    

def evaluate_model(model, X_test, y_test, category_names):
    """
    Shows model metrics, for each category.

    """
    y_pred = model.predict(X_test)
    
    print(classification_report(y_pred, y_test, target_names=category_names))
    
    f1 = f1_score(y_test, y_pred, average='micro')
    p = precision_score(y_test, y_pred, average='micro')
    r = recall_score(y_test, y_pred, average='micro')
    a = accuracy_score(y_test, y_pred)

    print("f1_score", f1)
    print("precision_score", p)
    print("recall_score", r)
    print("accuracy_score", a)


def save_model(model, model_filepath):
    """
    Saves the trained model for later usage.
    """
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2)
        
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
