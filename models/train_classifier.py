import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
from sklearn.externals import joblib

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
    """
    Determines whether a sentence starts with a verb
    
    Args:
        text: the sentence to be analysed
    
    Returns:
        True if a verb is the first word, otherwise False
    """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            #print(len(pos_tags))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
    """
    Fits this custom estimator
    
    Args:
        x: independent features
        y: dependant variable
    
    Returns:
        The custom estimator
    """
        return self

    def transform(self, X):
    """
    Applies starting_verb to a series of messages
    
    Args:
        X: the dataframe that contains a set of messages to be processed.
    
    Returns:
        A dataframe resulting in processing the input dataframe
    """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
class ArmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,n_estimators=100,min_samples_split=2):
        print('ARM: {},{}'.format(n_estimators,min_samples_split))
        self.m_nestimators = n_estimators
        self.m_minsamplessplit = min_samples_split
        self.classy = MultiOutputClassifier(RandomForestClassifier(n_estimators=n_estimators,min_samples_split=min_samples_split))
    
    def fit(self,X,y=None):
        return self.classy.fit(X,y)
        
    def set_params(self, **params):
        print('ARM Setparams: {}'.format(params))
        self.classy.set_params(**params)
    
    def predict(self,X,y=None):
        return self.classy.predict(X)

def load_data(database_filepath):
"""
Reads information from a database and creates three output elements.

Args:
    database_filepath: Path where the database file is located,
                       including filename

Returns:
    X: A numpy matrix containing all the features
    y: A numpy vector containing the labels
    categories: names (of features) for each of the columns in X in order
"""
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponseTable',engine)
    X = df[['message']].values.flatten()#,'genre']]
    categories = [colname for colname in df.columns if colname not in ['message','genre']]
    y = df[categories].values
    return X,y,categories

def tokenize(text):
"""
Reads information from a database and creates three output elements.

Args:
    text: string to be tokenized

Returns:
    tokens: a list of words (tokens) contained in the input string
"""
    alphanumregex = r'[a-z]'
    p = re.compile(alphanumregex)
    text = text.lower().strip()
    # tokenize text
    tokens = word_tokenize(text)
    # Lemmatize
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    # Remove stop words
    tokens = [token.strip() for token in tokens if token not in stopwords.words('english')]
    # Remove tokens that are not words
    tokens = [token for token in tokens if (len(p.findall(token))>0)]
    return tokens

def build_model():
"""
Creates a model object that can be trained and used for predictions
It uses an ML pipeline and grid search for optimization

Args:
    None

Returns:
    cv: the model to use
"""
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) #ArmClassifier())
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [100, 200],
        #'clf__estimator__min_samples_split': [2, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }
    cv =GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
"""
Creates and prints performance metrics for the model

Args:
    model: the model for which the metrics need be printed
    X_test: the features to test the model
    Y_test: the labels linked to X_test
    category_names: names for the features in X_test

"""
    ypred = model.predict(X_test)
    df_results = pd.DataFrame(index = ['f1-score', 'recall','precision'],columns=category_names)
    for i,col in enumerate(df_results.columns):
        for line in classification_report(Y_test[:,i],ypred[:,i]).split('\n'):
            first = True
            thisLine = False
            indexThisLine = 0
            for value in line.split():
                if thisLine:
                    if indexThisLine == 0:
                        df_results.loc['precision',col] = float(value)
                    if indexThisLine == 1:
                        df_results.loc['recall',col] = float(value)
                    if indexThisLine == 2:
                        df_results.loc['f1-score',col] = float(value)
                    indexThisLine +=1
                if first:
                    first = False
                    if value == '1':
                        thisLine = True
    df_results = df_results.fillna(0)
    print(df_results)

def save_model(model, model_filepath):
"""
Saves the trained model in a pickle file

Args:
    model: model to be saved
    model_filepath: Path where the model needs to be stored.

"""
    filename = model_filepath
    joblib.dump(model, filename)


def main():
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