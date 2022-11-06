#!/usr/bin/env python
# coding: utf-8

# In[33]:

import numpy as np
from sklearn import metrics


import argparse
import os
import sys
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

try:
    from sklearn.externals import joblib
except:
    import joblib


def run(arguments):
    test_file = None
    train_file = None
    validation_file = None
    joblib_file = "LR_model.pkl"

    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-e', '--test', help='Test attributes (to predict)')
    group1.add_argument('-n', '--train', help='Train data')
    parser.add_argument('-v', '--validation', help='Validation data')

    args = parser.parse_args(arguments)

    Train = False
    Test = False
    Validation = False

    if args.test != None:
        Test = True

    else:
        if args.train != None:
            print(f"Training data file: {args.train}")
            Train = True

        if args.validation != None:
            print(f"Validation data file: {args.validation}")
            Validation = True

    if Train and Validation:
        file_train = pd.read_csv(args.train, quotechar='"', usecols=[0, 1, 2, 3],
                                 dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        # real review? 1=real review, 0=fake review
        # Category: Type of product
        # Product rating: Rating given by user
        # Review text: What reviewer wrote

        # Create TfIdf vector of review using 5000 words as features
        vectorizer = TfidfVectorizer(max_features=5000)

        print(len(file_train[file_train['real review?'] == 1]))
        print(len(file_train[file_train['real review?'] == 0]))
        # Transform text data to list of strings
        corpora = file_train['text_'].astype(str).values.tolist()
        # Obtain featurizer from data
        vectorizer.fit(corpora)
        # Create feature vector
        X = pd.DataFrame(vectorizer.transform(corpora).toarray())
        X["rating"] = file_train["rating"]
        X['category'] = file_train["category"]
        X = pd.concat([X, pd.get_dummies(X['category'], prefix='category', dummy_na=True)], axis=1).drop(['category'], axis=1)

        print(X)

        print("Words used as features:")
        try:
            print(vectorizer.get_feature_names_out())
        except:
            print(vectorizer.get_feature_names())

        # Saves the words used in training
        with open('vectorizer.pk', 'wb') as fout:
            pickle.dump(vectorizer, fout)

        file_validation = pd.read_csv(args.validation, quotechar='"', usecols=[0, 1, 2, 3],
                                      dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})

        corpora_validate = list(file_validation['text_'].astype(str).values)
        X_validate = pd.DataFrame(vectorizer.transform(corpora_validate).toarray())
        X_validate["rating"] = file_validation["rating"]
        X_validate['category'] = file_validation["category"]
        X_validate = pd.concat([X_validate, pd.get_dummies(X_validate['category'], prefix='category', dummy_na=True)], axis=1).drop(['category'],
                                                                                                         axis=1)

        best_accuracy = 0

        # TODO: The following code is performing regularization incorrectly.
        # Your goal
        # is to fix the code.
        for C in [1]:
            lr = LogisticRegression(penalty="l1", tol=0.001, C=C, fit_intercept=True, solver="saga",
                                    intercept_scaling=1, random_state=42)
            # # You can safely ignore any "ConvergenceWarning" warnings
            lr.fit(X, file_train['real review?'])
            # Get logistic regression predictions

            # rf = RandomForestClassifier(max_depth=20, random_state=42)
            # rf.fit(X, file_train['real review?'])

            # lr = GradientBoostingClassifier(n_estimators=100, learning_rate=.1,
            # max_depth = 1, random_state = 0).89fit(X, file_train['real review?'])

            # lr = MLPClassifier(random_state=42, max_iter=300).fit(X,file_train['real review?'])
            # lr = svm.SVC().fit(X, file_train['real review?'])

            y_hat = lr.predict_proba(X_validate)[:, 1]

            y = file_validation['real review?']
            lr_auc = roc_auc_score(y, y_hat)
            print(lr_auc)

            joblib.dump(lr, joblib_file)

    elif Test:
        # This part will be used to apply your model to the test data
        vectorizer = pickle.load(open('vectorizer.pk', 'rb'))

        # Read test file
        file_test = pd.read_csv(args.test, quotechar='"', usecols=[0, 1, 2, 3],
                                dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        # Transform text into list of strigs
        corpora = file_test['text_'].astype(str).values.tolist()
        # Use the words obtained in training to encode in testing

        X = pd.DataFrame(vectorizer.transform(corpora).toarray())
        X["rating"] = file_test["rating"]
        X['category'] = file_test["category"]
        X = pd.concat([X, pd.get_dummies(X['category'], prefix='category', dummy_na=True)], axis=1).drop(['category'],
                                                                                                         axis=1)

        # Load trained logistic regression model
        lr = joblib.load(joblib_file)

        # Competition evaluation is AUC... what is the correct output for AUC evaluation?
        y_hat = lr.predict_proba(X)[:, 1]

        y_pred = (y_hat > 0.5) + 0  # + 0 makes it an integer

        print(f"ID,real review?")
        final = pd.DataFrame()
        for i, y in enumerate(y_hat):
            final = final.append({'ID': int(i), 'real review?': y}, ignore_index=True)
            print(f"{int(i)},{y}")
        final["product number"] = final["product number"].astype(int)
        print(final)
        final.to_csv("Loan_ToSubmit.csv", index=False)


    else:
        print("Training requires both training and validation data files. Test just requires test attributes.")


if __name__ == "__main__":
    run(sys.argv[1:])

