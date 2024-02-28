"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
# (you will probably need to import more things here)
from regression import logreg
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
sk_prediction = clf.predict(X[:2, :])
sk_prob_predict = clf.predict_proba(X[:2, :])
print(clf.score(X, y))

regresssor = logreg.LogisticRegressor(num_feats=3)

def test_prediction():
	print( regresssor.make_prediction(X = X ))
	print( sk_prob_predict )

test_prediction()

def test_loss_function():
	pass

def test_gradient():
	pass

def test_training():
	pass