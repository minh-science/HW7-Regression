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
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.metrics import accuracy_score, log_loss

def test_prediction():
	# code from main.py
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	# end code from main.py

	# create new training set and weights with known true values
	X_pytest=np.array([[1,1,1],[2,2,2],[1,1,1]])
	W_pytest = np.array([1,2,3])
	prediction_truth = [0.99752738, 0.99999386, 0.99752738]

	# replace training set and weights 
	log_model.W = W_pytest
	prediction_pytest = log_model.make_prediction(X_pytest)

	# assert that prediction is equal to true values 
	for i in range(len(prediction_pytest)):
		assert np.isclose( prediction_pytest[i], prediction_truth[i], 0.0000001)
test_prediction()

def test_loss_function():
	# code from main.py
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	# end code from main.py

	y_true_pytest = np.array([1,2,3])
	y_pred_pytest = np.array([0.1,0.2,0.3])

	loss_truth = 2.7322952972161265

	loss_pytest = log_model.loss_function(y_true= y_true_pytest, y_pred=y_pred_pytest) 
	assert np.isclose(loss_truth, loss_pytest, 0.000001)
test_loss_function()

def test_gradient():
	# code from main.py
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	# end code from main.py

	W_pytest = np.array([1,2,3])
	log_model.W = W_pytest

	X_pytest=np.array([[1,2,3],[4,5,6],[7,8,9]])
	y_true_pytest = np.array([3,2,1])

	gradient_truth = [-2.00000028, -3.00000055, -4.00000083]
	gradient_pytest = log_model.calculate_gradient(X= X_pytest, y_true= y_true_pytest )

	for i in range(len(gradient_truth)):
		assert np.isclose(gradient_truth[i], gradient_pytest[i], 0.000001)

test_gradient()

def test_training():
	pass