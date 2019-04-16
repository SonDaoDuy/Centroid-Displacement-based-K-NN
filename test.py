import numpy as np
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from CDNN import CDNN

def test_iris(k):
	print("---------------Iris dataset------------------")
	print("Loading data.....")
	data = datasets.load_iris()
	n_classes = len(np.unique(data.target))
	print("Done loading data!\n")
	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
	data_dim = len(X_train[0])
	print("Number of classes: %d" % n_classes)
	print("Data dimension: %d" % data_dim)
	print("Number of training samples: %d" % (len(y_train)))
	print("Number of testing samples: %d\n" % (len(y_test)))
	#predict with cdnn
	predict = []
	start_cdnn = time.time()
	for i in range(len(X_test)):
		predict_item = CDNN(X_train, y_train, X_test[i], k)
		predict.append(predict_item)
	t = time.time() - start_cdnn
	acc = accuracy_score(y_test, predict)
	print("Predict time for CDNN: %.3fs" % (t))
	print("Accuracy for CDNN with k = %d: %.3f\n" % (k, acc))

	#predict with sklearn knn
	for weights in ['uniform', 'distance']:
		knn = neighbors.KNeighborsClassifier(k, weights=weights)
		start_knn = time.time()
		knn.fit(X_train, y_train)
		predict = knn.predict(X_test)
		t = time.time() - start_knn
		acc = accuracy_score(y_test, predict)
		print("Predict time for kNN with %s weights: %.3fs" % (weights, t))
		print("Accuracy for kNN with k = %d and %s weights: %.3f\n" % (k, weights, acc))

	print("-----------------------------------------------\n")

def test_digits(k):
	print("---------------Digits dataset------------------")
	print("Loading data.....")
	data = datasets.load_digits()
	n_classes = len(np.unique(data.target))
	print("Done loading data!\n")
	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
	data_dim = len(X_train[0])
	print("Number of classes: %d" % n_classes)
	print("Data dimension: %d" % data_dim)
	print("Number of training samples: %d" % (len(y_train)))
	print("Number of testing samples: %d\n" % (len(y_test)))
	#predict with cdnn
	predict = []
	start_cdnn = time.time()
	for i in range(len(X_test)):
		predict_item = CDNN(X_train, y_train, X_test[i], k)
		predict.append(predict_item)
	t = time.time() - start_cdnn
	acc = accuracy_score(y_test, predict)
	print("Predict time for CDNN: %.3fs" % (t))
	print("Accuracy for CDNN with k = %d: %.3f\n" % (k, acc))

	#predict with sklearn knn
	for weights in ['uniform', 'distance']:
		knn = neighbors.KNeighborsClassifier(k, weights=weights)
		start_knn = time.time()
		knn.fit(X_train, y_train)
		predict = knn.predict(X_test)
		t = time.time() - start_knn
		acc = accuracy_score(y_test, predict)
		print("Predict time for kNN with %s weights: %.3fs" % (weights, t))
		print("Accuracy for kNN with k = %d and %s weights: %.3f\n" % (k, weights, acc))

	print("-----------------------------------------------\n")

def test_breast_cancer(k):
	print("---------------Breast Cancer dataset------------------")
	print("Loading data.....")
	data = datasets.load_breast_cancer()
	n_classes = len(np.unique(data.target))
	print("Done loading data!\n")
	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
	data_dim = len(X_train[0])
	print("Number of classes: %d" % n_classes)
	print("Data dimension: %d" % data_dim)
	print("Number of training samples: %d" % (len(y_train)))
	print("Number of testing samples: %d\n" % (len(y_test)))
	#predict with cdnn
	predict = []
	start_cdnn = time.time()
	for i in range(len(X_test)):
		predict_item = CDNN(X_train, y_train, X_test[i], k)
		predict.append(predict_item)
	t = time.time() - start_cdnn
	acc = accuracy_score(y_test, predict)
	print("Predict time for CDNN: %.3fs" % (t))
	print("Accuracy for CDNN with k = %d: %.3f\n" % (k, acc))

	#predict with sklearn knn
	for weights in ['uniform', 'distance']:
		knn = neighbors.KNeighborsClassifier(k, weights=weights)
		start_knn = time.time()
		knn.fit(X_train, y_train)
		predict = knn.predict(X_test)
		t = time.time() - start_knn
		acc = accuracy_score(y_test, predict)
		print("Predict time for kNN with %s weights: %.3fs" % (weights, t))
		print("Accuracy for kNN with k = %d and %s weights: %.3f\n" % (k, weights, acc))

	print("-----------------------------------------------\n")


def main():
	'''
	Test CDNN function
	'''
	for i in range(5):
		k = (i+1)*4
		print("Testing with k = %d\n" % (k))
		test_iris(k)
		test_digits(k)
		test_breast_cancer(k)

if __name__ == '__main__':
	main()