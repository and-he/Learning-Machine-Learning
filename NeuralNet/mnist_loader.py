import pickle
import gzip

import numpy as np

def load_data():
	file = gzip.open('mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = pickle.load(file, encoding = "latin1")
	file.close()
	return (training_data, validation_data, test_data)

def load_data_wrapper():
	train_d, val_d, test_d = load_data()
	training_inputs = [ np.reshape(x, (784, 1)) for x in train_d[0] ]
	training_results = [ vectorized_result(y) for y in train_d[1] ]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [ np.reshape(x, (784, 1)) for x in val_d[0] ]
	validation_data = zip(validation_inputs, val_d[1])
	test_inputs = [ np.reshape(x, (784, 1)) for x in test_d[0] ]
	test_data = zip(test_inputs, test_d[1])
	return (training_data, validation_data, test_data)

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e