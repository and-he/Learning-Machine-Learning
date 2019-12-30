import random
import numpy as np

class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
		training_data = list(training_data)
		n = len(training_data)

		if test_data:
			test_data = list(test_data)
			num_of_tests = len(test_data)

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [ training_data[k:k + mini_batch_size]
							for k in range(0, n, mini_batch_size) ]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, learning_rate)
			if test_data:
				print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), num_of_tests));
			else:
				print("Epoch {} complete".format(j))

	def update_mini_batch(self, mini_batch, learning_rate):
		nabla_b = [ np.zeros(b.shape) for b in self.biases ]
		nabla_w = [ np.zeros(w.shape) for w in self.weights ]
		for x, y in mini_batch: #for every training input(x) and desired output(y) in our minibatch, calculate its gradient descent
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [ nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b) ]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
			#nabla_w = [ nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w) ]
		self.weights = [ weight - (learning_rate / len(mini_batch)) * new_weight
						for weight, new_weight in zip(self.weights, nabla_w)]
		self.biases = [ bias - (learning_rate/len(mini_batch)) * new_bias
						for bias, new_bias in zip(self.biases, nabla_b) ]

	def backprop(self, x, y):
		nabla_b = [ np.zeros(b.shape) for b in self.biases ]
		nabla_w = [ np.zeros(w.shape) for w in self.weights ]

		activation = x
		activations = [x]
		z_vectors = []
		for bias, weight in zip(self.biases, self.weights):
			z_vector = np.dot(weight, activation) + bias
			z_vectors.append(z_vector)
			activation = sigmoid(z_vector)
			activations.append(activation)

		#delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_vectors[-1])
		delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(z_vectors[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for last_layer in range(2, self.num_layers):
			z_vector = z_vectors[-last_layer]
			sig_prime = sigmoid_prime(z_vector)
			delta = np.dot(self.weights[-last_layer + 1].transpose(), delta) * sig_prime
			nabla_b[-last_layer] = delta
			nabla_w[-last_layer] = np.dot(delta, activations[-last_layer - 1].transpose())
		return(nabla_b, nabla_w)

	def evaluate(self, test_data):
		test_results = [ (np.argmax(self.feedforward(x)), y) 
						for (x, y) in test_data ]
		print("in evaluate function, size of test_results is: ", test_results.shape)
		return sum( int( x == y ) 
					for (x, y) in test_results )

	def cost_derivative(self, output_activations, y):
		return (output_activations - y)

def sigmoid(z):
	return 1.0 / ( 1.0 + np.exp(-z) )

def sigmoid_prime(z):
	return sigmoid(z) * ( 1 - sigmoid(z) )


