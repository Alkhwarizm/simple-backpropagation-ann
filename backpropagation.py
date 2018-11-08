import numpy as np

class FeedForwardNN():
    
    def __init__(self, hidden_layer_sizes=(5,), 
                 batch_size=1, learning_rate=0.1, 
                 max_iter=100, momentum=0, tol=0.0001):
        assert(len(hidden_layer_sizes) <= 10), "Max layer size is 10."
        self.layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.momentum = momentum
        self.weights = []
        self.delta_weights = []
        self.epoch = 0
        self.activate = lambda x: 1 / (1 + np.exp(-x))
        self.o_err = lambda o, t: o*(1-o)*(t-o)
        
    def __initiate_network(self, n_input=4):
        network_without_bias = [n_input, *self.layer_sizes]
        self.neurons_at_layers = [n+1 for n in network_without_bias] + [1] # Add bias and output
        for i in range(len(self.neurons_at_layers) - 1):
            shape = (self.neurons_at_layers[i+1], self.neurons_at_layers[i])
            self.weights.append(np.random.random(shape) - 0.5)
            self.delta_weights.append(np.zeros(shape))
        self.convergent = False
            
    def score(self, X, y):
        cummulative_error = 0
        for data, label in zip(X, y):
            output = self.__feed_forward(data)
            cummulative_error += self.o_err(output, label)
        return cummulative_error
            
    def predict(self, x, discrete=False):
        data = np.array(x)
        predictions = []
        for x in data:
            predictions.append(self.__feed_forward(x))
        if discrete:
            return np.around(predictions)
        else:
            return predictions
        
    def fit(self, X, y):
        self.__initiate_network(X.shape[1])
        self.batch_size = min(X.shape[0], self.batch_size)
        iterator = 0
        while self.epoch < self.max_iter and not self.convergent:
            iterator = 0
            while iterator < X.shape[0]:
                batch_X = X[iterator:iterator + self.batch_size]
                batch_y = y[iterator:iterator + self.batch_size]
                self.__feed_batch(batch_X, batch_y)
                iterator += self.batch_size
            self.epoch += 1
            
    
    def __feed_batch(self, batch_X, batch_y):
        error = 0
        for x, y in zip(batch_X, batch_y):
            output = self.__feed_forward(x)
            error += self.o_err(output, y)
        self.__backpropagate(error)
        self.__update_weights()
            
            
    def __feed_forward(self, x):
        assert (len(self.weights) > 0), "Network is not initialized."
        inputs = np.append(x, 1) # Add bias
        self.outputs = [inputs]
        for layer in self.weights:
            inputs = self.activate(np.dot(layer, inputs))
            self.outputs.append(inputs)
        return inputs[0]
        
        
    def __backpropagate(self, error):
        propagation = np.array([error])
        self.errors = [propagation]
        for weight, output in zip(reversed(self.weights[1:]), reversed(self.outputs[1:-1])):
            propagation = output*(1-output) * np.dot(weight.transpose(), propagation)
            self.errors.append(propagation)
        self.errors = list(reversed(self.errors))
            
            
    def __update_weights(self):
        for i in range(len(self.weights)):
            shape = (len(self.errors[i]), 1)
            weight_changes = self.learning_rate * (self.errors[i].reshape(shape) * self.outputs[i])
            self.delta_weights[i] = weight_changes + (self.momentum * self.delta_weights[i])
            self.weights[i] = self.weights[i] + self.delta_weights[i]