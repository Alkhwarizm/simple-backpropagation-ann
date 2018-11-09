import numpy as np

class FeedForwardNN():
    """
    Implementasi algoritma backpropagation sederhana. Kelas ini didesain 
    dengan antarmuka mirip dengan antarmuka kelas MLPRegressor dari sklearn.
    Kelas ini mengimplementasi mini batch gradient descent dengan ukuran 
    mini batch yang bisa diatur.
    """
    
    def __init__(self, hidden_layer_sizes=(5,), 
                 batch_size=1, learning_rate=0.1, 
                 max_iter=100, momentum=0, tol=0.0001):
        """
        Parameter: 
            hidden_layer_sizes - Jumlah neuron pada hidden setiap hidden layer.
                                 Panjang tuple menunjukkan jumlah layer dan elemen
                                 ke-i tuple menunjukkan jumlah neuron pada layer 
                                 ke-i.
                                
            batch_size         - Jumlah data pada setiap mini batch.

            learning_rate      - Parameter learning rate untuk update weight.

            max_iter           - Jumlah epoch maksimal yang akan dilakukan.

            momentum           - Parameter momentum untuk update weight.
        """
        assert(len(hidden_layer_sizes) <= 10), "Max layer size is 10."
        self.layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.momentum = momentum
        self.tol = tol
        self.weights = []
        self.delta_weights = []
        self.epoch = 0
        self.activate = lambda x: 1 / (1 + np.exp(-x))
        self.o_err = lambda o, t: o*(1-o)*(t-o)
        
    def __initiate_network(self, n_input=4):
        """
        Inisialisasi fully connected network dengan n_input input neuron. 
        Inisialisasi weight dan bias. 

        Parameter:
            n_input - Jumlah neuron pada input layer.
        """
        network_without_bias = [n_input, *self.layer_sizes]
        self.neurons_at_layers = [n+1 for n in network_without_bias] + [1] # Add bias and output
        for i in range(len(self.neurons_at_layers) - 1):
            shape = (self.neurons_at_layers[i+1], self.neurons_at_layers[i])
            self.weights.append(np.random.random(shape) - 0.5)
            self.delta_weights.append(np.zeros(shape))
        self.prev_error = 0
        self.convergent = False
            
    def score(self, X, y):
        """
        Mengembalikan error kumulatif dari prediksi pada data X terhadap label y.

        Parameter:
            X - Data untuk diprediksi.

            y - Label untuk perbandingan dengan prediksi yang dihasilkan. 
        """
        cummulative_error = 0
        for data, label in zip(X, y):
            output = self.__feed_forward(data)
            cummulative_error += self.o_err(output, label)
        return cummulative_error
            
    def predict(self, x, discrete=False):
        """
        Mengembalikan nilai prediksi data x. Data x akan dimasukkan ke dalam
        neural network dan menghasilkan nilai prediksi.

        Parameter:
            x        - Data yang akan diprediksi.

            discrete - Jika bernilai true, nilai prediksi akan dibulatkan.
        """
        data = np.array(x)
        predictions = []
        for x in data:
            predictions.append(self.__feed_forward(x))
        if discrete:
            return np.around(predictions)
        else:
            return predictions
        
    def fit(self, X, y):
        """
        Metode untuk melakukan pembelajaran terhadap data X dan label y.

        Parameter:
            X - Data pembelajaran.
            
            y - Label pembelajaran.
        """
        self.__initiate_network(X.shape[1])
        self.batch_size = min(X.shape[0], self.batch_size)
        iterator = 0
        while self.epoch < self.max_iter and not self.convergent:
            iterator = 0
            while iterator < X.shape[0]:
                batch_X = X[iterator:iterator + self.batch_size]
                batch_y = y[iterator:iterator + self.batch_size]
                error = self.__feed_batch(batch_X, batch_y)
                delta_error = abs(self.prev_error - error)
                if delta_error <= self.tol:
                    self.convergent = True
                else:
                    self.prev_error = error
                iterator += self.batch_size
            self.epoch += 1
            
    
    def __feed_batch(self, batch_X, batch_y):
        """
        Metode untuk memasukkan mini batch ke dalam network, melakukan propagasi 
        pada error, dan melakukan update weight. Mengembalikan batch error.

        Parameter:
            batch_x - Data dalam mini batch.

            batch_y - Label untuk data dalam mini batch.
        """
        error = 0
        for x, y in zip(batch_X, batch_y):
            output = self.__feed_forward(x)
            error += self.o_err(output, y)
        self.__backpropagate(error)
        self.__update_weights()
        return error
            
            
    def __feed_forward(self, x):
        """
        Metode untuk memasukkan data ke dalam network. Mengembalikan
        nilai output.

        Parameter:
            x - Data yang akan dimasukkan ke dalam network.
        """
        inputs = np.append(x, 1) # Add bias
        self.outputs = [inputs]
        for layer in self.weights:
            inputs = self.activate(np.dot(layer, inputs))
            self.outputs.append(inputs)
        return inputs[0]
        
        
    def __backpropagate(self, error):
        """
        Metode untuk mempropagasikan error dari output layer hingga hidden layer.

        Parameter:
            error - Error untuk dipropagasikan ke hidden layer.
        """
        propagation = np.array([error])
        self.errors = [propagation]
        for weight, output in zip(reversed(self.weights[1:]), reversed(self.outputs[1:-1])):
            propagation = output*(1-output) * np.dot(weight.transpose(), propagation)
            self.errors.append(propagation)
        self.errors = list(reversed(self.errors))
            
            
    def __update_weights(self):
        """
        Metode untuk melakukan update weight.
        """
        for i in range(len(self.weights)):
            shape = (len(self.errors[i]), 1)
            weight_changes = self.learning_rate * (self.errors[i].reshape(shape) * self.outputs[i])
            self.delta_weights[i] = weight_changes + (self.momentum * self.delta_weights[i])
            self.weights[i] = self.weights[i] + self.delta_weights[i]