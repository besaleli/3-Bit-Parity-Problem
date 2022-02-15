import numpy as np
from tqdm.auto import tqdm
from linalg import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

def MSE(actual, pred):
    diff = pred - actual
    sq = diff ** 2
    return sq.mean()

class FNN:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, random_state=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.hidden_weights = None
        self.hidden_bias = None
        
        self.output_weights = None
        self.output_bias = None
        
        self.activation = activation
        
        self.random_state = random_state
        
    def __str__(self):
        cats = ['Dimensions', 'Hidden Layer Weights', 'Hidden Layer Bias', 
                'Output Weights', 'Output Bias', 'Activation Function', 'Random State']
        
        dims = '-'.join([str(i) for i in [self.input_dim, self.hidden_dim, self.output_dim]])
        
        vals = [dims, self.hidden_weights, self.hidden_bias, 
                self.output_weights, self.output_bias, self.activation.__name__, self.random_state]
        
        return tabulate(zip(cats, vals))
        
    def initialize_weights(self):
        # initialize random state
        rstate = np.random.RandomState(self.random_state)
        
        self.hidden_weights = rstate.uniform(size=(self.input_dim, self.hidden_dim))
        self.hidden_bias = rstate.uniform(size=(1, self.hidden_dim))
        
        self.output_weights = rstate.uniform(size=(self.hidden_dim, self.output_dim))
        self.output_bias = rstate.uniform(size=(1, self.output_dim))
    
    def perceptron_activation(self, X, layer):
        if layer == 'hidden':
            return self.activation(linear_transform(X, self.hidden_weights, self.hidden_bias))
        elif layer == 'output':
            return self.activation(linear_transform(X, self.output_weights, self.output_bias))
        
    def forward(self, X):
        hidden_layer_output = self.perceptron_activation(X, 'hidden')
        
        output = self.perceptron_activation(hidden_layer_output, 'output')
        
        return hidden_layer_output, output
        
    def backprop(self, X, Y, y_pred, hidden_layer_output, lr):
        error = Y - y_pred
        
        d_predicted_output = error * sigmoid_deriv(y_pred)
        
        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        
        d_hidden_layer = error_hidden_layer * sigmoid_deriv(hidden_layer_output)
        
        # update weights and biases
        self.output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        self.output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
        self.hidden_weights += X.T.dot(d_hidden_layer) * lr
        self.hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
        
    def fit(self, X, Y, lr=0.1, n_epochs=1, return_hidden_outputs=True, print_initial_weights=True):
        self.initialize_weights()
        
        if print_initial_weights:
            print('Initial Settings')
            print(self)
        
        hidden_outputs = {}
        loss = []
        print('\ntraining...')
        for epoch in tqdm(range(n_epochs), total=n_epochs, unit='epoch'):
            hidden_layer_output, y_pred = self.forward(X)
            hidden_outputs[epoch] = hidden_layer_output
            self.backprop(X, Y, y_pred, hidden_layer_output, lr)
            loss.append((epoch, MSE(Y, y_pred)))
            
        if return_hidden_outputs:
            return hidden_outputs, loss
        else:
            return loss
        
    def predict(self, X):
        return self.forward(X)[1]
    
    def evaluate(self, X_test, Y_test):
        tolist = lambda i: [int(np.round(x)) for x in i]
        
        metrics = [('accuracy', accuracy_score), ('precision', precision_score), 
               ('recall', recall_score), ('f1_score', f1_score)]
    
        _, y_pred = self.forward(X_test)

        scores = [(name, np.round(fun(tolist(Y_test), tolist(y_pred)), 3)) for name, fun in metrics]

        print(tabulate(scores, headers=['Metric', 'Score']))
        