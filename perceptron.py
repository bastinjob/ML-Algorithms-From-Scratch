import numpy as np
from metrics import accuracy


class Perceptron:

    def __init__(self):
        self.lr= 0.001
        self.n_iter = 10
        self.weights = None
        self.bias = None
        self.activation = self.unit_step_func


    def fit(self,X,y, n_iters, lr):
        self.n_iters = n_iters
        self.lr = lr
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias=0

        y_ = np.array([1 if i>0 else 0 for i in y])
        for epoch in range(self.n_iters):
            print('Epoch: ', epoch +1)
            for idx, x_i in enumerate(X):
                
                #compute y_pred
                y_pred = self.activation(np.dot(x_i, self.weights) + self.bias)

                #update
                error = y_[idx] - y_pred
                update = self.lr*(error)

                self.weights += update*x_i #weight gets updated by adding the product of lr*error*x_i
                self.bias += update #bias is updated by adding lr*error

    
    def unit_step_func(self, x):
        return np.where(x>0, 1,0)
    
    def predict(self,x):
        return self.activation((np.dot(x, self.weights) + self.bias))
    



if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.random.choice([0, 1], size=100)  # Binary labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train, n_iters=1000, lr=0.01)

    # Make predictions
    predictions = perceptron.predict(X_test)

    # Evaluate the Perceptron
    accuracy = accuracy(predictions,y_test)
    print(f"Perceptron accuracy: {accuracy * 100:.2f}%")