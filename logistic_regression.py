import numpy as np
from metrics import accuracy


class LogisticRegression:

    def __init__(self):
        self.lr = None
        self.n_iters = None
        self.weights  = None
        self.bias =0
        self.activation = self.sigmoid

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def fit(self,X,y, lr=0.001, n_iters=100):
        self.lr=lr
        self.n_iters = n_iters

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0


        for epoch in range(self.n_iters):

            print('Epoch: ', epoch+1)


            #compute the linear output
            lin_output = np.dot(X, self.weights) + self.bias
            y_pred = self.activation(lin_output)

            #compute gradient
            #The loss fucntion for logistic egression is binary crossentropy
            # L(W,b) = -(1/m)*Sum(y_i*log(y_pred) + (1-y_i)log(1-y_pred))
            #Gradient for this loss wrt W and b
            
            error = y_pred-y
            dw = (1/n_samples)*np.dot(X.T, error)*self.lr
            db = (1/n_samples)*np.sum(error)*self.lr
            self.weights -= dw
            self.bias -= db

        
    def predict(self,x):
        lin_output = np.dot(x, self.weights) + self.bias
        y_pred = self.activation(lin_output)
        pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return pred_class
        

if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.random.choice([0, 1], size=100)  # Binary labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression Model
    logo_reg = LogisticRegression()
    logo_reg.fit(X_train, y_train, n_iters=1000, lr=0.01)

    # Make predictions
    predictions = logo_reg.predict(X_test)

    # Evaluate the LogisticRegression
    accuracy = accuracy(predictions,y_test)
    print(f"Logistic Regression accuracy: {accuracy * 100:.2f}%")



    