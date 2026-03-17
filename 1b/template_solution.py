# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from pathlib import Path

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_features(X: np.ndarray) -> np.ndarray:
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component in a given row of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: matrix of floats: dim = (700,21), transformed input with 21 features
    """
    lin = X

    quad = X**2

    exp = np.exp(X)

    cos = np.cos(X)

    const = np.ones((700, 1), dtype=int)

    X_transformed = np.hstack((lin, quad, exp, cos, const))

    print(X_transformed)
    # TODO: Enter your code here
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit_logistic_regression(X, y):
    """
    This function receives training data points, transforms them, and then fits the logistic regression on this 
    transformed data. Finally, it outputs the weights of the fitted logistic regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of integers \in {0,1}, dim = (700,), input labels

    Returns
    ----------
    weights: array of floats: dim = (21,), optimal parameters of logistic regression
    """
    weights = np.zeros((21,))
    X_transformed = transform_features(X)

    learning_rate = 10e-6                               #2nd submission: 0.05
    loops = 10e8                                        #2nd submission: 500000
    n_rows = X_transformed.shape[0]
    
    for i in range(int(loops)):
        z = X_transformed @ weights
        yhat = 1 / (1 + np.exp(-z))                     #yhat = prediction

        error = yhat - y                
        gradient = (1 / n_rows) * (X_transformed.T @ error)
        
        weights = weights - (learning_rate * gradient)
    loss = np.mean(np.logaddexp(0, z) - y * z)

    # TODO: Enter your code here
    assert weights.shape == (21,)
    print (loss)
    return weights


# Main function. You don't have to change this
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    # Data loading
    data = pd.read_csv(base_dir / "data" / "train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print("Sample data: \n", data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit_logistic_regression(X, y)
    # Save results in the required format
    np.savetxt(base_dir / "results.csv", w, fmt="%.12f")

