# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

def load_data():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels               price_CHF
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   

    #X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    #y_train = np.zeros_like(train_df['price_CHF'])
    #X_test = np.zeros_like(test_df)

    y_train_df = train_df["price_CHF"].copy()               #only price_CHF (=target)
    X_train_df = train_df.drop(columns=["price_CHF"])
    
    X_test_df = test_df.copy()

    #to fill missing values, we calculate the mean of a column, depending on the seasons. 
    # this mean is then filled into the NaN values of the data 
    # numeric_only=True is used so that it doesb't calculate a mean over the seasons
    seasonal_means_X = X_train_df.groupby('season').mean(numeric_only=True)
    seasonal_means_y = train_df.groupby('season')['price_CHF'].mean()

    #loop through each season 
    for season in seasonal_means_X.index:
        
        # true/false mask for the specific season
        train_mask = X_train_df['season'] == season
        test_mask = X_test_df['season'] == season

        # fill in the means 
        X_train_df.loc[train_mask] = X_train_df.loc[train_mask].fillna(seasonal_means_X.loc[season])
        X_test_df.loc[test_mask] = X_test_df.loc[test_mask].fillna(seasonal_means_X.loc[season])
        
        y_train_df.loc[train_mask] = y_train_df.loc[train_mask].fillna(seasonal_means_y[season])

    #encode season 
    X_train_df = pd.get_dummies(X_train_df, columns=["season"])  
    X_test_df = pd.get_dummies(X_test_df, columns=["season"])

    #Make sure train and test have the same dummy columns in the same order
    X_train_df, X_test_df = X_train_df.align(X_test_df, join='left', axis=1, fill_value=0)

    # convert to numpy
    X_train = X_train_df.values.astype(float)  
    X_test = X_test_df.values.astype(float)      
    y_train = y_train_df.values.astype(float)
    
  

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
        self.model = GaussianProcessRegressor(kernel=Matern())
        self.model.fit(X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self.model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

