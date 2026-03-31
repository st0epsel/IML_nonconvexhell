# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

def model_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values column-by-column using linear regression
    based on the other columns in df.
    Assumes all columns in df are numeric.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue

        known = df[df[col].notna()].copy()
        missing = df[df[col].isna()].copy()

        if len(known) == 0 or len(missing) == 0:
            continue

        X_known = known.drop(columns=[col])
        y_known = known[col]
        X_missing = missing.drop(columns=[col])

        # Temporary fallback so the model has complete inputs
        train_means = X_known.mean()
        X_known = X_known.fillna(train_means)
        X_missing = X_missing.fillna(train_means)

        Xk = X_known.values.astype(float)
        yk = y_known.values.astype(float)
        Xm = X_missing.values.astype(float)

        # Add bias term
        Xk = np.hstack([np.ones((Xk.shape[0], 1)), Xk])
        Xm = np.hstack([np.ones((Xm.shape[0], 1)), Xm])

        # Stable linear regression
        w = np.linalg.pinv(Xk) @ yk

        preds = Xm @ w

        df.loc[df[col].isna(), col] = preds

    return df


def iterative_impute(df: pd.DataFrame, n_passes: int = 3) -> pd.DataFrame:
    """
    Repeat model-based imputation multiple times.
    """
    df = df.copy()
    for _ in range(n_passes):
        df = model_impute(df)
    return df


def seasonal_model_impute(df: pd.DataFrame, n_passes: int = 3) -> pd.DataFrame:
    """
    Perform model-based imputation separately within each season.
    Expects a column named 'season'.
    """
    df = df.copy()
    parts = []

    for season in df["season"].dropna().unique():
        subset = df[df["season"] == season].copy()

        season_values = subset["season"]
        numeric_part = subset.drop(columns=["season"])

        numeric_part = iterative_impute(numeric_part, n_passes=n_passes)

        numeric_part["season"] = season_values
        parts.append(numeric_part)

    df_imputed = pd.concat(parts).sort_index()
    return df_imputed


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
    #####################

    # idea: look only at X, not y
    # predict the missing data of the other columns with model imputation
    # combine with y again and drop all rows with missing values in price_CHF 
    # -> that way, we can still use this rows for imputation, but don't need to estimate our y.
    
    # Model imputation based on seasons
    #train & test data
    X_train_imputed = seasonal_model_impute(train_df.drop(columns=["price_CHF"]), n_passes=3)
    X_test_imputed = seasonal_model_impute(test_df, n_passes=3)

    # combine again, drop empty rows in y
    train_combined = X_train_imputed.copy()
    train_combined["price_CHF"] = train_df["price_CHF"]
    train_combined = train_combined.dropna(subset=["price_CHF"])

    # split again
    X_train_df = train_combined.drop(columns=["price_CHF"])
    y_train = train_combined["price_CHF"].values.astype(float)
    X_test_df = X_test_imputed

    # encode seasons
    X_train_df = pd.get_dummies(X_train_df, columns=["season"])
    X_test_df = pd.get_dummies(X_test_df, columns=["season"])

    X_train_df, X_test_df = X_train_df.align(X_test_df, join='left', axis=1, fill_value=0)

    # convert to numpy
    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    
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

