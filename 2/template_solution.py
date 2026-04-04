# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process.kernels import RationalQuadratic

@dataclass
class config:
    DEFAULT_MAX_ITER: int = 10
    RANDOM_STATE: int = 42
    OUTPUT_FOLDER: str = ""              # Leer lah wenns im gliche Ordner sel gspeicheret werde wie's file isch       
    BASE_DIR = Path(__file__).resolve().parent
    IMPUTER_INITIAL_STRATEGY = 'median'



def v_print(message: Any, verbose: bool = False) -> None:
    if verbose:
        print(message)
    return


def load_data(base_dir: Path | None = None, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    base_dir: Path, directory containing the data files
    verbose: bool, whether to print information in this step or not

    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels               price_CHF
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    v_print("\nLoading Data...", verbose)
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    # Load training data
    train_df = pd.read_csv(base_dir / "train.csv")
    
    v_print(f"\nTraining data: {train_df.shape[0]} rows, {train_df.shape[1]} columns", verbose)
    v_print(train_df.head(2), verbose)
    
    # Load test data
    test_df = pd.read_csv(base_dir / "test.csv")

    v_print(f"\nTest data: {test_df.shape[0]} rows, {test_df.shape[1]} columns", verbose)
    v_print(test_df.head(2), verbose)

    return test_df, train_df


def model_impute(df: pd.DataFrame, max_iter: int = config.DEFAULT_MAX_ITER, verbose: bool = False) -> pd.DataFrame:
    """
    Impute missing values column-by-column using linear regression
    based on the other columns in df.
    Assumes all columns in df are numeric.
    """
    cols = df.columns
    idx = df.index

    estimator = BayesianRidge(
        compute_score=True,
        fit_intercept=True,
        verbose=verbose
    )
    
    imputer = IterativeImputer(
        max_iter=max_iter,
        estimator=estimator, 
        random_state=config.RANDOM_STATE,
        verbose=2 if verbose else 0,
        initial_strategy=config.IMPUTER_INITIAL_STRATEGY,
        imputation_order='ascending'
    )
    
    imputed_matrix = imputer.fit_transform(df)
    return pd.DataFrame(imputed_matrix, columns=cols, index=idx)


def process_impute_data_global(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function performs global data processing and imputation.
    It fits one imputer on training features and applies it to test features.

    Parameters
    ----------
    train_df: pd.DataFrame, training data
    test_df: pd.DataFrame, test data
    verbose: bool, whether to print information during processing

    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels               price_CHF
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Get the labels and features. Keep only the labels that are observable during training
    y_train_full = train_df["price_CHF"]
    X_train_df = train_df.drop(columns=["price_CHF"]).copy()
    X_test_df = test_df.copy()

    # One-hot encode season before imputation.
    X_train_df = pd.get_dummies(X_train_df, columns=["season"])
    X_test_df = pd.get_dummies(X_test_df, columns=["season"])
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

    # Use the same imputer for both test and training data but only use training data to fit the imputer.
    estimator = BayesianRidge(
        compute_score=True,
        fit_intercept=True,
        verbose=verbose
    )
    
    imputer = IterativeImputer(
        max_iter=config.DEFAULT_MAX_ITER,
        estimator=estimator, 
        random_state=config.RANDOM_STATE,
        verbose=2 if verbose else 0,
        initial_strategy=config.IMPUTER_INITIAL_STRATEGY,
        imputation_order='ascending'
    )

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_df),
        columns=X_train_df.columns,
        index=X_train_df.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_df),
        columns=X_test_df.columns,
        index=X_test_df.index
    )

    # Keep only observed labels for supervised training.
    labeled_mask = y_train_full.notna()
    X_train = X_train_imputed.loc[labeled_mask].to_numpy(dtype=float)
    y_train = np.asarray(y_train_full.loc[labeled_mask], dtype=float)
    X_test = X_test_imputed.to_numpy(dtype=float)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def process_impute_data_seasonal(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function performs seasonal data processing and imputation. It imputes missing values separately for each season.

    Parameters
    ----------
    train_df: pd.DataFrame, training data
    test_df: pd.DataFrame, test data
    verbose: bool, whether to print information during processing

    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels               price_CHF
    X_test: matrix of floats: dim = (100, ?), test input with features
    """

    def seasonal_model_impute(df: pd.DataFrame, max_iter: int = config.DEFAULT_MAX_ITER) -> pd.DataFrame:
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

            numeric_part = model_impute(numeric_part, max_iter=max_iter)

            numeric_part["season"] = season_values
            parts.append(numeric_part)

        df_imputed = pd.concat(parts).sort_index()
        return df_imputed

    v_print("\nPerforming seasonal model-based imputation...", verbose)
    X_train_imputed = seasonal_model_impute(train_df.drop(columns=["price_CHF"]), max_iter=config.DEFAULT_MAX_ITER)
    X_test_imputed = seasonal_model_impute(test_df, max_iter=config.DEFAULT_MAX_ITER)

    # combine again, drop empty rows in y
    train_combined = X_train_imputed.copy()
    train_combined["price_CHF"] = train_df["price_CHF"]
    train_combined = train_combined.dropna(subset=["price_CHF"])

    # split again
    X_train_df = train_combined.drop(columns=["price_CHF"])
    y_train = np.asarray(train_combined["price_CHF"], dtype=float)
    X_test_df = X_test_imputed

    # encode seasons
    X_train_df = pd.get_dummies(X_train_df, columns=["season"])
    X_test_df = pd.get_dummies(X_test_df, columns=["season"])

    X_train_df, X_test_df = X_train_df.align(X_test_df, join='left', axis=1, fill_value=0)

    # convert to numpy
    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    
    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ConstantKernel
        self._x_train = None
        self._y_train = None
        self.model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(alpha=0.564, length_scale=0.309) + WhiteKernel(noise_level=0.1),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        self.model.fit(X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model == None:
            raise ValueError("Vergässe s Model z fitte vor de prediction")
        y_pred = np.asarray(self.model.predict(X_test), dtype=float)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":

    # Data loading
    print("\nLoading data...")
    test_df, train_df = load_data(config.BASE_DIR, verbose=False)

    # Data Preprocessing and Imputation
    print("\nProcessing and imputing data...")
    X_train, y_train, X_test = process_impute_data_global(train_df, test_df, verbose=False)

    # Data scaling and normalization
    print("\nScaling data...")
    scaler = StandardScaler() #scalethe input and test dataset
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training and inference
    model = Model()
    print("\nFitting the model...")
    model.fit(X_train=X_train, y_train=y_train) 

    # Use this function for inference
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    out_filename = Path(config.BASE_DIR, config.OUTPUT_FOLDER, 'results.csv')
    print(f"Saving results to {out_filename / 'results.csv'}...")
    dt.to_csv(out_filename, index=False)
    print("\nResults file successfully generated!")

