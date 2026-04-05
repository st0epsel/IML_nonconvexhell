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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, RBF, RationalQuadratic, ConstantKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import KFold, learning_curve
import matplotlib.pyplot as plt


@dataclass
class config:
    DEFAULT_MAX_ITER: int = 10
    RANDOM_STATE: int = 42
    OUTPUT_FOLDER_NAME: str = "results"  # Leer lah wenns im gliche Ordner sel gspeicheret werde wie's file isch       
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_FOLDER = Path(BASE_DIR, OUTPUT_FOLDER_NAME)
    IMPUTER_INITIAL_STRATEGY = 'median'
    SHOW_LEARNING_CURVE: bool = True

    # Make sure the output folder exists
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    




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


def process_impute_data_global(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function performs global data processing and imputation.
    It fits one imputer on training features and applies it to test features as well.

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
    def __init__(self, kernel, name: str = "Model") -> None:
        super().__init__()
        self.name = name
        self._x_train = None
        self._y_train = None
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=config.RANDOM_STATE,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, show_learning_curve: bool = False) -> None:
        self.model.fit(X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train
        if show_learning_curve:
            filename = f"{self.name}_learning_curve.png"
            self.create_learning_curve_plot(self.model, X_train, y_train, Path(config.OUTPUT_FOLDER, filename))
            print(f"   Learning curve for {self.name} saved to {Path(config.OUTPUT_FOLDER, filename)}")
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model == None:
            raise ValueError("Vergässe s Model z fitte vor de prediction")
        y_pred = np.asarray(self.model.predict(X_test), dtype=float)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred
    
    @staticmethod
    def create_learning_curve_plot(estimator, X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
        """
        Minimal learning curve using CV train/test R2.
        """
        cv = KFold(n_splits=2, shuffle=True, random_state=config.RANDOM_STATE)
        lc_result = learning_curve(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring="r2",
            train_sizes=np.linspace(0.2, 1.0, 100),
            n_jobs=-1,
            shuffle=True,
            random_state=config.RANDOM_STATE,
        )
        train_sizes, train_scores, test_scores = lc_result[:3]

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(7, 4))
        plt.plot(train_sizes, train_mean, marker="", label="Training R2")
        plt.plot(train_sizes, test_mean, marker="", label="Testing R2 (CV)")
        plt.xlabel("Training Set Size")
        plt.ylabel("R2 Score")
        plt.title("Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


class BayesianModelSelector(object):

    def __init__(self, candidates: list[Model], log_priors: np.ndarray | None = None, verbose: bool = False) -> None:
        if candidates == []:
            raise ValueError("No candidate models provided for selection.")
        if log_priors is not None and len(log_priors) != len(candidates):
            raise ValueError("Mismatch of log prior and candidate models dimensions.")
        if log_priors is None:
            log_priors = np.log(np.array([1.0 / len(candidates)] * len(candidates))) # trivial priors
        
        self.candidates = candidates
        self.log_priors = log_priors
        self.verbose = verbose
        self.weights = np.zeros(len(candidates))

    def compare_on_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Performs Bayesan model performance estimation using the
        log marginal likelihood returned by sklearn's GaussianProcessRegressor
        May the best model win!

        Args:
            candidates: list of candidate models
            X_train: training input data
            y_train: training output data
            verbose: whether to print information during model selection

        Returns:
            List of relative model performance scores in the same order as the candidate list
        """
        
        scores_abs = np.zeros(len(self.candidates))

        # Getting log evidence scores for the candidate models
        for index, candidate in enumerate(self.candidates):
            if self.verbose:
                print(f"\n   Candidate {index}: kernel={candidate.model.kernel_}")

            candidate.fit(X_train=X_train, y_train=y_train, show_learning_curve=config.SHOW_LEARNING_CURVE)
            score = float(candidate.model.log_marginal_likelihood_value_)
            scores_abs[index] = score

            if self.verbose:
                print(f"              log marginal likelihood={score:.5f}")


        # Model weight generation
        log_evidence = scores_abs
        log_post = log_evidence + self.log_priors 
        weights = np.exp(log_post - np.max(log_post))
        if self.verbose:
            print(f"\nModel weights (unnormalized): {weights/np.sum(weights)}")
        weights[weights < 1e-4] = 0.0 # Remove very bad models from the list entirely
        self.weights = weights / np.sum(weights)
        return
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        results = np.zeros([X_test.shape[0],len(self.candidates)])
        for index, candidate in enumerate(self.candidates):
            results[:, index] = candidate.predict(X_test)
        y_pred = np.sum(results * self.weights, axis=1)
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
    print("\nBayesian model comparison...")
    candidate_models = [
        Model(name = "M0", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + WhiteKernel(noise_level=0.1))),
        Model(name = "M1", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M2", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))),
        Model(name = "M3", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RationalQuadratic(alpha=0.564, length_scale=0.309) + WhiteKernel(noise_level=0.1))),
        Model(name = "M4", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M5", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))),
        Model(name = "M6", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M7", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M8", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M9", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M10", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RationalQuadratic(alpha=1.0, length_scale=1.0) + WhiteKernel(noise_level=0.1))),
    ]
    comparator = BayesianModelSelector(candidates=candidate_models, verbose=True)
    comparator.compare_on_train(X_train=X_train, y_train=y_train)
    
    # Predicting on test data using the combination of the most suited models for the test data
    print("\nPredicting on test data...")
    y_pred = comparator.predict(X_test=X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    out_filename = Path(config.OUTPUT_FOLDER, 'results.csv')
    print(f"Saving results to {out_filename} ...")
    dt.to_csv(out_filename, index=False)
    print("\nResults file successfully generated!")
    

