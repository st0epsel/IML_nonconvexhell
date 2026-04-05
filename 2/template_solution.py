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
        max_iter=config.IMPUTATION_MAX_ITER,
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

    def fit_imputer(df: pd.DataFrame, max_iter: int = config.IMPUTATION_MAX_ITER, verbose: bool = False) -> IterativeImputer:
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

        imputer.fit(df)
        return imputer

    def apply_imputer(df: pd.DataFrame, imputer: IterativeImputer) -> pd.DataFrame:
        imputed_matrix = imputer.transform(df)
        return pd.DataFrame(imputed_matrix, columns=df.columns, index=df.index)

    def seasonal_model_impute(
        train_features_df: pd.DataFrame,
        test_features_df: pd.DataFrame,
        max_iter: int = config.IMPUTATION_MAX_ITER
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit model-based imputers on train data per season and use them to
        transform both train and test data for that same season.
        """
        train_df_local = train_features_df.copy()
        test_df_local = test_features_df.copy()

        train_parts = []
        test_parts = []

        seasons_in_train = train_df_local["season"].dropna().unique()

        for season in seasons_in_train:
            train_subset = train_df_local[train_df_local["season"] == season].copy()
            test_subset = test_df_local[test_df_local["season"] == season].copy()

            train_season_values = train_subset["season"]
            test_season_values = test_subset["season"]

            train_numeric = train_subset.drop(columns=["season"])
            test_numeric = test_subset.drop(columns=["season"])

            imputer = fit_imputer(train_numeric, max_iter=max_iter, verbose=verbose)
            train_imputed_numeric = apply_imputer(train_numeric, imputer)

            train_imputed_numeric["season"] = train_season_values
            train_parts.append(train_imputed_numeric)

            if not test_numeric.empty:
                test_imputed_numeric = apply_imputer(test_numeric, imputer)
                test_imputed_numeric["season"] = test_season_values
                test_parts.append(test_imputed_numeric)

        remaining_test = test_df_local.loc[~test_df_local.index.isin(pd.concat(test_parts).index)] if test_parts else test_df_local
        if not remaining_test.empty:
            fallback_train_numeric = train_df_local.drop(columns=["season"])
            fallback_imputer = fit_imputer(fallback_train_numeric, max_iter=max_iter, verbose=verbose)
            remaining_numeric = remaining_test.drop(columns=["season"])
            remaining_imputed = apply_imputer(remaining_numeric, fallback_imputer)
            remaining_imputed["season"] = remaining_test["season"]
            test_parts.append(remaining_imputed)

        train_imputed = pd.concat(train_parts).sort_index()
        test_imputed = pd.concat(test_parts).sort_index() if test_parts else test_df_local.copy()
        return train_imputed, test_imputed

    v_print("\nPerforming seasonal model-based imputation...", verbose)
    X_train_imputed, X_test_imputed = seasonal_model_impute(
        train_features_df=train_df.drop(columns=["price_CHF"]),
        test_features_df=test_df,
        max_iter=config.IMPUTATION_MAX_ITER
    )

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

class Standardizer(object):
    def __init__(self, ignore_columns: list[str] | None = None) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.columns: list[str] = []
        self.ignore_columns = ignore_columns or []
        self.scaler_columns: list[str] = []
        self.target_fitted = False

    def fit(self, X_train: pd.DataFrame) -> None:
        self.columns = list(X_train.columns)
        self.scaler_columns = [column for column in self.columns if column not in self.ignore_columns]
        self.scaler.fit(X_train[self.scaler_columns])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()
        X_scaled[self.scaler_columns] = self.scaler.transform(X[self.scaler_columns])
        return X_scaled

    def fit_target(self, y_train: pd.Series | np.ndarray) -> None:
        y_array = np.asarray(y_train, dtype=float).reshape(-1, 1)
        observed_mask = ~np.isnan(y_array).ravel()
        if not np.any(observed_mask):
            raise ValueError("Cannot fit target Standardizer without observed labels.")
        self.target_scaler.fit(y_array[observed_mask].reshape(-1, 1))
        self.target_fitted = True

    def transform_target(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self.target_fitted:
            raise ValueError("Target Standardizer must be fitted before transforming labels.")
        y_array = np.asarray(y, dtype=float).reshape(-1, 1)
        transformed = y_array.copy()
        observed_mask = ~np.isnan(y_array).ravel()
        if np.any(observed_mask):
            transformed[observed_mask] = self.target_scaler.transform(y_array[observed_mask].reshape(-1, 1))
        return transformed.ravel()

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        if not self.target_fitted:
            raise ValueError("Target Standardizer must be fitted before inverse transforming predictions.")
        y_array = np.asarray(y_scaled, dtype=float).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_array).ravel()

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
            self._create_learning_curve_plot(X_train, y_train, Path(config.OUTPUT_FOLDER, filename))
            print(f"   Learning curve for {self.name} saved to {Path(config.OUTPUT_FOLDER, filename)}")
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model == None:
            raise ValueError("Vergässe s Model z fitte vor de prediction")
        y_pred = np.asarray(self.model.predict(X_test), dtype=float)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred
    
    def _create_learning_curve_plot(self, X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
        """
        Minimal learning curve using CV train/test R2.
        """
        cv = KFold(n_splits=2, shuffle=True, random_state=config.RANDOM_STATE)
        lc_result = learning_curve(
            estimator=self.model,
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
                kernel_descr = candidate.model.get_params().get("kernel")
                print(f"\n   Candidate {candidate.name}: kernel={kernel_descr}")

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

@dataclass
class config:
    # System
    RANDOM_STATE: int = 42
    OUTPUT_FOLDER_NAME: str = "results"  # Leer lah wenns im gliche Ordner sel gspeicheret werde wie's file isch       
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_FOLDER = Path(BASE_DIR, OUTPUT_FOLDER_NAME)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Imputation
    IMPUTATION_MAX_ITER: int = 50
    IMPUTER_INITIAL_STRATEGY = 'median'
    IMPUTER_STRATEGY = 'global' # options 'global' and 'seasonal'

    # Model training
    SHOW_LEARNING_CURVE: bool = True

     

# Main function. You don't have to change this
def main():
    # Data loading
    print("\nLoading data...")
    test_df, train_df = load_data(config.BASE_DIR, verbose=False)

    # Data scaling
    print("\nScaling data...")
    feature_scaler = Standardizer(ignore_columns=["season"])
    feature_scaler.fit(train_df.drop(columns=["price_CHF"]))

    target_scaler = Standardizer()
    target_scaler.fit_target(train_df["price_CHF"])

    train_df_scaled = feature_scaler.transform(train_df.drop(columns=["price_CHF"]))
    train_df_scaled["price_CHF"] = target_scaler.transform_target(train_df["price_CHF"])
    test_df_scaled = feature_scaler.transform(test_df)

    # Data Preprocessing and Imputation
    print("\nProcessing and imputing data...")
    X_train, y_train, X_test = process_impute_data_global(train_df_scaled, test_df_scaled, verbose=False)

    # Model training and inference
    print("\nBayesian model comparison...")
    candidate_models = [
        #Model(name = "M0", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + WhiteKernel(noise_level=0.1))),
        #Model(name = "M1", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1))),
        #Model(name = "M2", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))),
        Model(name = "M3", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RationalQuadratic(alpha=0.564, length_scale=0.309) + WhiteKernel(noise_level=0.1))),
        #Model(name = "M4", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1))),
        #Model(name = "M5", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))),
        Model(name = "M6", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1))),
        #Model(name = "M7", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1))),
        #Model(name = "M8", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1))),
        Model(name = "M9", kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1))),
    ]
    comparator = BayesianModelSelector(candidates=candidate_models, verbose=True)
    comparator.compare_on_train(X_train=X_train, y_train=y_train)
    
    # Predicting on test data using the combination of the most suited models for the test data
    print("\nPredicting on test data...")
    y_pred_scaled = comparator.predict(X_test=X_test)
    y_pred = target_scaler.inverse_transform_target(y_pred_scaled)

    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    out_filename = Path(config.OUTPUT_FOLDER, 'results.csv')
    print(f"Saving results to {out_filename} ...")
    dt.to_csv(out_filename, index=False)
    print("\nResults file successfully generated!")
    

if __name__ == "__main__":
    main()