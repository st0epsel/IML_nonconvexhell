# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, RBF, RationalQuadratic, ConstantKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, ARDRegression, ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import KFold, RepeatedKFold, learning_curve, cross_val_score as skl_cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


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


class Standardizer(object):
    def __init__(self, ignore_columns: list[str] = []) -> None:
        super().__init__()
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.ignore_columns = ignore_columns 
        self.target_fitted = False
        self.columns = None

    def fit(self, X_train: pd.DataFrame) -> None:
        self.columns = list(X_train.columns)
        self.scaler_columns = [column for column in self.columns if column not in self.ignore_columns]
        self.scaler.fit(X_train[self.scaler_columns])
        self.target_fitted = True
        return None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.target_fitted:
            raise ValueError("vergässe de standardizer z fitte")
        X_scaled = X.copy()
        X_scaled[self.scaler_columns] = self.scaler.transform(X[self.scaler_columns])
        return X_scaled
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)

    def inverse_transform(self, X_scaled: pd.DataFrame, columns: list[str] = []) -> pd.DataFrame:
        X_inv_scaled = X_scaled.copy()
        if columns:
            X_inv_scaled[columns] = self.scaler.inverse_transform(X_scaled[columns])
        else:
            X_inv_scaled[self.scaler_columns] = self.scaler.inverse_transform(X_scaled[self.scaler_columns])
        return X_inv_scaled
        


class Model(object):
    def __init__(self, model, name: str = "Model") -> None:
        super().__init__()
        self.name = name
        self._x_train = None
        self._y_train = None
        self.model = model

    def fit(
            self,
            X_train: np.ndarray, 
            y_train: np.ndarray
        ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model == None:
            raise ValueError("Vergässe s Model z fitte vor de prediction")
        y_pred = np.asarray(self.model.predict(X_test), dtype=float)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


class config:
    # System
    RANDOM_STATE: int = 42
    OUTPUT_FOLDER_NAME: str = "results"  # Leer lah wenns im gliche Ordner sel gspeicheret werde wie's file isch       
    BASE_DIR = Path(__file__).resolve().parent
    print(f"Base directory: {BASE_DIR}")
    OUTPUT_FOLDER = Path(BASE_DIR, OUTPUT_FOLDER_NAME)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # KFold CV
    N_CV_SPLITS: int = 5
    N_CV_REPEATS: int = 1

    # Imputation
    IMPUTATION_MAX_ITER: int = 20
    IMPUTER_INITIAL_STRATEGY = 'median'

    # GT for sanity check
    GT_85_filename: Path = Path(BASE_DIR, "GT_results_85.csv")
    GT_92_filename: Path = Path(BASE_DIR, "GT_results_92.csv")

    # Final model selection
    ENSEMBLE_TOP_K: int = 4  # Number of top performing models to include in ensemble.
    ENSEMBLE_WEIGHT_SCALING: int = 5  # Higher values will increase the weight difference between better and worse performing models.
    ENSEMBLE_USE_MANUAL_WEIGHTS: bool = True  # If True, the weights specified in ENSEMBLE_MANUAL_WEIGHTS will be used instead of the weights calculated based on model performance.
    ENSEMBLE_MANUAL_WEIGHTS: list[tuple[str, float]] | None = [
        #("RF", 0.0660298842159948),
        #("ExtraTrees", 0.0711377775293724),
        ("Voting_TreeKernel", 0.07565146292024437),
        ("Stacking_TreeKernel", 0.10832804133412499),
        ("RidgeCV", 0.11214108833478026),
        #("ElasticNetCV", 0.0782937774068802),
        #("HGBR_Baseline", 0.06055213592281012),
        #("GP_Mix", 0.10934891877496583),
        #("ARDR", 0.07215046712007794),
        #("GBR_Tuned", 0.05747804803469602),
        #("KRR_RBF_Tuned", 0.046848491567301996),
        #("SVR_RBF_Tuned", 0.08417436951212214),
        #("HGBR_Tuned", 0.057865537326629),
    ]  # If not None, this list of (model_name, weight) tuples will overwrite the weights calculated based on model performance. This can be used to manually adjust the ensemble weights based on domain knowledge or other considerations beyond just the R2 score.

    # Verbosity
    VERBOSE: bool = True
    VER_PREPROCESSING: bool = False
    VER_CV: bool  = True
    VER_MODEL_TRAINING: bool = True
    VER_MODEL_EVALUATION: bool = True
    VER_FINAL_TRAINING: bool = True
    VER_VINAL_PREDICTION: bool = True

    MODELS: list[str] = [
        "RF",
        "ExtraTrees",
        "Voting_TreeKernel",
        "Stacking_TreeKernel",
        "RidgeCV",
        "ElasticNetCV",
        "HGBR_Baseline",
        "GP_Mix",
        "ARDR",
        "GBR_Tuned",
        "KRR_RBF_Tuned",
        "SVR_RBF_Tuned",
        "HGBR_Tuned"
    ]

    MODEL_CANDIDATES: list[Model] = [
        Model(
            name="RF",
            model=RandomForestRegressor(
                n_estimators=900,
                max_depth=None,
                min_samples_leaf=1,
                max_features=0.6,
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        Model(
            name="ExtraTrees",
            model=ExtraTreesRegressor(
                n_estimators=1200,
                max_depth=None,
                min_samples_leaf=1,
                max_features=0.75,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        Model(
            name="Voting_TreeKernel",
            model=VotingRegressor(
                estimators=[
                    ("et", ExtraTreesRegressor(n_estimators=900, min_samples_leaf=1, max_features=0.75, n_jobs=-1, random_state=RANDOM_STATE)),
                    ("rf", RandomForestRegressor(n_estimators=700, min_samples_leaf=1, max_features=0.6, n_jobs=-1, random_state=RANDOM_STATE)),
                    ("hgbr", HistGradientBoostingRegressor(learning_rate=0.03, max_iter=700, max_leaf_nodes=63, min_samples_leaf=8, l2_regularization=0.2, random_state=RANDOM_STATE)),
                    ("svr", SVR(kernel="rbf", C=20.0, epsilon=0.03, gamma=0.08)),
                ],
            ),
        ),
        Model(
            name="Stacking_TreeKernel",
            model=StackingRegressor(
                estimators=[
                    ("et", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=1, max_features=0.75, n_jobs=-1, random_state=RANDOM_STATE)),
                    ("rf", RandomForestRegressor(n_estimators=500, min_samples_leaf=1, max_features=0.6, n_jobs=-1, random_state=RANDOM_STATE)),
                    ("hgbr", HistGradientBoostingRegressor(learning_rate=0.03, max_iter=600, max_leaf_nodes=63, min_samples_leaf=8, l2_regularization=0.2, random_state=RANDOM_STATE)),
                    ("krr", KernelRidge(kernel="rbf", alpha=0.3, gamma=0.08)),
                    ("svr", SVR(kernel="rbf", C=20.0, epsilon=0.03, gamma=0.08)),
                ],
                final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 15)),
                passthrough=True,
                cv=4,
                n_jobs=-1,
            ),
        ),
        Model(
            name="RidgeCV",
            model=RidgeCV(
                alphas=np.logspace(-4, 3, 30),
                cv=5,
            ),
        ),
        Model(
            name="ElasticNetCV",
            model=ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                alphas=np.logspace(-4, 1, 30),
                cv=5,
                max_iter=20000,
                random_state=RANDOM_STATE,
            ),
        ),
        Model(
            name="HGBR_Baseline",
            model=HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=700,
                max_leaf_nodes=63,
                min_samples_leaf=8,
                l2_regularization=0.2,
                random_state=RANDOM_STATE,
            ),
        ),
        Model(
            name="GP_Mix",
            model=GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)),
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=RANDOM_STATE,
            ),
        ),
        Model(
            name="ARDR",
            model=ARDRegression(),
        ),
        Model(
            name="GBR_Tuned",
            model=GradientBoostingRegressor(
                learning_rate=0.025,
                n_estimators=800,
                max_depth=2,
                min_samples_leaf=5,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        ),
        Model(
            name="KRR_RBF_Tuned",
            model=KernelRidge(
                kernel="rbf",
                alpha=0.3,
                gamma=0.08,
            ),
        ),
        Model(
            name="SVR_RBF_Tuned",
            model=SVR(
                kernel="rbf",
                C=25.0,
                epsilon=0.03,
                gamma=0.08,
            ),
        ),
        Model(
            name="HGBR_Tuned",
            model=HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=900,
                max_leaf_nodes=63,
                min_samples_leaf=8,
                l2_regularization=0.2,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=40,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    
def v_print(message: Any, verbose: bool = config.VERBOSE) -> None:
    if verbose:
        print(message)
    return None

# Main function. You don't have to change this
def main():
    """
    Improved Pipeline

    0. Preparation
    
    1: Model Selection with KFold CV:
        1. Split original train_df into train_fold and validation_fold
        2. Fit scaler on training fold only. Transform both train and validation fold with the same scaler.
        3. Fit Imputer on training fold only. Transform both train and validation fold with the same imputer.
        4. Drop missing price_CHF rows form the training fold
        5. Train the model on the training fold and evaluate on the validation fold. 
        6. Score only on validation rows with observed price_CHF values to get a more accurate estimate of model performance.

    2: FinalModel training (Voting Array):
        1. After cross-validation, select the best performing model(s) based on average R2 score across folds.
        2. Train the selected model(s) on the entire training data (with appropriate scaling and imputation) and evaluate on the test data.
        3. Save predictions in the required format.

    Basis:
    - each fold gets its own scaler
    - each fold gets its own imputer
    - each fold gets its own trained model(s)
    """

    ##################################################################
    # 0. Preparation #################################################
    ##################################################################

    # Data loading
    v_print("\nLoading data", config.VER_PREPROCESSING)
    test_df, train_df = load_data(config.BASE_DIR, verbose=False)

    # One-hot encode seasons
    allowed_seasons = {'spring', 'summer', 'autumn', 'winter'}
    for split_name, df in [('train', train_df), ('test', test_df)]:
        invalid_mask = ~df["season"].isin(allowed_seasons)
        if invalid_mask.any():
            invalid_values = sorted({('NaN' if pd.isna(v) else str(v)) for v in df.loc[invalid_mask, "season"].unique()})
            raise ValueError(f"Unexpected season value(s) found in {split_name} data: {invalid_values}")
    test_df = pd.get_dummies(test_df, columns=['season'], prefix='season')
    train_df = pd.get_dummies(train_df, columns=['season'], prefix='season')

    # Prepare Imputer estimator (we use a simple model for imputation to save time, since we have to fit an imputer in each fold of the cross-validation)
    estimator = BayesianRidge(
        compute_score=True,
        fit_intercept=True,
        verbose=False
    )

    if not config.ENSEMBLE_USE_MANUAL_WEIGHTS:

        ##################################################################
        # 1. Model Selection with KFold CV ###############################
        ##################################################################

        # Prepare folds and model candidates for cross-validation
        v_print("\nKfold CV preparation...", config.VER_CV)

        models = [model for model in config.MODEL_CANDIDATES if model.name in config.MODELS]
        model_performance = {model.name: [] for model in models}

        cv_strategy = RepeatedKFold(n_splits=config.N_CV_SPLITS, n_repeats=config.N_CV_REPEATS, random_state=config.RANDOM_STATE)

        for i, (train_index, validation_index) in enumerate(cv_strategy.split(train_df)):

            # Fold Nr. N
            v_print(f"\nProcessing fold {i+1} / {config.N_CV_SPLITS * config.N_CV_REPEATS} ...", config.VER_CV)

            # Split data into train and validation fold
            train_fold = train_df.iloc[train_index].reset_index(drop=True)
            val_fold = train_df.iloc[validation_index].reset_index(drop=True)

            # Remember the indices of rows with empty price_CHF values in validation and testing fold for exclusion of these rows in model training. 
            train_no_CHF_indices = train_fold[train_fold["price_CHF"].isna()].index
            val_no_CHF_indices = val_fold[val_fold["price_CHF"].isna()].index

            # Scaler 
            v_print("   Scaling data...", config.VER_PREPROCESSING)
            scaler = Standardizer(ignore_columns=["season_spring", "season_summer", "season_autumn", "season_winter"])

            train_fold_scaled = scaler.fit_transform(train_fold)
            val_fold_scaled = scaler.transform(val_fold)

            # Split data into X and y for training and validation fold
            X_train_scaled = train_fold_scaled.drop(columns=["price_CHF"])
            y_train_scaled = pd.DataFrame(train_fold_scaled["price_CHF"])
            X_val_scaled = val_fold_scaled.drop(columns=["price_CHF"])
            y_val_scaled = pd.DataFrame(val_fold_scaled["price_CHF"])

            # Imputer
            v_print("   Imputing missing values...", config.VER_PREPROCESSING)
            
            imputer = IterativeImputer(
                max_iter=config.IMPUTATION_MAX_ITER,
                estimator=estimator, 
                random_state=config.RANDOM_STATE,
                verbose=0,
                initial_strategy=config.IMPUTER_INITIAL_STRATEGY,
                imputation_order='ascending'
            )

            X_train_scaled_imputed = pd.DataFrame(imputer.fit_transform(X_train_scaled))
            X_val_scaled_imputed = pd.DataFrame(imputer.transform(X_val_scaled))

            # Drop rows with missing price_CHF values for training and validation fold (these rows will not contribute to model training and evaluation, but we keep them in the fold for consistent imputation and scaling)
            X_train_scaled_imputed = X_train_scaled_imputed.drop(index=train_no_CHF_indices, errors='ignore').reset_index(drop=True)
            X_val_scaled_imputed = X_val_scaled_imputed.drop(index=val_no_CHF_indices, errors='ignore').reset_index(drop=True)

            v_print(f"      Training fold:   {X_train_scaled_imputed.shape[0]} rows after dropping missing price_CHF values", config.VERBOSE)
            v_print(f"      Validation fold: {X_val_scaled_imputed.shape[0]} rows after dropping missing price_CHF values", config.VERBOSE)

            # Train models on fold
            for model in models:
                v_print(f"   Training model {model.name}...", config.VER_MODEL_TRAINING)
                model.fit(X_train_scaled_imputed.drop(columns=["price_CHF"]).values, y_train_scaled.values.ravel())

                # Evaluate model on validation fold
                val_predictions = model.predict(X_val_scaled_imputed.drop(columns=["price_CHF"]).values)

                # Calculate R2 score
                r2 = r2_score(y_val_scaled.values, val_predictions)
                v_print(f"      R2 score on validation fold: {r2:.4f}", config.VER_MODEL_EVALUATION)

                # Store model performance for this fold
                model_performance[model.name].append(r2)

        v_print("\nModel evaluation and selection after cross-validation...", config.VER_MODEL_EVALUATION)

        # Model evaluation and selection after cross-validation
        eval_df = pd.DataFrame()
        eval_df["model"] = list(model_performance.keys())
        eval_df["r2_score"] = np.mean(list(model_performance.values()), axis=1)
        eval_df["rank"] = eval_df["r2_score"].rank(ascending=False)
        eval_df["weight"] = np.exp(config.ENSEMBLE_WEIGHT_SCALING * eval_df["r2_score"]) / np.sum(np.exp(config.ENSEMBLE_WEIGHT_SCALING * eval_df["r2_score"]))
        eval_df["weight"] = np.where(eval_df["rank"] > config.ENSEMBLE_TOP_K, 0.0, eval_df["weight"])
        # Normalize weights again after dropping models that are not in the ensemble
        eval_df["weight"] = eval_df["weight"] / eval_df["weight"].sum()

        voting_estimators = [(model.name, model.model) for model in models if model.name in eval_df[eval_df["weight"] > 0]["model"].values]
        final_model = VotingRegressor(estimators=voting_estimators, weights=eval_df.set_index("model").loc[[name for name, weight in voting_estimators], "weight"].values, n_jobs=-1)
        if config.VER_MODEL_EVALUATION:
            print("\nModel performance summary after cross-validation:")
            print(eval_df.sort_values("r2_score", ascending=False).reset_index(drop=True))
    
    else:
        print("\nUsing manual ensemble weights specified in config.ENSEMBLE_MANUAL_WEIGHTS ...")
        # If manual ensemble weights are used, we skip the model selection step and directly prepare the final ensemble model with the specified weights.
        voting_estimators = [(model.name, model.model) for model in config.MODEL_CANDIDATES if model.name in dict(config.ENSEMBLE_MANUAL_WEIGHTS)]
        final_model = VotingRegressor(estimators=voting_estimators, weights=np.array([weight for name, weight in config.ENSEMBLE_MANUAL_WEIGHTS if name in dict(voting_estimators)]), n_jobs=-1)

    ##################################################################
    # 2. Final Model training (Voting Array) #########################
    ##################################################################
    
    # Prepare final training data
    v_print("Prepearing final training data with scaling and imputation...", config.VER_FINAL_TRAINING)
    train_no_CHF_indices = train_df[train_df["price_CHF"].isna()].index

    # Scaling source and target training and testing data.
    f_scaler_source = Standardizer(ignore_columns=["price_CHF","season_spring", "season_summer", "season_autumn", "season_winter"])
    f_scaler_target = Standardizer(ignore_columns=[col for col in train_df.columns if col != "price_CHF"])

    Xf_train_scaled = f_scaler_source.fit_transform(train_df)
    yf_train_scaled = f_scaler_target.fit_transform(train_df)
    Xf_test_scaled = f_scaler_source.transform(test_df) 

    #print(f" scaled data: \n{Xf_train_scaled.head(2)} \n{yf_train_scaled.head(2)} \n{Xf_test_scaled.head(2)}")  

    # Imputing missin data in training and test data with imputer fitted on training data
    imputer_final = IterativeImputer(
            max_iter=config.IMPUTATION_MAX_ITER,
            estimator=estimator, 
            random_state=config.RANDOM_STATE,
            verbose=0,
            initial_strategy=config.IMPUTER_INITIAL_STRATEGY,
            imputation_order='ascending'
        )
    
    Xf_train_scaled_imputed = pd.DataFrame(imputer_final.fit_transform(Xf_train_scaled.drop(columns=["price_CHF"])), columns=Xf_train_scaled.drop(columns=["price_CHF"]).columns)
    Xf_test_scaled_imputed = pd.DataFrame(imputer_final.transform(Xf_test_scaled), columns=Xf_test_scaled.columns)

    print(f"\n imputed data: \n{Xf_train_scaled_imputed.head(5)} \n{Xf_test_scaled_imputed.head(5)}")
    print(f"\n training data y: \n{yf_train_scaled.head(5)}")

    # Delete rows with missing price_CHF values from training data
    Xf_train_scaled_imputed.drop(index=train_no_CHF_indices, inplace=True)
    yf_train_scaled.drop(index=train_no_CHF_indices, inplace=True)

    # Fit model on entire training data
    v_print("Training final model on entire training data...", config.VER_FINAL_TRAINING)
    final_model.fit(Xf_train_scaled_imputed.values, yf_train_scaled.values.ravel())

    # Predict on test data
    y_pred_scaled = pd.DataFrame(final_model.predict(Xf_test_scaled_imputed.values), columns=["price_CHF"])

    # inverse prediction scaling
    y_pred = f_scaler_target.inverse_transform(y_pred_scaled)

    #print(f"\n test_imputed_with_pred:\n{test_imputed_with_pred.head()}")
    #print(f"\n test_imputed with pred inverted head:\n{test_imputed_with_pred_inverted.head()}")

    # Compare predictions to decent GT
    gt_85 = pd.read_csv(config.GT_85_filename)
    gt_92 = pd.read_csv(config.GT_92_filename)

    gt_85_pred_r2 = r2_score(y_true=np.array(gt_85["price_CHF"]), y_pred=y_pred)
    gt_92_pred_r2 = r2_score(y_true=np.array(gt_92["price_CHF"]), y_pred=y_pred)
    gt_85_92_r2 = r2_score(y_true=np.array(gt_85["price_CHF"]), y_pred=np.array(gt_92["price_CHF"]))
    print(f"\nR2 score of predictions compared to GT_85: {gt_85_pred_r2:.4f}")
    print(f"R2 score of predictions compared to GT_92: {gt_92_pred_r2:.4f}")
    print(f"R2 score between GT_85 and GT_92 (sanity check): {gt_85_92_r2:.4f}")


    # Save results in the required format
    dt = pd.DataFrame(y_pred.values, columns=['price_CHF']) 
    dt.columns = ['price_CHF']
    out_filename = Path(config.OUTPUT_FOLDER, 'results.csv')
    print(f"Saving results to {out_filename} ...")
    dt.to_csv(out_filename, index=False)
    print("\nResults file successfully generated!")

    
if __name__ == "__main__":
    main()