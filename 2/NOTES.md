# Ex2 Notes

## Possible Improvements
### Imputation
- use feature scaled estimator inside imputer
- Compare against simpler baselines: Median/KNN imputation can sometimes outperform iterative when sample size is modest and missing pattern is not random.

### Model Type 
Add stronger tabular baselines:
- HistGradientBosstingRegressor
- CatBoostRegressor (very strong with missing + categorical)
- LightGBM/XGBoost

Ensemble by out-of-fold performance:
- Fit ridge blender after out-of-fold predictions

### Model training improvements
- repeated k-fold for stability, not a single split.
- optimizer restarts should be higher than 5


##
Query:

How would i compare how well models of different kinds are doing? One solution I see would be to split the training dataset into a train and validate subset and keep the subset for validation. Once comparison is done, we could then use the insight gained on the ideal pipeline on the whole training set.

Answer:

Split the labeled training data into train and validation.
Fit each candidate pipeline only on the train part.
Score each model on the validation part using one metric, usually R², RMSE, or MAE.
Pick the best pipeline.
Retrain that chosen pipeline on the full training set.
Predict on the final test set.


