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


