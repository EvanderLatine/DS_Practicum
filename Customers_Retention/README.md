# Customers Retention

## Task
Build a classification model to predict the probability of a customer decreasing
their purchasing activity, so the business can target retention actions and plan
marketing strategies.

## Models and Methods
- Data preparation: merge multiple data sources, define target, train/test split
  with stratification.
- Feature processing: `ColumnTransformer` with `OneHotEncoder` for categorical
  features and numeric scaling; numeric scaler is selected during model search.
- Model selection: `RandomizedSearchCV` with ROC AUC and 5-fold CV across
  `DecisionTreeClassifier`, `KNeighborsClassifier`, `LogisticRegression`, `SVC`.
- Explainability: SHAP values for feature importance.
- Segmentation: KMeans to group customers for strategy design.

## Results and Conclusions
- Baseline best model: KNN with 4 neighbors, Manhattan distance, ROC AUC ~ 0.88
  on the test set.
- The resulting probabilities and segments can be used to prioritize customers
  for retention initiatives.

## Reproducibility
1. Create the environment:
   - `conda env create -f da_practicum_env.yml`
   - `conda activate practicum`
2. Ensure data files exist in `Customers_Retention/source`:
   - `market_file.csv`, `market_money.csv`, `market_time.csv`, `money.csv`
3. Open and run the notebook:
   - `jupyter lab`
   - Run all cells in `Customers_Retention/research.ipynb`.
