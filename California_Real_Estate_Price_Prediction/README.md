# California Housing Price Prediction

Linear regression models for predicting median house values using PySpark MLlib.

## Project Overview

This project builds and compares two linear regression models to predict median housing prices (`median_house_value`) using the California Housing 1990 dataset.

**Models:**
- **Model 1:** All features (numerical + one-hot encoded `ocean_proximity`)
- **Model 2:** Numerical features only

**Evaluation Metrics:** RMSE, MAE, R²

## Dataset

California Housing dataset with 20,640 samples and 10 features:

| Feature | Type | Description |
|---------|------|-------------|
| longitude | double | Geographic coordinate |
| latitude | double | Geographic coordinate |
| housing_median_age | double | Median age of houses in the block |
| total_rooms | double | Total number of rooms in the block |
| total_bedrooms | double | Total number of bedrooms in the block |
| population | double | Population in the block |
| households | double | Number of households in the block |
| median_income | double | Median income of households |
| ocean_proximity | string | Proximity to ocean (categorical) |
| median_house_value | double | **Target variable** |

## Setup

### Prerequisites

- Conda package manager
- Java 17+ (required for PySpark 4.x)

### Installation

```bash
# Create conda environment from the parent directory
conda env create -f ../da_practicum_env.yml

# Activate environment
conda activate practicum

# Verify Java version (must be 17+)
java -version
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook notebook.ipynb

# Or run all cells via command line
jupyter nbconvert --to notebook --execute notebook.ipynb
```

## Solution

### Data Preprocessing

1. **Missing Values:** 207 null values in `total_bedrooms` handled via `Imputer` inside Pipeline (median strategy, computed only on training data to prevent data leakage)
2. **Categorical Encoding:** `ocean_proximity` transformed via StringIndexer + OneHotEncoder
3. **Feature Scaling:** StandardScaler applied to numerical features (z-score normalization)

**Important:** All preprocessing steps are performed AFTER train/test split and are fitted only on training data to avoid data leakage.

### Pipeline Architecture

**Model 1 Pipeline (all features):**
```
Imputer → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → VectorAssembler → LinearRegression
```

**Model 2 Pipeline (numerical only):**
```
Imputer → VectorAssembler → StandardScaler → LinearRegression
```

**Feature vectors:**
- `features_all`: 13 dimensions (8 numerical scaled + 5 OHE categorical)
- `features_numerical_only`: 8 dimensions (numerical scaled only)

### Model Training

- **Algorithm:** Linear Regression (PySpark MLlib) with L2 regularization (regParam=0.01)
- **Train/Test Split:** 80/20 with seed=42 (performed BEFORE preprocessing)
- **Training samples:** 16,560
- **Test samples:** 4,080

## Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Model 1 (all features) | 70,786.68 | 50,863.76 | 0.6378 |
| Model 2 (numerical only) | 71,791.60 | 51,804.75 | 0.6275 |

### Key Findings

1. **Categorical feature impact:** Adding `ocean_proximity` improves RMSE by ~1,005 (1.40% improvement)
2. **R² improvement:** +0.0104 with the categorical feature
3. **Conclusion:** Ocean proximity is a meaningful predictor of housing prices in California

## Future Improvements

1. **Feature Engineering:**
   - Create derived features (rooms_per_household, bedrooms_ratio)
   - Log-transform skewed features
   - Add polynomial features

2. **Hyperparameter Tuning:**
   - Use CrossValidator to optimize `regParam` and `elasticNetParam`
   - Experiment with Lasso (L1) or ElasticNet (L1 + L2) regularization

3. **Alternative Algorithms:**
   - Gradient Boosted Trees (GBTRegressor)
   - Random Forest (RandomForestRegressor)
   - Decision Tree (DecisionTreeRegressor)

## Project Structure

```
California_Real_Estate_Price_Prediction/
├── README.md                 # Project documentation
├── notebook.ipynb            # Main Jupyter notebook
├── source/
│   └── housing.csv           # Dataset
├── CLAUDE.md                 # Claude Code instructions
├── Technical_Task.txt        # Project requirements (Russian)
└── Python_DS_Code_Convention.md  # Code style guide
```

## Tech Stack

- **Framework:** PySpark 4.0.0, MLlib
- **Language:** Python 3.9
- **Runtime:** Java 17 (OpenJDK)
