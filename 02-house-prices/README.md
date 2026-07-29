Second project — first regression task, first time using feature scaling and regularization.

## Goal

Predict house sale prices from property characteristics. The focus was on new concepts over Project 1: continuous targets need different metrics than classification, feature scale matters a lot more, and regularization (Ridge) is a new tool to understand.

## Dataset

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Tasks

- Selected a subset of features rather than using all ~80 columns: MSZoning, LotArea, Neighborhood, YearBuilt, OverallQual, OverallCond, SaleType, SaleCondition.
- Encoded categoricals — one-hot encoded MSZoning, Neighborhood, SaleType, and SaleCondition.
- Split the data 80/20 (no stratification needed here — that's a classification concept, not relevant for a continuous target).
- Scaled features with StandardScaler — fit only on the training set, then applied that same scaler to transform the test set. Applied the leakage lesson from Project 1 correctly from the start this time.
- Trained two models — plain Linear Regression and Ridge (L2-regularized) — and compared them across alpha values of 1, 2, and 10.
- Evaluated with RMSE, MAE, and R², plus a predicted-vs-actual scatterplot to visually check for patterns in the errors.

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 43,480 | 27,831 | 0.7534 |
| Ridge (α = 1) | 43,491 | 27,831 | 0.7534 |
| Ridge (α = 2) | 43,501 | 27,830 | 0.7533 |
| Ridge (α = 10) | 43,560 | 27,867 | 0.7526 |

The model explains about 75% of the variance in sale price using just 8 features — a solid result given how much of the full dataset (~80 columns) wasn't used.

## Key Learning

Regularization strength depends on how much there is to regularize. Ridge barely moved the results here, even across a 10x change in alpha (1 to 10). Ridge's L2 penalty shrinks coefficients to reduce overfitting, and with only 8 curated, low-collinearity features, the plain Linear Regression model wasn't overfitting much to begin with, so there was little for Ridge to correct. I expect alpha to matter much more with a larger, messier, more collinear feature set (e.g. using all ~80 raw columns instead of a hand-picked subset).

## Tools

Python, Pandas, scikit-learn (Linear Regression, Ridge, StandardScaler, train/test split, metrics), Matplotlib
