First project — a classic classification problem used to get fluent with Pandas and scikit-learn.

## Goal

Predict whether a passenger survived the Titanic disaster based on features like class, age, sex, fare, and family size. Accuracy wasn't the point here — the point was building a clean, correct workflow: EDA → cleaning → feature engineering → train/test split → model comparison → evaluation.

## Dataset

https://www.kaggle.com/competitions/titanic

## Tasks

- Dropped PassengerId, Name, Ticket, Cabin — low signal or too much missing data for a first pass.
- Handled missing values— dropped the 2 rows missing Embarked, and filled missing Age with the mean.
- Engineered a feature — combined SibSp + Parch into a single FamilySize column.
- Encoded categoricals — one-hot encoded Sex and Embarked (dropped the redundant Sex_female column since it's just the inverse of Sex_male).
- Split the data 80/20 with stratification on the target, so the survival rate is consistent across train and test.
- Trained two models — Logistic Regression and Random Forest — and compared them with accuracy, a confusion matrix, and a full classification report (precision/recall/F1), not just accuracy alone.

## Results

| Model | Accuracy | Precision (survived) | Recall (survived) |
|---|---|---|---|
| Logistic Regression | 80.9% | 0.78 | 0.69 |
| Random Forest | 79.2% | 0.75 | 0.69 |

Logistic Regression slightly outperformed Random Forest — not surprising given the dataset is small and the relationships are fairly simple (a random forest's strength — capturing complex feature interactions — has less to work with here).

## Mistakes Made

The most important lesson wasn't a modeling technique — it was a bug. My first version computed the mean for filling missing Age values before splitting into train/test:

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())  
X_train, X_test, y_train, y_test = train_test_split(...)
```

This is data leakage — the imputation was computed using statistics from data that should have been unseen. The fix is to split first, then compute the mean from the training set only, and apply that same value to fill the test set:

```python
X_train, X_test, y_train, y_test = train_test_split(...)
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_train['Age'].mean()) 
```

The effect on accuracy here was small, but the same mistake with feature scaling or other preprocessing can meaningfully inflate reported results.

## Tools

Python, Pandas, scikit-learn (Logistic Regression, Random Forest, train/test split, metrics)
