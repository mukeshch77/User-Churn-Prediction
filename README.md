# User Churn Prediction

This repository provides an end-to-end solution for predicting user churn based on event data. The solution uses a combination of data preprocessing, feature engineering, and machine learning (specifically Random Forest) to identify users who are likely to churn. The model aims to provide actionable business recommendations based on user behavior.

## Project Overview

The goal of this project is to predict user churn, defined as users who have not interacted with the platform for the last 30 days, based on their event activities (e.g., views, cart additions, purchases). This is achieved by:

1. **Data Preprocessing:** Cleaning and transforming raw event data.
2. **Exploratory Data Analysis (EDA):** Visualizing user behavior and event types.
3. **Feature Engineering:** Creating meaningful features to be used for modeling.
4. **Predictive Modeling:** Using Random Forest to predict churn and evaluate the model.
5. **Model Interpretation:** Using SHAP to explain model decisions and identify important features.
6. **Business Recommendations:** Providing actionable insights based on model results.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP

You can install the required libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

## Project Structure

The directory structure of the project is as follows:

```
.
├── data
│   └── events.csv        # Event data for users (user interactions like views, cart, purchase)
├── src
│   ├── data_preprocessing.py  # Code for data loading and preprocessing
│   ├── feature_engineering.py  # Code for feature engineering
│   ├── modeling.py         # Code for model training, hyperparameter tuning, and evaluation
│   └── interpretability.py  # Code for model interpretation using SHAP
├── README.md              # Project documentation
└── requirements.txt       # Python package dependencies
```

## Step-by-Step Explanation

### 1. Data Loading and Inspection

```python
data = pd.read_csv(data_path, parse_dates=['event_time'])
```

The dataset is loaded from the provided `events.csv` file, with the `event_time` column parsed as a `datetime` object. This allows us to extract features based on dates and times. 

After loading the data, we inspect its first few rows, summarize the data, and check for any missing values:

```python
print(data.head())
print(data.info())
print(data.isnull().sum())
```

### 2. Exploratory Data Analysis (EDA)

#### Event Type Distribution

The first visualization is a count plot of different event types (e.g., `view`, `cart`, `purchase`). This gives an overview of the distribution of events across users.

```python
sns.countplot(data['event_type'], order=data['event_type'].value_counts().index)
```

#### User Activity Analysis

We create aggregated user-level features such as:

- First activity date (`first_activity`)
- Last activity date (`last_activity`)
- Number of unique days the user has interacted with the platform (`unique_days`)
- Total number of events (`total_events`)
- Total money spent by the user (`total_spent`)

```python
user_activity = data.groupby('user_id').agg({
    'event_date': ['min', 'max', 'nunique'],
    'event_type': 'count',
    'price': 'sum'
}).reset_index()
```

We also define churn based on whether a user has been inactive (no events) for the last 30 days.

```python
user_activity['churned'] = user_activity['last_activity'].apply(lambda x: 1 if (last_date - x).days > 30 else 0)
```

### 3. Feature Engineering

We create additional features based on event types, such as the ratio of views, carts, and purchases for each user. This helps the model capture the behavior patterns of users.

```python
user_features = data.groupby('user_id').agg({
    'event_type': lambda x: x.value_counts(normalize=True).to_dict(),
    'price': ['mean', 'max', 'sum'],
    'category_id': lambda x: x.nunique(),
    'product_id': lambda x: x.nunique()
}).reset_index()
```

### 4. Predictive Modeling

The target variable is the `churned` label, while the features are based on user behavior and spending. We split the data into training and test sets, then train a Random Forest classifier on the data. We use a grid search for hyperparameter tuning to find the best-performing model.

```python
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1', verbose=2)
grid_search.fit(X_train, y_train)
```

After training the model, we evaluate its performance using metrics such as the classification report, ROC AUC score, and F1 score.

```python
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
```

### 5. Feature Importance and Interpretability

Using SHAP (SHapley Additive exPlanations), we analyze the model's decisions and visualize the importance of different features.

```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type='bar')
```

This step helps in understanding the factors contributing to churn prediction.

### 6. Business Recommendations

Based on the churn predictions, we provide business recommendations:

1. **Identify high churn risk users**: Users with low purchase ratio and high view ratio.
2. **Personalized Offers**: Target these users with tailored offers to prevent churn.
3. **Enhance User Experience**: Focus on improving the most popular categories and enhancing session engagement.

## Conclusion

By following this approach, the project provides actionable insights to predict user churn. The combination of machine learning and interpretability (through SHAP) allows businesses to act on these predictions to improve user retention.
