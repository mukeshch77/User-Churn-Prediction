import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score
import shap

# Load the Dataset
data_path = './events.csv.gz'  # Update with the actual path

data = pd.read_csv(data_path, parse_dates=['event_time'])

# Data Inspection
print("Dataset Overview:")
print(data.head())
print("\nData Summary:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.countplot(data['event_type'], order=data['event_type'].value_counts().index)
plt.title('Event Type Distribution')
plt.xlabel('Event Type')
plt.ylabel('Count')
plt.show()

# Analyzing user activity
data['event_date'] = data['event_time'].dt.date
user_activity = data.groupby('user_id').agg({
    'event_date': ['min', 'max', 'nunique'],
    'event_type': 'count',
    'price': 'sum'
}).reset_index()
user_activity.columns = ['user_id', 'first_activity', 'last_activity', 'unique_days', 'total_events', 'total_spent']

# Defining Churn
# Define churn as no activity (view/cart/purchase) for the last 30 days
last_date = data['event_time'].max().date()
user_activity['churned'] = user_activity['last_activity'].apply(lambda x: 1 if (last_date - x).days > 30 else 0)

print(user_activity['churned'].value_counts(normalize=True))

# Feature Engineering
user_features = data.groupby('user_id').agg({
    'event_type': lambda x: x.value_counts(normalize=True).to_dict(),
    'price': ['mean', 'max', 'sum'],
    'category_id': lambda x: x.nunique(),
    'product_id': lambda x: x.nunique()
}).reset_index()
user_features.columns = ['user_id', 'event_type_distribution', 'avg_price', 'max_price', 'total_spent', 'unique_categories', 'unique_products']

# Merge features with user activity
final_data = user_activity.merge(user_features, on='user_id', how='left')

# Encode event type distribution as separate features
final_data['view_ratio'] = final_data['event_type_distribution'].apply(lambda x: x.get('view', 0))
final_data['cart_ratio'] = final_data['event_type_distribution'].apply(lambda x: x.get('cart', 0))
final_data['purchase_ratio'] = final_data['event_type_distribution'].apply(lambda x: x.get('purchase', 0))
final_data = final_data.drop(columns=['event_type_distribution'])

# Step 6: Predictive Modeling
X = final_data.drop(columns=['user_id', 'first_activity', 'last_activity', 'churned'])
y = final_data['churned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1', verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

# Feature Importance and Interpretability
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type='bar')

# Business Recommendations
print("Recommendations:")
print("1. Identify high churn risk users based on features like low purchase ratio and high view ratio.")
print("2. Provide personalized offers to these users.")
print("3. Enhance user experience by targeting popular categories and improving session engagement.")