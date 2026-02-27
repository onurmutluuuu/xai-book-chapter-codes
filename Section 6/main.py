
import pandas as pd
import numpy as np
import dice_ml
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set seed for reproducibility
np.random.seed(42)

# --- DISPLAY SETTINGS ---
# Set floating point precision to 3 decimal places for academic reporting
pd.options.display.float_format = '{:.3f}'.format

print("\n--- Counterfactual Explanations using DiCE ---")

# 1. Data Preparation
data = load_diabetes()
feature_names = list(data.feature_names)
df = pd.DataFrame(data.data, columns=feature_names)

# Transform the target variable into a binary classification problem (Median-based risk analysis)
df['target'] = (data.target > np.median(data.target)).astype(int)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Black-Box Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# 3. DiCE Data and Model Configuration
# Direct list of feature names is provided to the continuous_features parameter
d_data = dice_ml.Data(dataframe=df, continuous_features=feature_names, outcome_name='target')
m_model = dice_ml.Model(model=rf_model, backend="sklearn")

# 4. Initialize Explainer
# Random method is utilized for non-differentiable models
exp = dice_ml.Dice(d_data, m_model, method="random")

# 5. Counterfactual Instance Generation
# Generate the closest counterfactual for the instance at index 10 to achieve the opposite outcome
query_instance = X_test.iloc[[10]]
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class="opposite")

# 6. Visualization of Results
print("\nOriginal Data and Counterfactual Transition:")
dice_exp.visualize_as_dataframe(show_only_changes=True)

# 7. Data Retrieval for Analysis

res_df = dice_exp.cf_examples_list[0].final_cfs_df
print(res_df.round(3).to_string(index=False))