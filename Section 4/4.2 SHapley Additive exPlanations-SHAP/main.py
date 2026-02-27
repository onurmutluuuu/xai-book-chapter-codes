# Required library: pip install shap
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# 1. Prepare Data and Model (Adapted to Breast Cancer dataset)
data = load_breast_cancer()
X_train, y_train = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)

# 2. Initialize SHAP Explainer and calculate SHAP values
explainer_shap = shap.TreeExplainer(rf_clf)
shap_values = explainer_shap.shap_values(X_train)
# shap_values structure for multi-output: list of arrays, or a 3D array (n_samples, n_features, n_outputs)

# Focus on Class 0 (Malignant)
# For class 0, we retrieve SHAP values for all samples associated with this specific class
shap_values_class_0_all_samples = shap_values[:, :, 0] 
X_train_df = pd.DataFrame(X_train, columns=feature_names)

# Expected value (base value) for class 0
expected_val_class_0 = explainer_shap.expected_value[0]

# Method 1: Summary Plot (Global Explanation)
print("\n--- Displaying SHAP Summary Plot ---")
plt.figure(figsize=(10, 6))
# Use shap_values_class_0_all_samples to represent feature importance across all instances for class 0
shap.summary_plot(shap_values_class_0_all_samples, X_train_df, plot_type="dot", show=False)
plt.title(f"SHAP Summary Plot (For Class: {class_names[0]})")
plt.show()

# Method 2: Waterfall Plot (Local Explanation)
print("--- Displaying SHAP Waterfall Plot ---")
instance_idx = 0
plt.figure(figsize=(10, 6))

# For waterfall plot, extract SHAP values for a single instance for class 0
shap_values_instance_class_0 = shap_values[instance_idx, :, 0]

shap.waterfall_plot(shap.Explanation(
    values=shap_values_instance_class_0, # Values for a single instance, single class
    base_values=expected_val_class_0,    # Expected value for the single class
    data=X_train_df.iloc[instance_idx],
    feature_names=feature_names
), show=False)
plt.title(f"Waterfall Plot for Instance {instance_idx} (Class: {class_names[0]})")
plt.show()

# Method 3: Force Plot (Local Explanation - Interactive)
print("--- Displaying SHAP Force Plot ---")
force_plot_html = shap.force_plot(
    expected_val_class_0,            # Expected value for the single class
    shap_values_instance_class_0,    # Values for a single instance, single class
    X_train_df.iloc[instance_idx],
    matplotlib=False,
    show=False
)
shap.save_html("force_plot_instance_0.html", force_plot_html)
print("Note: 'force_plot_instance_0.html' file has been generated.")