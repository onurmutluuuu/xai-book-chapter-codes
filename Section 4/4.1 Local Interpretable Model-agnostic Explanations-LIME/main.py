import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np.random.seed(42)

# --- 1. LIME for Tabular Data (Breast Cancer) ---
print("\n--- LIME (Tabular) ---")
data = load_breast_cancer()

# Split the data for model training
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
feature_names = data.feature_names
class_names = data.target_names

# Train the model
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)

# Initialize the LIME Explainer
explainer_tabular = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    random_state=42
)

# Select an instance to explain (First sample from the test set)
instance_idx = 0
instance = X_test[instance_idx]
prediction = rf_clf.predict_proba(instance.reshape(1, -1))

print(f"True class for sample {instance_idx}: {class_names[y_test[instance_idx]]}")
print(f"Prediction probabilities: {prediction}")

# Generate the explanation
exp_tabular = explainer_tabular.explain_instance(
    instance,
    rf_clf.predict_proba,
    num_features=5
)

# List contributing features in the console
predicted_class_idx = prediction.argmax()
print(f"\nTop 5 features for class '{class_names[predicted_class_idx]}':")

for feature, weight in exp_tabular.as_list(label=predicted_class_idx):
    print(f"{feature}: {weight:.4f}")

# Organizing the results into a DataFrame
rows = []
for feature, weight in exp_tabular.as_list(label=predicted_class_idx):
    if weight > 0:
        effect = "Positive (Supports Prediction)"
    else:
        effect = "Negative (Contradicts Prediction)"

    rows.append([feature, round(weight, 4), effect])

df_lime = pd.DataFrame(rows, columns=["Feature Condition", "Weight (Impact)", "Decision Impact"])

print("\nExplanation Summary Table:")
print(df_lime)


