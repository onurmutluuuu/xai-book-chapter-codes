import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
file_path = 'Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# 2. Data Preprocessing
features = ['Sex', 'Age', 'Pclass']
target = 'Survived'

# Handle missing values in Age with median
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encode Sex column to numerical values
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

X = df[['Sex_encoded', 'Age', 'Pclass']]
y = df[target]

feature_names = ['Sex', 'Age', 'Pclass']
class_names = ['Died', 'Survived']

# 3. Train Model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# 4. Visualization (No Title, English Labels)
plt.figure(figsize=(14, 8), dpi=300)
plot_tree(clf, 
          feature_names=feature_names, 
          class_names=class_names, 
          filled=True, 
          rounded=True, 
          fontsize=10)

plt.savefig("decision_tree_english_no_title.png", bbox_inches='tight')
plt.show()

# 5. Table 2: Feature Importance Levels
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance (Gini)': importances
}).sort_values(by='Importance (Gini)', ascending=False)

print("\nTable 2: Feature Importance Levels Determined by Decision Tree")
print(feature_importance_df.to_markdown(index=False))
