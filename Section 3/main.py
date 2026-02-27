import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.patches as mpatches

# 1. Data Prep & Preprocessing
df = pd.read_csv('C:/Users/OHYP/Desktop/makale denemeleri/KITAP BOLUM/code/Section 3/Titanic-Dataset.csv')

features = ['Pclass', 'Sex', 'Age', 'Fare']
target = 'Survived'

# Impute missing values (Median for Age)
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encode categorical variables (Sex: Female=0, Male=1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

X = df[features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Black-Box Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print(f"Model Test Accuracy: {rf_model.score(X_test, y_test):.3f}")

# --- ANALYSIS 1: Permutation Feature Importance (PFI) ---
result_pfi = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = result_pfi.importances_mean.argsort()

# --- ANALYSIS 2: Partial Dependence Plots (PDP) ---
# Inspecting effect of Age and Fare

# Visualization (1 Row, 3 Columns for better layout)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

# Plot 1: Colorful Boxplot (PFI)
# Create boxplot and store the result to modify artists
bp = ax1.boxplot(
    result_pfi.importances[sorted_idx].T,
    vert=False,
    labels=X.columns[sorted_idx],
    patch_artist=True,
    widths=0.6
)

# Define custom colors for each feature box
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8) # Transparency
    patch.set_edgecolor('grey')

# Customize whiskers and caps
for whisker in bp['whiskers']: whisker.set(color='#7570b3', linewidth=1.5, linestyle="--")
for cap in bp['caps']: cap.set(color='#7570b3', linewidth=1.5)
for median in bp['medians']: median.set(color='red', linewidth=2)

ax1.set_title('Figure 5a: Feature Importance (PFI)', fontsize=14, fontweight='bold', color='#333333')
ax1.set_xlabel('Decrease in Accuracy', fontsize=12)
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# Plot 2: PDP for Age (Purple Theme)
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=['Age'], kind='average', ax=ax2,
    line_kw={'color': '#800080', 'linewidth': 3, 'label': 'Average Effect'}
)
ax2.set_title('Figure 5b: PDP (Age)', fontsize=14, fontweight='bold', color='#800080')
ax2.grid(True, linestyle=':', alpha=0.6, color='#800080')
ax2.set_facecolor('#f9f2fa') # Light purple background

# Plot 3: PDP for Fare (Teal Theme)
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=['Fare'], kind='average', ax=ax3,
    line_kw={'color': '#008080', 'linewidth': 3, 'label': 'Average Effect'}
)
ax3.set_title('Figure 5c: PDP (Fare)', fontsize=14, fontweight='bold', color='#008080')
ax3.grid(True, linestyle=':', alpha=0.6, color='#008080')
ax3.set_facecolor('#f0fafa') # Light teal background

plt.tight_layout()
plt.show()