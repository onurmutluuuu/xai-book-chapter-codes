import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load Data
df = pd.read_csv('advertising.csv')

# 2. Prepare Variables
# Features: TV, Radio, Newspaper -> Target: Sales
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# 3. Train Model
model = LinearRegression()
model.fit(X, y)

# 4. Create Coefficient Table (Simple English)
coef_df = pd.DataFrame({
    'Ad Channel': ['TV', 'Radio', 'Newspaper'],
    'Impact (Coefficient)': model.coef_
})

# Print Table for the user
print("Table 1: Impact of Ad Channels on Sales")
print(coef_df.round(3))

# 5. Create Figure 1 (Bar Chart in English)
plt.figure(figsize=(12, 9))
colors = ['skyblue', 'orange', 'lightgray']
bars = plt.bar(coef_df['Ad Channel'], coef_df['Impact (Coefficient)'], color=colors, edgecolor='black')

# Titles and Labels
plt.title('Figure 1: Marginal Effects of Ad Spend on Sales', fontsize=12, fontweight='bold')
plt.ylabel('Sales Increase (per Unit Spend)', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 3), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('Figure_1_Marginal_Effects.png', dpi=300)
plt.show()
