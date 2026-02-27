import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f

# 1. Data Prep
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'Experience': np.random.uniform(0, 30, n),
    'Education': np.random.choice([12, 16, 18, 21], n)
})

# Target: Salary
df['Salary'] = (5000 + 200 * (df['Experience'] ** 1.5) - 5 * (df['Experience'] ** 2) +
                3000 * df['Education'] + np.random.normal(0, 5000, n))

X = df[['Experience', 'Education']]
y = df['Salary']

# 2. Model (s: spline, f: factor)
gam = LinearGAM(s(0) + f(1)).fit(X, y)

# 3. Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ['Experience Effect', 'Education Effect']

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)

    # Fix: Set valid category for term 1 when plotting term 0 to avoid domain error
    if i == 0: XX[:, 1] = 12

    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    ax.plot(XX[:, i], pdep, label='Effect f(x)', color='blue', lw=2)
    ax.plot(XX[:, i], confi, c='r', ls='--', label='95% CI')

    ax.set_title(titles[i], weight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Contribution')
    ax.grid(True, ls=':', alpha=0.6)

    if i == 0: ax.legend()

plt.tight_layout()
plt.show()