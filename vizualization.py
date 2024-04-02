import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('xx_projetos/alzheimer/Alzheimer.csv')
df = df.dropna(axis=0).reset_index(drop=True)
df['Group'] = df['Group'].map({'Nondemented': 0, 'Demented': 1, 'Converted':2})
df['M/F'] = df['M/F'].map({'M': 0, 'F': 1})

print(df.head())
correlation_matrix = df.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix (Spearman)')
plt.show()
