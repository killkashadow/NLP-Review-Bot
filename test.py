import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/all_reviews.csv', sep=';')
print(df['rating'].value_counts())
df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('распределение оценок')
plt.savefig('ratings_distribution.png')
print("график сохранён как ratings_distribution.png")
