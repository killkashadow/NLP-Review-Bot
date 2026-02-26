import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib

# читаем данные
df = pd.read_csv('data/raw/all_reviews.csv', sep=';')
print('всего отзывов:', len(df))
print('оценки:\n', df['rating'].value_counts().sort_index())

# оставляем только 1,2 и 4,5 (тройки выкидываем)
df = df[df['rating'].isin([1,2,4,5])]
# позитив = 1 если оценка 4 или 5, иначе 0
df['sentiment'] = (df['rating'] >= 4).astype(int)
print('\nпосле фильтрации:\n', df['sentiment'].value_counts())

# чистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.strip() != '']
print('осталось после чистки:', len(df))

X = df['clean_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nтренировочных: {len(X_train)}, тестовых: {len(X_test)}')

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('\n' + '-'*40)
print('отчёт по классам')
print('-'*40)
print(classification_report(y_test, y_pred, target_names=['негатив', 'позитив']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['негатив', 'позитив'], yticklabels=['негатив', 'позитив'])
plt.ylabel('реальный')
plt.xlabel('предсказанный')
plt.title('матрица ошибок')
plt.show()

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/sentiment_model.pkl')
print('модель сохранена в models/sentiment_model.pkl')
