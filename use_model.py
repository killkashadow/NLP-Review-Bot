import joblib
import re

# функция очистки (та же, что и при обучении)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

# загружаем модель
model = joblib.load('models/sentiment_model.pkl')
print('модель загружена, пиши отзыв (или "выход")\n')

while True:
    user_input = input('текст: ')
    if user_input.lower() in ['выход', 'exit', 'quit']:
        break

    cleaned = clean_text(user_input)
    pred = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]

    if pred == 1:
        print('😊 позитив')
    else:
        print('😠 негатив')

    print(f'негатив: {proba[0]:.2f}, позитив: {proba[1]:.2f}\n')
