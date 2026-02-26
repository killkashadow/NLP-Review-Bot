import os
import csv
import json
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

save_path = 'data/raw/all_reviews.csv'
target_per_domain = 600
timeout = 15
delay = (2, 4)  # случайная задержка

targets = [
    "www.apple.com", "www.samsung.com", "www.canon.com",
    "www.adobe.com", "www.microsoft.com", "www.sony.com",
    "www.deliveroo.com", "www.starbucks.com", "www.uber.com",
    "www.booking.com", "www.hotels.com", "www.tripadvisor.com"
]

def save_csv(reviews):
    """сохраняет отзывы в csv"""
    exists = os.path.isfile(save_path)
    with open(save_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if not exists:
            writer.writerow(['rating', 'text', 'source'])
        writer.writerows(reviews)

def get_page(url, ua, retries=3):
    """загружает страницу, меняет юзер-агент, при ошибках повторяет"""
    for attempt in range(retries):
        try:
            headers = {'User-Agent': ua.random}
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code == 403:
                print("  доступ запрещён (403), maybe бан")
                return None
            else:
                print(f"  статус {r.status_code}, попытка {attempt+1}/{retries}")
        except Exception as e:
            print(f"  ошибка: {e}, попытка {attempt+1}/{retries}")
        time.sleep(2**attempt)
    return None

def parse_trustpilot(domain, limit):
    """собирает отзывы с trustpilot для одного домена"""
    collected = 0
    page = 1
    ua = UserAgent()
    print(f"\n--> парсим {domain}, цель {limit}")

    while collected < limit:
        url = f"https://www.trustpilot.com/review/{domain}?page={page}"
        resp = get_page(url, ua)
        if not resp:
            break

        try:
            soup = BeautifulSoup(resp.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            if not script:
                print("  нет данных, maybe конец")
                break

            data = json.loads(script.string)
            reviews = data.get('props', {}).get('pageProps', {}).get('reviews', [])
            if not reviews:
                print("  отзывы кончились")
                break

            batch = []
            for rev in reviews:
                text = rev.get('text', '').replace('\n', ' ').strip()
                rating = rev.get('rating')
                if text and rating is not None:
                    batch.append([rating, text, 'trustpilot'])
                    collected += 1

            if batch:
                save_csv(batch)
                print(f"  страница {page}: +{len(batch)} отзывов (всего {collected})")
            else:
                print(f"  страница {page}: пусто")

            page += 1
            time.sleep(random.uniform(*delay))

        except Exception as e:
            print(f"  ошибка при обработке: {e}")
            break
    return collected

def remove_dupes():
    """удаляет дубликаты по тексту"""
    if not os.path.exists(save_path):
        print("файла нет, чистить нечего")
        return
    df = pd.read_csv(save_path, sep=';')
    before = len(df)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.to_csv(save_path, index=False, sep=';')
    print(f"дубликатов удалено: {before - len(df)}")


if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    total = 0
    for site in targets:
        total += parse_trustpilot(site, target_per_domain)
        print(f"пока собрано {total} отзывов")

    remove_dupes()
    print("готово!")
