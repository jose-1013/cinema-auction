import pandas as pd
import json
import random

base = 'data/ml-latest-small'

# CSV 읽기
ratings = pd.read_csv(f'{base}/ratings.csv')
movies  = pd.read_csv(f'{base}/movies.csv')
tags    = pd.read_csv(f'{base}/tags.csv')

# 1. movies.json — 영화 목록
movies['genres_list'] = movies['genres'].str.split('|')
movies_json = movies.rename(columns={'movieId':'id','title':'title','genres':'genre'}).to_dict(orient='records')

with open('data/movies.json','w',encoding='utf-8') as f:
    json.dump(movies_json, f, ensure_ascii=False, indent=2)

# 2. ratings_processed.json — 유저×영화 평점 행렬 (Item-CF용)
movie_counts = ratings.groupby('movieId')['rating'].count()
popular_ids  = movie_counts[movie_counts >= 50].index
ratings_filtered = ratings[ratings['movieId'].isin(popular_ids)]

matrix = ratings_filtered.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
matrix_dict = {str(uid): {str(mid): float(v) for mid, v in row.items() if v > 0}
               for uid, row in matrix.iterrows()}

with open('data/ratings_processed.json','w') as f:
    json.dump(matrix_dict, f)

# 3. 온보딩용 후보 풀 — 100편
candidates = movies[movies['movieId'].isin(movie_counts.index)].copy()
candidates['count'] = candidates['movieId'].map(movie_counts)

high = candidates[candidates['count'] >= 150]
mid  = candidates[(candidates['count'] >= 60) & (candidates['count'] < 150)]
low  = candidates[(candidates['count'] >= 25) & (candidates['count'] < 60)]

result   = []
seen_ids = set()

def sample_by_genre(pool, max_per_genre, target):
    genre_bucket = {}
    picked = []
    for _, row in pool.sort_values('count', ascending=False).iterrows():
        if row['movieId'] in seen_ids: continue
        genres = row['genres'].split('|')
        for g in genres:
            if genre_bucket.get(g, 0) < max_per_genre:
                genre_bucket[g] = genre_bucket.get(g, 0) + 1
                picked.append({
                    'id': int(row['movieId']),
                    'title': row['title'],
                    'genre': row['genres'].replace('|', ' · '),
                    'count': int(row['count'])
                })
                seen_ids.add(row['movieId'])
                break
        if len(picked) >= target:
            break
    return picked

result += sample_by_genre(high, 8, 40)
result += sample_by_genre(mid,  6, 70)
result += sample_by_genre(low,  4, 100)

random.shuffle(result)

with open('data/candidates.json','w',encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

n_high = len([x for x in result if x['count'] >= 150])
n_mid  = len([x for x in result if 60 <= x['count'] < 150])
n_low  = len([x for x in result if x['count'] < 60])

print(f"✅ movies.json         — {len(movies_json)}편")
print(f"✅ ratings_processed   — {len(matrix_dict)}명 유저")
print(f"✅ candidates.json     — {len(result)}편 후보 풀 (상{n_high}/중{n_mid}/하{n_low})")