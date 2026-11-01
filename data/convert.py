import pandas as pd
import json
import os

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
# 평가 수 50개 이상인 영화만 사용
movie_counts = ratings.groupby('movieId')['rating'].count()
popular_ids  = movie_counts[movie_counts >= 50].index
ratings_filtered = ratings[ratings['movieId'].isin(popular_ids)]

# pivot → dict of dict
matrix = ratings_filtered.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
matrix_dict = {str(uid): {str(mid): float(v) for mid, v in row.items() if v > 0}
               for uid, row in matrix.iterrows()}

with open('data/ratings_processed.json','w') as f:
    json.dump(matrix_dict, f)

# 3. 온보딩용 후보 풀 생성
candidate_ids = movie_counts[movie_counts >= 100].index  # 100개 이상으로 완화
candidates = movies[movies['movieId'].isin(candidate_ids)].copy()
candidates['count'] = candidates['movieId'].map(movie_counts)
candidates = candidates.sort_values('count', ascending=False)

# 장르별 최대 5편씩, 목표 30편
genre_bucket = {}
result = []
for _, row in candidates.iterrows():
    genres = row['genres'].split('|')
    for g in genres:
        if genre_bucket.get(g, 0) < 5:
            genre_bucket[g] = genre_bucket.get(g, 0) + 1
            result.append({
                'id': int(row['movieId']),
                'title': row['title'],
                'genre': row['genres'].replace('|', ' · '),
                'count': int(row['count'])
            })
            break
    if len(result) >= 30:
        break

with open('data/candidates.json','w',encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ movies.json         — {len(movies_json)}편")
print(f"✅ ratings_processed   — {len(matrix_dict)}명 유저")
print(f"✅ candidates.json     — {len(result)}편 후보 풀")