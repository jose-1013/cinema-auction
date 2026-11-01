import json
import numpy as np
import pandas as pd
from math import log2
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 데이터 로드
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
with open('data/movies.json', encoding='utf-8') as f:
    movies_data = json.load(f)

movie_counts = ratings.groupby('movieId')['rating'].count()
popular_ids  = set(movie_counts[movie_counts >= 50].index)

user_ids   = sorted(ratings['userId'].unique())
test_users = set(user_ids[:50])

# 테스트 유저 완전 제외하고 학습
ratings_train = ratings[
    ratings['movieId'].isin(popular_ids) &
    ~ratings['userId'].isin(test_users)
]

train_user_ids  = sorted(ratings_train['userId'].unique())
train_movie_ids = sorted(ratings_train['movieId'].unique())
u2i = {u:i for i,u in enumerate(train_user_ids)}
m2i = {m:i for i,m in enumerate(train_movie_ids)}
i2m = {i:m for m,i in m2i.items()}

rows = [u2i[u] for u in ratings_train['userId']]
cols = [m2i[m] for m in ratings_train['movieId']]
vals = list(ratings_train['rating'])
mat  = csr_matrix((vals, (rows, cols)), shape=(len(train_user_ids), len(train_movie_ids)))

# Item-Item 유사도
item_sim = cosine_similarity(mat.T, dense_output=False)

def recommend_cf(bids, top_n=10):
    seen  = set(int(k) for k in bids)
    total = sum(bids.values()) or 1
    scores = defaultdict(float)
    for mid_str, coins in bids.items():
        mid = int(mid_str)
        if mid not in m2i or not coins: continue
        w   = coins / total
        idx = m2i[mid]
        sims = item_sim[idx].toarray().flatten()
        for j, s in enumerate(sims):
            jmid = i2m[j]
            if jmid in seen: continue
            scores[jmid] += w * s
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]

# 오프라인 평가
ndcg_list, recall_list, hit_list = [], [], []

for uid in test_users:
    # 원본에서 인기 영화 평점만 가져오기
    user_ratings = ratings[
        (ratings['userId'] == uid) &
        (ratings['movieId'].isin(popular_ids))
    ]
    if len(user_ratings) < 10: continue

    sorted_r = user_ratings.sort_values('rating', ascending=False)
    half  = len(sorted_r) // 2
    train = sorted_r.iloc[:half]
    test  = sorted_r.iloc[half:]

    # 별점 → 코인 변환
    total_r = train['rating'].sum()
    bids = {str(int(r['movieId'])): r['rating'] / total_r * 100
            for _, r in train.iterrows()}

    recs    = recommend_cf(bids, top_n=10)
    rec_ids = [r[0] for r in recs]

    # 정답: 별점 3.5 이상
    relevant = set(test[test['rating'] >= 3.5]['movieId'].astype(int))
    if not relevant: continue

    hits = [1 if r in relevant else 0 for r in rec_ids]
    hit_list.append(int(any(hits)))
    recall_list.append(sum(hits) / len(relevant))

    dcg  = sum(h / log2(i+2) for i, h in enumerate(hits))
    idcg = sum(1 / log2(i+2) for i in range(min(len(relevant), 10)))
    ndcg_list.append(dcg / idcg if idcg else 0)

print(f"테스트 유저 수  : {len(ndcg_list)}명")
print(f"HitRate@10     : {np.mean(hit_list):.4f}")
print(f"Recall@10      : {np.mean(recall_list):.4f}")
print(f"NDCG@10        : {np.mean(ndcg_list):.4f}")