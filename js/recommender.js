// js/recommender.js — Item-based Collaborative Filtering

let ratingsData = {};   // { userId: { movieId: rating } }
let itemSim = {};       // { movieId: { movieId: similarity } }
let moviesData = [];    // movies.json 전체

// 데이터 로드
async function loadData() {
  const [ratingsRes, moviesRes] = await Promise.all([
    fetch('data/ratings_processed.json'),
    fetch('data/movies.json')
  ]);
  ratingsData = await ratingsRes.json();
  moviesData  = await moviesRes.json();
  buildItemSim();
}

// 영화 간 코사인 유사도 계산
function cosineSim(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, normA = 0, normB = 0;
  for (const k of keys) {
    const va = a[k] || 0, vb = b[k] || 0;
    dot   += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  return (normA && normB) ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
}

// movieId → { userId: rating } 역인덱스
function buildItemSim() {
  const itemVec = {};
  for (const [uid, movies] of Object.entries(ratingsData)) {
    for (const [mid, r] of Object.entries(movies)) {
      if (!itemVec[mid]) itemVec[mid] = {};
      itemVec[mid][uid] = r;
    }
  }
  itemSim = {};
  const ids = Object.keys(itemVec);
  for (let i = 0; i < ids.length; i++) {
    itemSim[ids[i]] = {};
    for (let j = 0; j < ids.length; j++) {
      if (i === j) continue;
      itemSim[ids[i]][ids[j]] = cosineSim(itemVec[ids[i]], itemVec[ids[j]]);
    }
  }
}

// 추천 실행
// bids: { movieId: coins }  — 경매 결과
function recommend(bids, topN = 15) {
  const seenIds = new Set(Object.keys(bids).map(String));
  const scores  = {};
  const total   = Object.values(bids).reduce((a, b) => a + b, 0) || 1;

  for (const [bid_mid, coins] of Object.entries(bids)) {
    if (!coins) continue;
    const weight = coins / total;           // 투자 비율
    const sims   = itemSim[String(bid_mid)] || {};
    for (const [mid, sim] of Object.entries(sims)) {
      if (seenIds.has(mid)) continue;
      scores[mid] = (scores[mid] || 0) + weight * sim;
    }
  }

  return Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([mid, score]) => {
      const m = moviesData.find(x => String(x.id) === mid);
      return m ? { ...m, score: +score.toFixed(4) } : null;
    })
    .filter(Boolean);
}

// TMDB 포스터 연동
const TMDB_KEY = '89bbb8ace3f13ba901fe796d93a01f16';
const TMDB_BASE = 'https://image.tmdb.org/t/p/w300';
let tmdbLinks = {};

async function loadTmdbLinks() {
  const res = await fetch('data/tmdb_links.json');
  tmdbLinks = await res.json();
}

async function fetchPoster(movieId) {
  const tmdbId = tmdbLinks[String(movieId)];
  if (!tmdbId) return null;
  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/movie/${tmdbId}?api_key=${TMDB_KEY}&language=ko-KR`
    );
    const data = await res.json();
    return data.poster_path ? TMDB_BASE + data.poster_path : null;
  } catch { return null; }
}

async function enrichWithPosters(filmList) {
  await loadTmdbLinks();
  return Promise.all(filmList.map(async f => {
    const url = await fetchPoster(f.id);
    return { ...f, posterUrl: url };
  }));
}