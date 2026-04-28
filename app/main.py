from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import logging
import os
import time
from collections import defaultdict, deque

from app.model_loader import load_model, load_metadata
from app.schemas import PredictRequest
from prometheus_client import Counter, Histogram, generate_latest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Static UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Metrics
REQUEST_COUNT = Counter("request_count", "Total API Requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Time spent processing request")

# ✅ NEW: rate limit metric
RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Total number of rate limit violations"
)

# ---------------- RATE LIMIT CONFIG ----------------
RATE_LIMIT = 5          # requests
WINDOW_SECONDS = 60     # per 60 seconds

ip_requests = defaultdict(deque)

def get_client_ip(request: Request):
    # handles reverse proxy case
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def check_rate_limit(ip: str):
    now = time.time()
    q = ip_requests[ip]

    # remove old timestamps
    while q and q[0] < now - WINDOW_SECONDS:
        q.popleft()

    if len(q) >= RATE_LIMIT:
        RATE_LIMIT_HITS.inc()
        logger.warning(f"Rate limit exceeded for IP {ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    q.append(now)

# Load model
model = load_model()
meta = load_metadata()
N_ITEMS = meta["n_items"]

# LOAD MOVIE NAMES FROM u.item 
ITEM_MAP = {}

def load_movie_names():
    path = "app/data/u.item"
    if not os.path.exists(path):
        logger.warning("u.item file not found, fallback to IDs")
        return {}

    movie_map = {}

    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            movie_id = int(parts[0]) - 1
            movie_name = parts[1]

            movie_map[movie_id] = movie_name

    logger.info(f"Loaded {len(movie_map)} movie names")
    return movie_map


ITEM_MAP = load_movie_names()

# UI
@app.get("/")
def home():
    return FileResponse("app/templates/index.html")

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

# Recommendation endpoint
@app.post("/recommend")
def recommend(request: Request, req: PredictRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()

    # ✅ IP-based rate limiting
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    user = req.user
    k = req.k

    logger.info(f"Generating top-{k} recommendations for user {user}")

    user_tensor = torch.tensor([user] * N_ITEMS)
    item_tensor = torch.arange(N_ITEMS)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor)

    topk = torch.topk(scores, k)

    items = topk.indices.tolist()
    scores = topk.values.tolist()

    results = []
    for i, s in zip(items, scores):
        name = ITEM_MAP.get(i, f"Movie {i}")

        results.append({
            "item_id": i,
            "name": name,
            "score": float(s)
        })

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "user": user,
        "recommendations": results
    }