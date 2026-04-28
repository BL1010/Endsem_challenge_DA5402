# Movie Recommendation System (Endsem Project)

Name: Priyavrata Tiwari  
Course: DA5402  

## Video Demo
[https://drive.google.com/file/d/15NkXriSFeZrm-H6SmAyCOgiumXyZPPfM/view?usp=sharing]

---

## Overview

This project implements a movie recommendation system using the ML-100K dataset.  
It includes a trained PyTorch model, FastAPI backend, UI, monitoring with Prometheus and Grafana, and rate limiting.

---

## Features

### Recommendation System
- Predicts top-K movies for a given user
- Uses a trained PyTorch model
- Efficient scoring over all items

### Movie Names
- Automatically loaded from:
  app/data/u.item
- No manual JSON required

### UI
- Simple browser interface
- Input: user ID and K
- Output: recommended movie names and scores

### Monitoring
Prometheus metrics:
- request_count
- request_latency_seconds
- rate_limit_hits

### Rate Limiting
- IP-based limiting
- Example: 5 requests per second per IP
- Tracks total limit violations

---

## Docker Setup

### Build and Run

```bash
docker compose up --build
```
### Access Services 
```bash
UI:
http://localhost:8000

FastAPI Docs:
http://localhost:8000/docs

Prometheus:
http://localhost:9090

Grafana:
http://localhost:3000
```

### Grafan Setup 

1. Open:
    ```bash
    http://localhost:3000
    ```
2. Login:
   username: admin
   pasword: admin
3. Add Data Source
   - Type: prometheus
   - URL: http://prometheus:9090

## Project Structure 
```bash
app/
│
├── main.py
├── model_loader.py
├── schemas.py
│
├── data/
│   └── u.item
│
├── templates/
│   └── index.html
│
└── static/
```
