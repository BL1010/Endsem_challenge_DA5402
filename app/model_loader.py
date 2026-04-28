import torch
import json
from src.model import NeuMF

def load_metadata():
    with open("model_meta.json") as f:
        return json.load(f)

def load_model():
    meta = load_metadata()

    n_users = meta["n_users"]
    n_items = meta["n_items"]

    model = NeuMF(n_users, n_items)

    state_dict = torch.load(
        "model_weights.pt",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("✅ Model loaded successfully")

    return model