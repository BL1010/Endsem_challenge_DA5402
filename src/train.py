import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
import os
import json

from data_download import download_movielens
from prepare_data import prepare

from model import NeuMF
from data import load_data, preprocess, split
from utils import rmse, mae, accuracy

if not os.path.exists("data/ratings.csv"):
    download_movielens()
    prepare()

def make_loader(df, batch_size=256):
    return DataLoader(
        TensorDataset(
            torch.tensor(df['user'].values),
            torch.tensor(df['item'].values),
            torch.tensor(df['rating'].values, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=True
    )

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for u, i, r in loader:
            preds = model(u, i)
            loss = loss_fn(preds, r)

            total_loss += loss.item()

            all_preds.append(preds)
            all_targets.append(r)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader),
        "rmse": rmse(preds, targets),
        "mae": mae(preds, targets),
        "accuracy": accuracy(preds, targets)
    }

def train():
    df = load_data("data/ratings.csv")
    df, n_users, n_items = preprocess(df)

    train_df, val_df, test_df = split(df)

    train_loader = make_loader(train_df)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    model = NeuMF(n_users, n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    mlflow.set_experiment("recsys-neumf")

    with mlflow.start_run():

        for epoch in range(5):
            model.train()

            for u, i, r in train_loader:
                preds = model(u, i)
                loss = loss_fn(preds, r)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_metrics = evaluate(model, train_loader, loss_fn)
            val_metrics = evaluate(model, val_loader, loss_fn)

            print(f"\nEpoch {epoch}")
            print("Train:", train_metrics)
            print("Val:", val_metrics)

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_accuracy": val_metrics["accuracy"],
            }, step=epoch)

        test_metrics = evaluate(model, test_loader, loss_fn)

        print("\nTest:", test_metrics)

        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_accuracy": test_metrics["accuracy"],
        })

        mlflow.pytorch.log_model(model, "model")

    torch.save(model.state_dict(), "model_weights.pt")

    with open("model_meta.json", "w") as f:
        json.dump({
            "n_users": n_users,
            "n_items": n_items
        }, f)

if __name__ == "__main__":
    train()