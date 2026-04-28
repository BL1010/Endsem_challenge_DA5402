import pandas as pd

def prepare():
    path = "data/ml-100k/u.data"

    df = pd.read_csv(
        path,
        sep="\t",
        names=["userId", "itemId", "rating", "timestamp"]
    )

    df = df[["userId", "itemId", "rating"]]

    df.to_csv("data/ratings.csv", index=False)

    print("Processed dataset saved to data/ratings.csv")

if __name__ == "__main__":
    prepare()