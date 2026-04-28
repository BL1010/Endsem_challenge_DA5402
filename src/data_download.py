import os
import urllib.request
import zipfile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "data"

def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(DATA_URL, zip_path)

    extract_path = os.path.join(DATA_DIR, "ml-100k")

    if not os.path.exists(extract_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

    print("Dataset ready!")

if __name__ == "__main__":
    download_movielens()