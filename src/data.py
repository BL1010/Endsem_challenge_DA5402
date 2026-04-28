import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    user_map = {u: i for i, u in enumerate(df['userId'].unique())}
    item_map = {i: j for j, i in enumerate(df['itemId'].unique())}

    df['user'] = df['userId'].map(user_map)
    df['item'] = df['itemId'].map(item_map)

    return df, len(user_map), len(item_map)

def split(df):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test