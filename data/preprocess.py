import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path='ncf-movielens-keras/data/ratings.dat'):
    df = pd.read_csv(path, sep='::', engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user'] = user_encoder.fit_transform(df['user_id'])
    df['item'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    X = df[['user', 'item']].values
    y = df['rating'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test), num_users, num_items