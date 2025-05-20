from data.preprocess import load_data
from models.ncf_keras import build_ncf_model
from utils.metrics import evaluate_model

def run():
    (X_train, y_train), (X_test, y_test), num_users, num_items = load_data()

    model = build_ncf_model(num_users, num_items)

    model.summary()

    model.fit(
        [X_train[:, 0], X_train[:, 1]], y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.1
    )

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run()