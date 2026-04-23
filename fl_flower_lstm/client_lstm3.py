# client_lstm3.py
import flwr as fl
import numpy as np
from tensorflow import keras
import client_utils_lstm as cu
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter("ignore")


class LSTMClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"Client {client_id} get_parameters.")
        return cu.get_model_parameters(model)

    def fit(self, parameters, config):
        cu.set_model_parameters(model, parameters)

        epochs = int(config.get("local_epochs", 1))
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
        )

        loss = float(history.history["loss"][-1])
        mae = float(history.history["mae"][-1])
        print(f"Client {client_id} train: loss={loss:.4f}, mae={mae:.4f}")

        return cu.get_model_parameters(model), len(X_train), {"loss": loss, "MAE": mae}

    def evaluate(self, parameters, config):
        cu.set_model_parameters(model, parameters)

        # Predict
        y_pred = model.predict(X_test, verbose=0).reshape(-1)

        # Metrics (same as RF project)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        line = "-" * 40
        print(line)
        print(f"LSTM Client {client_id} evaluation")
        print(f"RMSE : {rmse:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(line)
        
        return float(mse), len(X_test), {"RMSE": float(rmse), "MSE": float(mse), "MAE": float(mae)}

if __name__ == "__main__":
    client_id = 3
    print(f"LSTM Client {client_id}:\n")

    # per-client dataset
    X_train, y_train, X_test, y_test = cu.load_series_dataset("data3.csv", seq_len=10)

    print("Train samples:", len(X_train))
    print("Test  samples:", len(X_test))
    print("y_train mean:", float(np.mean(y_train)))
    print("y_test mean :", float(np.mean(y_test)), "\n")

    input_shape = X_train.shape[1:]  # (seq_len, n_features)
    model: keras.Model = cu.create_lstm_model(input_shape)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=LSTMClient(),
    )
