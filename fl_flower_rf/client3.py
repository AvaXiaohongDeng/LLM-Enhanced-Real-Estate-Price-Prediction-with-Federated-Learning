# client3.py
import client_utils
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter("ignore")



# Create the flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"Client {client_id} received the parameters.")
        return client_utils.get_params(model)

    def fit(self, parameters, config):
        print("Parameters before setting: ", parameters)
        client_utils.set_params(model, parameters)
        print("Parameters after setting: ", model.get_params())

        model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}.")

        trained_params = client_utils.get_params(model)
        print("Trained Parameters: ", trained_params)

        return trained_params, len(X_train), {}

    def evaluate(self, parameters, config):
        client_utils.set_params(model, parameters)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        line = "-" * 40
        print(line)
        print(f"RMSE : {rmse:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(line)

        # Flower expects: loss, num_examples, metrics_dict
        return mse, len(X_test), {"RMSE": rmse, "MSE": mse, "MAE": mae}

if __name__ == "__main__":
    client_id = 3
    print(f"Client {client_id}:\n")

    # This client uses data1.csv
    X_train, y_train, X_test, y_test = client_utils.load_dataset("data3.csv")

    print("y_train mean:", float(np.mean(y_train)))
    print("y_test mean :", float(np.mean(y_test)), "\n")

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features = 0.8,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
