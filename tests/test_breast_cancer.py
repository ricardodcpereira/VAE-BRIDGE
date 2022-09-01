"""
Usage example of the Variational Autoencoder Filter for Bayesian Ridge Imputation method (VAE-BRIDGE)
    with the Breast Cancer Wisconsin dataset. The Mean Texture feature is injected with Missing Not
    At Random values. The simulated missing rate is 40%. The dataset is scaled to the range [0, 1].
    The imputation is evaluated through the Mean Absolute Error.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from vaebridge import ConfigVAE, VAEBRIDGE

if __name__ == '__main__':
    dataset = load_breast_cancer(return_X_y=True)
    dataset = np.concatenate((dataset[0], dataset[1].reshape(-1, 1)), axis=1)
    X_train, X_test = train_test_split(dataset, test_size=0.33)
    X_test_md = X_test.copy()
    missing_feature_idx = 1  # Mean Texture

    x_f = X_test_md[:, missing_feature_idx]
    num_mv_mnar = round(X_test_md.shape[0] * 0.4)  # Missing Rate = 40%
    ordered_idx = np.lexsort((np.random.random(x_f.size), x_f))
    X_test_md[ordered_idx[:num_mv_mnar], missing_feature_idx] = np.nan

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test_md = scaler.transform(X_test_md)

    vae_config = ConfigVAE()
    vae_config.verbose = 0
    vae_config.epochs = 200
    vae_config.neurons = [15]
    vae_config.dropout_rates = [0.2]
    vae_config.latent_dimension = 5
    vae_config.number_features = X_train.shape[1]

    vae_bridge_model = VAEBRIDGE(vae_config, missing_feature_idx=missing_feature_idx, k=0.2)
    print("[VAE-BRIDGE] Training and performing imputation...")
    vae_bridge_model.fit(X_train)
    X_test_imp = vae_bridge_model.transform(X_test_md)

    mae = mean_absolute_error(X_test[ordered_idx[:num_mv_mnar], missing_feature_idx],
                              X_test_imp[ordered_idx[:num_mv_mnar], missing_feature_idx])
    print(f"[VAE-BRIDGE] MAE for the breast cancer wisconsin dataset: {mae:.3f}")
