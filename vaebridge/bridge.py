import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge
from . import ConfigVAE, VariationalAutoEncoder


class VAEBRIDGE(BaseEstimator):
    """
    Implementation of the Variational Autoencoder Filter for Bayesian Ridge Imputation method (VAE-BRIDGE),
    according to the scikit-learn architecture: methods ``fit()`` and ``transform()``.

    Attributes:
        _config_vae (ConfigVAE): Data class with the configuration for the Variational Autoencoder architecture.
        _missing_feature_idx (int): Index of the feature containing missing values.
        _k (float): Percentage of instances to be retained after the filtering process. Value of ``_k`` ∈ [0, 1].
        _regression_model: Regression model used for the imputation. It must follow the scikit-learn architecture.
        _fitted (bool): Boolean flag used to indicate if the ``fit()`` method was already invoked.
        _binary_features: List of features' indexes that are binary.
        _vae_model (VariationalAutoEncoder): Variational Autoencoder model used for filtering purposes.
        _encoded_data_train: Encoded representation of the complete instances, obtained during the fitting process.
        _X_train_val: Original complete instances used by the fitting process.
    """
    def __init__(self, config_vae: ConfigVAE, missing_feature_idx: int, k: float = 0.2, regression_model=None):
        self._config_vae = config_vae
        self._missing_feature_idx = missing_feature_idx
        self._k = k
        self._regression_model = BayesianRidge() if regression_model is None else regression_model
        self._fitted = False
        self._binary_features = []

        self._config_vae.number_features -= 1  # The feature containing missing values is ignored by the VAE.
        self._vae_model = VariationalAutoEncoder(self._config_vae)

        self._encoded_data_train = None
        self._X_train_val = None

    @staticmethod
    def _get_k_nearest_neighbors(x1, x2, k):
        """
        Obtains the ``k`` nearest neighbors of ``x2`` in ``x1``.

        Args:
            x1: List of possible neighbors. Each neighbor is represented by a mean,
                a standard deviation and a sample from the respective Gaussian distribution.
            x2: Data point for which the neighbors are being found. It is represented by a mean,
                a standard deviation and a sample from the respective Gaussian distribution.
            k (int): Number of nearest neighbors to find.

        Returns: List of indexes in ``x1`` from the ``k`` nearest neighbors of ``x2``.

        """
        distances = []
        for var in range(len(x1) - 1):  # Only the mean and standard deviation are considered.
            cur_var_x2 = x2[var, :].reshape(1, -1)
            cur_distances = []
            for cur_var_x1 in x1[var, :]:
                cur_distances.append(np.linalg.norm(cur_var_x1.reshape(1, -1) - cur_var_x2))
            distances.append(np.asarray(cur_distances))

        distances = np.asarray(distances)
        distances = np.sum(distances, axis=0)
        k = len(distances) - 1 if len(distances) < k else k
        return distances.argsort()[:k]

    def fit(self, X, y=None, **fit_params):
        """
        Fits the Variational Autoencoder model used for filtering purposes.

        Args:
            X: Data used to train the Variational Autoencoder.
            y: Not applicable. This parameter only exists to maintain compatibility with the scikit-learn architecture.
            **fit_params: Can be used to supply an optional validation dataset ``X_val``.

        Returns: Instance of self.

        """
        if not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a NumPy Array.")
        if np.isnan(X).astype(int).sum() > 0:
            raise TypeError(f"'X' must not contain missing values.")

        X_val = None
        if "X_val" in fit_params:
            X_val = fit_params["X_val"]
            if not isinstance(X_val, np.ndarray):
                raise TypeError("'X_val' must be a NumPy Array.")
            if np.isnan(X_val).astype(int).sum() > 0:
                raise TypeError(f"'X_val' must not contain missing values.")

        self._binary_features = []
        for f in range(X.shape[1]):
            X_f = X[:, f]
            if np.unique(X_f[~np.isnan(X_f)]).shape[0] <= 2:
                self._binary_features.append(f)

        X_wout_mf = np.delete(X, self._missing_feature_idx, axis=1)
        X_val_wout_mf = None

        if X_val is None:
            self._X_train_val = X
            X_train_val_wout_mf = X_wout_mf
        else:
            X_val_wout_mf = np.delete(X_val, self._missing_feature_idx, axis=1)
            self._X_train_val = np.concatenate((X, X_val), axis=0)
            X_train_val_wout_mf = np.concatenate((X_wout_mf, X_val_wout_mf), axis=0)

        self._vae_model.fit(X_wout_mf, X_wout_mf, X_val_wout_mf, X_val_wout_mf)
        self._encoded_data_train = self._vae_model.encode(X_train_val_wout_mf)
        self._fitted = True
        return self

    def transform(self, X, y=None):
        """
        Performs the imputation of missing values in the feature ``_missing_feature_idx``
        of ``X`` using the VAE-BRIDGE method.

        Args:
            X: Data to be imputed, which contains missing values in the feature ``_missing_feature_idx``.
            y: Not applicable. This parameter only exists to maintain compatibility with the scikit-learn architecture.

        Returns: ``X`` with the missing values from the feature ``_missing_feature_idx`` already imputed.

        """
        if not self._fitted:
            raise RuntimeError("The fit method must be called before transform.")
        if not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a NumPy Array.")

        X_wout_mf = np.delete(X, self._missing_feature_idx, axis=1)
        if np.isnan(X_wout_mf).astype(int).sum() > 0:
            raise TypeError(f"'X' can only contain missing values in feature {self._missing_feature_idx}.")

        X_mask = np.isnan(X)
        X_imputed = X.copy()
        k_abs = int(self._k * self._encoded_data_train[0].shape[0])
        k_abs = k_abs if k_abs > 0 else 1  # The minimum of instances is 1.

        for i in np.where(X_mask[:, self._missing_feature_idx])[0]:
            encoded_data_transform = self._vae_model.encode(X_wout_mf[i, :].reshape(1, -1))
            min_k_idx = self._get_k_nearest_neighbors(self._encoded_data_train, encoded_data_transform, k=k_abs)

            selected_instances = self._X_train_val[min_k_idx]
            y_reg = selected_instances[:, self._missing_feature_idx].reshape(-1, 1).ravel()
            x_reg = np.delete(selected_instances, self._missing_feature_idx, axis=1)
            self._regression_model.fit(x_reg, y_reg)

            y_pred = self._regression_model.predict(X_wout_mf[i, :].reshape(1, -1)).reshape(1, -1)
            X_imputed[i, self._missing_feature_idx] = y_pred[0]

        if self._missing_feature_idx in self._binary_features:
            X_imputed[:, self._missing_feature_idx] = np.around(np.clip(X_imputed[:, self._missing_feature_idx], 0, 1))

        return X_imputed
