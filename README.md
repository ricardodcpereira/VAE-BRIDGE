# VAE-BRIDGE - Variational Autoencoder Filter for Bayesian Ridge Imputation

Codebase for the paper *VAE-BRIDGE: Variational Autoencoder Filter for Bayesian Ridge Imputation of Missing Data*

### Paper Details
- Authors: Ricardo Cardoso Pereira, Pedro Henriques Abreu, Pedro Pereira Rodrigues
- Abstract: The missing data issue is often found in real-world datasets and it is usually handled with imputation strategies that replace the missing values with new data. Recently, generative models such as Variational Autoencoders have been applied for this imputation task. However, they were always used to perform the entire imputation, which has presented limited results when comparing to other state-of-the-art methods. In this work, a new approach called Variational Autoencoder Filter for Bayesian Ridge Imputation is introduced. It uses a Variational Autoencoder at the beginning of the imputation pipeline to filter the instances that are later fitted to a Bayesian ridge regression used to predict the new values. The approach was compared to four state-of-the-art imputation methods using 10 datasets from the healthcare context covering clinical trials, all injected with missing values under different rates. The proposed approach significantly outperformed the remaining methods in all settings, achieving an overall improvement between 26% and 67%.
- Published in: 2020 International Joint Conference on Neural Networks (IJCNN)
- Year: 2020
- DOI: https://doi.org/10.1109/IJCNN48605.2020.9206615
- Contact: rdpereira@dei.uc.pt

### Notes
- The VAE-BRIDGE package follows the scikit-learn architecture, implementing the `fit()` and `transform()` methods.
- The data to be imputed must be a NumPy Array.
- The categorical features must be binarized before running VAE-BRIDGE (e.g., through one-hot encoding).
- The Variational Autoencoder architecture can be customized through the `ConfigVAE` data class. 
- A detailed usage example is available in `tests/test_breast_cancer.py`.

### Quick Start Example
```python
import numpy as np
from vaebridge import ConfigVAE, VAEBRIDGE

data = np.asarray([[0.31, 0.22, 0.69, 0.78],
                   [0.43, 0.23, 0.67, 0.98],
                   [0.58, 0.18, 0.78, 0.96]])

vae_config = ConfigVAE()
vae_config.number_features = 4

vae_bridge_model = VAEBRIDGE(vae_config, missing_feature_idx=2, k=0.2)
vae_bridge_model.fit(data)

new_data = np.asarray([[0.47, 0.31, np.nan, 0.85]])
new_data_imputed = vae_bridge_model.transform(new_data)
```
