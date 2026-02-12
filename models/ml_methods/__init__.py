"""
ML Methods Package - Additional model types for the APEX ensemble.

All models follow the scikit-learn interface (.fit/.predict) so they
integrate directly into UltimateSignalGenerator.initializeModels().
"""

from models.ml_methods.elastic_net import ElasticNetRegressor
from models.ml_methods.bayesian_ridge import BayesianRidgeRegressor
from models.ml_methods.svr_model import SVRRegressor
from models.ml_methods.gaussian_process import GPRegressor
from models.ml_methods.anomaly_detector import AnomalyAwareRegressor
from models.ml_methods.stacking_ensemble import StackingMetaLearner
from models.ml_methods.catboost_model import CatBoostRegressorWrapper

# Deep learning models (optional - require torch)
try:
    from models.ml_methods.lstm_model import LSTMRegressor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from models.ml_methods.transformer_model import TransformerRegressor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

__all__ = [
    "ElasticNetRegressor",
    "BayesianRidgeRegressor",
    "SVRRegressor",
    "GPRegressor",
    "AnomalyAwareRegressor",
    "StackingMetaLearner",
    "CatBoostRegressorWrapper",
    "LSTM_AVAILABLE",
    "TRANSFORMER_AVAILABLE",
]

if LSTM_AVAILABLE:
    __all__.append("LSTMRegressor")
if TRANSFORMER_AVAILABLE:
    __all__.append("TransformerRegressor")
