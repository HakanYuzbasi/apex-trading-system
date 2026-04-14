import numpy as np

def calculate_obi_cython(bid_size: float, ask_size: float) -> float:
    """Python fallback for OBI calculation."""
    if bid_size + ask_size == 0:
        return 0.0
    return (bid_size - ask_size) / (bid_size + ask_size)

def calculate_vpin_cython(buy_vols: np.ndarray, sell_vols: np.ndarray, bucket_vol: float) -> float:
    """Python fallback for vPIN calculation."""
    n = buy_vols.shape[0]
    if n == 0 or bucket_vol == 0:
        return 0.0
    
    imb_sum = np.sum(np.abs(buy_vols - sell_vols))
    return imb_sum / (n * bucket_vol)

def cython_kalman_update(
    price_a: float, 
    price_b: float, 
    theta: np.ndarray, 
    covariance: np.ndarray,
    process_noise: float,
    observation_noise: float
):
    """Python fallback for Kalman Filter update."""
    obs_x = price_b
    obs_y = 1.0 # Intercept 
    
    # 1. Prediction Step
    p_cov00 = covariance[0, 0] + process_noise
    p_cov01 = covariance[0, 1]
    p_cov10 = covariance[1, 0]
    p_cov11 = covariance[1, 1] + process_noise
    
    # 2. Innovation
    predicted_price_a = obs_x * theta[0] + obs_y * theta[1]
    innovation = price_a - predicted_price_a
    
    # 3. Innovation Variance
    s = (obs_x * p_cov00 + obs_y * p_cov10) * obs_x + \
        (obs_x * p_cov01 + obs_y * p_cov11) * obs_y + \
        observation_noise
                    
    # 4. Kalman Gain
    k0 = (p_cov00 * obs_x + p_cov01 * obs_y) / s
    k1 = (p_cov10 * obs_x + p_cov11 * obs_y) / s
    
    # 5. Update State
    new_theta = np.zeros(2, dtype=np.float64)
    new_theta[0] = theta[0] + k0 * innovation
    new_theta[1] = theta[1] + k1 * innovation
    
    # 6. Update Covariance
    new_cov = np.zeros((2, 2), dtype=np.float64)
    new_cov[0, 0] = (1.0 - k0 * obs_x) * p_cov00 - (k0 * obs_y) * p_cov10
    new_cov[0, 1] = (1.0 - k0 * obs_x) * p_cov01 - (k0 * obs_y) * p_cov11
    new_cov[1, 0] = (-k1 * obs_x) * p_cov00 + (1.0 - k1 * obs_y) * p_cov10
    new_cov[1, 1] = (-k1 * obs_x) * p_cov01 + (1.0 - k1 * obs_y) * p_cov11
    
    return (new_theta, new_cov, innovation, s)
