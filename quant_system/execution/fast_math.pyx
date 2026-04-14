# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp, fabs

# Initialize numpy C-API
np.import_array()

cpdef double calculate_obi_cython(double bid_size, double ask_size) noexcept:
    """Calculates OBI with microsecond precision."""
    if bid_size + ask_size == 0:
        return 0.0
    return (bid_size - ask_size) / (bid_size + ask_size)

cpdef double calculate_vpin_cython(np.ndarray[np.float64_t, ndim=1] buy_vols, np.ndarray[np.float64_t, ndim=1] sell_vols, double bucket_vol) noexcept:
    """Calculates vPIN (Volume-Synchronized Probability of Informed Trading)."""
    cdef int n = buy_vols.shape[0]
    if n == 0 or bucket_vol == 0:
        return 0.0
        
    cdef double imb_sum = 0.0
    cdef int i
    for i in range(n):
        imb_sum += fabs(buy_vols[i] - sell_vols[i])
        
    return imb_sum / (n * bucket_vol)

cpdef tuple cython_kalman_update(
    double price_a, 
    double price_b, 
    np.ndarray[np.float64_t, ndim=1] theta, 
    np.ndarray[np.float64_t, ndim=2] covariance,
    double process_noise,
    double observation_noise
):
    """
    Optimized 2x2 Kalman Filter Update for Pairs Trading.
    Returns: (new_theta, new_covariance, innovation, innovation_variance)
    """
    cdef double obs_x = price_b
    cdef double obs_y = 1.0 # Intercept 
    
    # 1. Prediction Step
    # Predicted Theta is theta (random walk assumption)
    # Predicted Covariance = covariance + Q
    cdef double p_cov00 = covariance[0, 0] + process_noise
    cdef double p_cov01 = covariance[0, 1]
    cdef double p_cov10 = covariance[1, 0]
    cdef double p_cov11 = covariance[1, 1] + process_noise
    
    # 2. Innovation
    cdef double predicted_price_a = obs_x * theta[0] + obs_y * theta[1]
    cdef double innovation = price_a - predicted_price_a
    
    # 3. Innovation Variance
    # S = H * P * H^T + R
    cdef double s = (obs_x * p_cov00 + obs_y * p_cov10) * obs_x + \
                    (obs_x * p_cov01 + obs_y * p_cov11) * obs_y + \
                    observation_noise
                    
    # 4. Kalman Gain
    # K = P * H^T * inv(S)
    cdef double k0 = (p_cov00 * obs_x + p_cov01 * obs_y) / s
    cdef double k1 = (p_cov10 * obs_x + p_cov11 * obs_y) / s
    
    # 5. Update State
    cdef np.ndarray[np.float64_t, ndim=1] new_theta = np.empty(2, dtype=np.float64)
    new_theta[0] = theta[0] + k0 * innovation
    new_theta[1] = theta[1] + k1 * innovation
    
    # 6. Update Covariance
    # P = (I - K*H) * P
    cdef np.ndarray[np.float64_t, ndim=2] new_cov = np.empty((2, 2), dtype=np.float64)
    # (I - KH) = [[1 - k0*x, -k0*y], [-k1*x, 1 - k1*y]]
    new_cov[0, 0] = (1.0 - k0 * obs_x) * p_cov00 - (k0 * obs_y) * p_cov10
    new_cov[0, 1] = (1.0 - k0 * obs_x) * p_cov01 - (k0 * obs_y) * p_cov11
    new_cov[1, 0] = (-k1 * obs_x) * p_cov00 + (1.0 - k1 * obs_y) * p_cov10
    new_cov[1, 1] = (-k1 * obs_x) * p_cov01 + (1.0 - k1 * obs_y) * p_cov11
    
    return (new_theta, new_cov, innovation, s)
