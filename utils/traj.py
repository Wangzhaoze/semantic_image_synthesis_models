from typing import Dict, List, Tuple, Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



class TrajectorySmoother:
    """
    A class for smoothing trajectory data using various filtering techniques.
    """

    def __init__(self):
        pass

    @staticmethod
    def moving_average(data: Union[List[float], np.ndarray], window: int) -> List[float]:
        """
        Apply moving average filter to the input data.
        
        Args:
            data: Input data sequence
            window: Size of the moving window
            
        Returns:
            Smoothed data using moving average
        """
        if window <= 0:
            raise ValueError("Window size must be positive")
        if len(data) == 0:
            return []
            
        return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values.tolist()

    @staticmethod
    def zscore_filter(data: Union[List[float], np.ndarray], threshold: float = 3.0) -> List[float]:
        """
        Filter outliers using Z-score method.
        
        Args:
            data: Input data sequence
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Filtered data with outliers removed
        """
        if len(data) == 0:
            return []
            
        z_scores = np.abs(zscore(data))
        filtered_data = [data[i] for i in range(len(data)) if z_scores[i] < threshold]
        return filtered_data

    @staticmethod
    def median_filter(data: Union[List[float], np.ndarray], window: int) -> List[float]:
        """
        Apply median filter to the input data.
        
        Args:
            data: Input data sequence
            window: Size of the median filter window
            
        Returns:
            Smoothed data using median filter
        """
        if window <= 0:
            raise ValueError("Window size must be positive")
        if len(data) == 0:
            return []
            
        return pd.Series(data).rolling(window=window, center=True, min_periods=1).median().values.tolist()

    @staticmethod
    def kalman_filter(data: Union[List[float], np.ndarray], 
                     process_variance: float = 1e-5, 
                     measurement_variance: float = 0.1) -> List[float]:
        """
        Apply Kalman filter to the input data.
        
        Args:
            data: Input data sequence
            process_variance: Variance of the process noise
            measurement_variance: Variance of the measurement noise
            
        Returns:
            Smoothed data using Kalman filter
        """
        if len(data) == 0:
            return []
            
        n = len(data)
        # Initialize arrays
        xhat = np.zeros(n)      # Estimated state
        P = np.zeros(n)         # Estimation error covariance
        
        # Initial values
        xhat[0] = data[0]
        P[0] = 1.0
        
        for k in range(1, n):
            # Prediction step
            xhat_minus = xhat[k-1]
            P_minus = P[k-1] + process_variance
            
            # Update step (Kalman gain)
            K = P_minus / (P_minus + measurement_variance)
            xhat[k] = xhat_minus + K * (data[k] - xhat_minus)
            P[k] = (1 - K) * P_minus
            
        return xhat.tolist()

    @staticmethod
    def ransac_filter(data: Union[List[float], np.ndarray], 
                     residual_threshold: float = 1.0,
                     polynomial_degree: int = 2) -> List[float]:
        """
        Apply RANSAC regression to filter outliers.
        
        Args:
            data: Input data sequence
            residual_threshold: Threshold for considering inliers
            polynomial_degree: Degree of polynomial for regression
            
        Returns:
            Filtered data using RANSAC
        """
        if len(data) < 2:
            return data
            
        n = len(data)
        t = np.arange(n).reshape(-1, 1)
        
        # Create polynomial regression model with RANSAC
        model = make_pipeline(
            PolynomialFeatures(degree=polynomial_degree),
            RANSACRegressor(
                residual_threshold=residual_threshold,
                random_state=42
            )
        )
        
        # Fit the model
        model.fit(t, data)
        
        # Get the RANSAC regressor from the pipeline
        ransac = model.steps[-1][1]
        
        # Predict using the model
        data_pred = model.predict(t)
        
        # Replace outliers with predicted values
        inlier_mask = ransac.inlier_mask_
        return np.where(inlier_mask, data, data_pred).tolist()

    @staticmethod
    def savgol_filter(data: Union[List[float], np.ndarray], 
                     window_length: int, 
                     polyorder: int) -> List[float]:
        """
        Apply Savitzky-Golay filter to the input data.
        
        Args:
            data: Input data sequence
            window_length: Length of the filter window
            polyorder: Order of the polynomial used for fitting
            
        Returns:
            Smoothed data using Savitzky-Golay filter
        """
        from scipy.signal import savgol_filter
        
        if len(data) < window_length:
            return data
            
        return savgol_filter(data, window_length, polyorder).tolist()

    @staticmethod
    def exponential_smoothing(data: Union[List[float], np.ndarray], 
                             alpha: float = 0.3) -> List[float]:
        """
        Apply exponential smoothing to the input data.
        
        Args:
            data: Input data sequence
            alpha: Smoothing factor (0 < alpha < 1)
            
        Returns:
            Smoothed data using exponential smoothing
        """
        if len(data) == 0:
            return []
            
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[i-1])
        return result