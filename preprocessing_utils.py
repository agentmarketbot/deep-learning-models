from typing import Tuple, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted = False
        self._validate_input = True
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a numpy array")
        self.input_shape_ = X.shape[1:]
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_data(X)
        
        try:
            # Add your preprocessing steps here
            X_processed = X.astype('float32')
            X_processed /= 255.0  # Normalize to [0,1]
            return X_processed
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {str(e)}")
    
    def _check_is_fitted(self):
        if not self.is_fitted:
            raise NotFittedError("DataPreprocessor must be fitted before transform")
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a numpy array")
        if len(X.shape) < 2:
            raise ValueError("Input X must be at least 2-dimensional")
        if X.shape[1:] != self.input_shape_:
            raise ValueError(f"Input shape {X.shape[1:]} does not match fitted shape {self.input_shape_}")
        return X

def prepare_data(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training and validation data using consistent preprocessing.
    
    Args:
        X_train: Training data
        X_val: Validation data
    
    Returns:
        Tuple of preprocessed training and validation data
    
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If preprocessing fails
    """
    try:
        preprocessor = DataPreprocessor()
        X_train_prep = preprocessor.fit_transform(X_train)
        X_val_prep = preprocessor.transform(X_val)
        return X_train_prep, X_val_prep
    except Exception as e:
        raise RuntimeError(f"Data preparation failed: {str(e)}")

def predict_with_validation(model: Any, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Make predictions with input validation and error handling.
    
    Args:
        model: Trained model with predict method
        X: Input data
        threshold: Classification threshold for binary problems
    
    Returns:
        Array of predictions
    
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If prediction fails
    """
    try:
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have predict method")
        
        predictions = model.predict(X)
        
        # For binary classification
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            predictions = np.array([1 if x > threshold else 0 for x in predictions])
            
        return predictions
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")