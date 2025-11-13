import numpy as np
from scipy.linalg import eigh


class CSP:
    """
    Common Spatial Pattern (CSP) for EEG feature extraction.
    
    CSP finds spatial filters that maximize variance for one class while 
    minimizing it for another class. Useful for motor imagery BCI tasks.
    
    Parameters
    ----------
    n_components : int
        Number of CSP components to retain (pairs of filters from each end of eigenvalue spectrum).
        Total filters = n_components * 2
    reg : float or None
        Regularization parameter for covariance matrices (Ledoit-Wolf shrinkage).
        If None, no regularization. Values typically between 0 and 1.
    log : bool
        If True, apply log-variance transform to features (recommended for classification).
    """
    
    def __init__(self, n_components=4, reg=None, log=True):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None
        
    def _regularize_cov(self, cov, n_samples):
        """Apply Ledoit-Wolf regularization to covariance matrix."""
        if self.reg is None or self.reg == 0:
            return cov
            
        # Ledoit-Wolf shrinkage
        alpha = self.reg
        trace = np.trace(cov)
        n_channels = cov.shape[0]
        target = (trace / n_channels) * np.eye(n_channels)
        return (1 - alpha) * cov + alpha * target
    
    def fit(self, X, y):
        """
        Fit CSP filters on labeled training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_samples)
            Training EEG data
        y : np.ndarray, shape (n_trials,)
            Class labels (0 or 1, or -1 and 1)
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n_trials, n_channels, n_samples), got shape {X.shape}")
        
        # Get unique classes
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")
        
        # Normalize labels to 0 and 1
        y_binary = (y == classes[1]).astype(int)
        
        n_trials, n_channels, n_samples = X.shape
        
        # Compute covariance matrices for each class
        cov_class0 = np.zeros((n_channels, n_channels))
        cov_class1 = np.zeros((n_channels, n_channels))
        
        n_class0 = 0
        n_class1 = 0
        
        for trial_idx in range(n_trials):
            trial = X[trial_idx]  # (n_channels, n_samples)
            
            # Compute normalized covariance matrix for this trial
            trial_cov = np.dot(trial, trial.T) / (n_samples - 1)
            trace = np.trace(trial_cov)
            trial_cov /= trace  # Normalize by trace
            
            if y_binary[trial_idx] == 0:
                cov_class0 += trial_cov
                n_class0 += 1
            else:
                cov_class1 += trial_cov
                n_class1 += 1
        
        # Average covariances
        cov_class0 /= n_class0
        cov_class1 /= n_class1
        
        # Apply regularization if specified
        cov_class0 = self._regularize_cov(cov_class0, n_class0)
        cov_class1 = self._regularize_cov(cov_class1, n_class1)
        
        # Solve generalized eigenvalue problem
        # We want to find W such that: W' C0 W = D and W' C1 W = I - D
        # This is equivalent to: (C0 + C1) W = C1 W Lambda
        cov_combined = cov_class0 + cov_class1
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov_class0, cov_combined)
        
        # Sort by eigenvalues (descending)
        ix = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[ix]
        eigenvectors = eigenvectors[:, ix]
        
        # Select filters: n_components from each end of spectrum
        # (highest variance for class 0, highest variance for class 1)
        n_comp = self.n_components
        selected_indices = np.concatenate([
            np.arange(n_comp),  # Highest eigenvalues (class 0)
            np.arange(n_channels - n_comp, n_channels)  # Lowest eigenvalues (class 1)
        ])
        
        self.filters_ = eigenvectors[:, selected_indices].T  # (n_filters, n_channels)
        
        # Compute patterns (inverse of filters, normalized)
        # Patterns show the spatial distribution of sources
        self.patterns_ = np.linalg.pinv(self.filters_)
        
        return self
    
    def transform(self, X, normalize=True):
        """
        Transform data using fitted CSP filters.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
            Data to transform
        normalize : bool
            If True, apply z-score normalization to features
            
        Returns
        -------
        features : np.ndarray, shape (n_trials, n_filters) or (n_filters,)
            CSP features (log-variance of filtered signals if self.log=True)
        """
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before transform. Call fit() first.")
        
        X = np.asarray(X)
        
        # Handle single trial
        single_trial = False
        if X.ndim == 2:
            X = X[np.newaxis, :, :]  # Add trial dimension
            single_trial = True
        
        if X.ndim != 3:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
        
        n_trials, n_channels, n_samples = X.shape
        n_filters = self.filters_.shape[0]
        
        features = np.zeros((n_trials, n_filters))
        
        for trial_idx in range(n_trials):
            trial = X[trial_idx]  # (n_channels, n_samples)
            
            # Apply spatial filters: filtered = W * trial
            filtered = np.dot(self.filters_, trial)  # (n_filters, n_samples)
            
            # Compute variance (or log-variance) for each filter
            if self.log:
                # Log-variance features (more suitable for LDA)
                var = np.var(filtered, axis=1)
                features[trial_idx] = np.log(var + 1e-10)  # Add small constant for numerical stability
            else:
                # Raw variance features
                features[trial_idx] = np.var(filtered, axis=1)
        
        # Normalize features if requested
        if normalize:
            if self.mean_ is None:
                # First time: compute mean and std
                self.mean_ = np.mean(features, axis=0)
                self.std_ = np.std(features, axis=0) + 1e-10
            features = (features - self.mean_) / self.std_
        
        # Return single trial if input was single trial
        if single_trial:
            return features[0]
        
        return features
    
    def fit_transform(self, X, y, normalize=True):
        """
        Fit CSP filters and transform training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_samples)
            Training data
        y : np.ndarray, shape (n_trials,)
            Class labels
        normalize : bool
            If True, apply z-score normalization to features
            
        Returns
        -------
        features : np.ndarray, shape (n_trials, n_filters)
            CSP features
        """
        self.fit(X, y)
        return self.transform(X, normalize=normalize)
    
    def get_spatial_patterns(self):
        """
        Get spatial patterns (inverse of filters).
        
        Returns
        -------
        patterns : np.ndarray, shape (n_channels, n_filters)
            Spatial patterns for visualization
        """
        if self.patterns_ is None:
            raise RuntimeError("CSP must be fitted first.")
        return self.patterns_
    
    def get_filters(self):
        """
        Get spatial filters.
        
        Returns
        -------
        filters : np.ndarray, shape (n_filters, n_channels)
            Spatial filters
        """
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted first.")
        return self.filters_