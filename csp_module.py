from scipy.linalg import eig
import numpy as np

class CSPFilter:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.filters = None
        self.eigenvalues = None
    
    def fit(self, X_left, X_right):
        """Train on left/right motor imagery trials"""
        # Algorithm from PDF Section 4
        
        # Compute normalized covariances
        cov_left = []
        for trial in X_left:
            C = trial @ trial.T
            C = C / np.trace(C)
            cov_left.append(C)
        C_left = np.mean(cov_left, axis=0)
        
        cov_right = []
        for trial in X_right:
            C = trial @ trial.T
            C = C / np.trace(C)
            cov_right.append(C)
        C_right = np.mean(cov_right, axis=0)
        
        # Solve generalized eigenvalue problem
        C_composite = C_left + C_right
        eigenvalues, eigenvectors = eig(C_left, C_composite)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort and select filters
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        m = self.n_components
        self.filters = np.column_stack([
            eigenvectors[:, :m],
            eigenvectors[:, -m:]
        ])
        self.eigenvalues = eigenvalues
        
        return self
    
    def transform(self, X):
        """Extract log-variance features"""
        if self.filters is None:
            raise ValueError("Must fit before transform")
        
        single_trial = False
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
            single_trial = True
        
        features = []
        for trial in X:
            Z = self.filters.T @ trial
            variances = np.var(Z, axis=1)
            total_var = np.sum(variances)
            features_trial = np.log(variances / total_var)
            features.append(features_trial)
        
        features = np.array(features)
        return features[0] if single_trial else features