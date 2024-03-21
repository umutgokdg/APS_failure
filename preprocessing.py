import numpy as np
import pandas as pd

class LDA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        n_components = X.shape[1]
        class_labels = np.unique(y)
        
        mean_overall = np.mean(X, axis=0)
        self.mean_ = mean_overall

        S_B = np.zeros((n_components, n_components))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            n_c = X_c.shape[0]
            
            mean_diff = (mean_c - mean_overall).reshape(n_components, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)
        
        # Within class scatter
        S_W = np.zeros((n_components, n_components))
        for c in class_labels:
            X_c = X[y == c]
            S_W += (X_c - mean_c).T @ (X_c - mean_c)

        # Determine SW^-1 SB
        A = np.linalg.inv(S_W) @ S_B
        # Get eigenvalues and eigenvectors of SW^-1 SB
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[idxs]

        #get the first n_components eigenvectors
        self.L = eigenvectors[:, :self.n_components]

        # compute variance explained ratio
        self.explained_variance_ratio = np.sum(eigenvalues[:self.n_components]) / np.sum(eigenvalues)



    def transform(self, X) -> np.ndarray:
        X_centered = X - self.mean_
        print(X_centered.shape, self.L.shape)
        F = X_centered.dot(self.L)
        return F

    def fit_transform(self, X, y) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Step 2: Compute covariance matrix
        cov_matrix = np.cov(X.T)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4: Sort eigenvectors by descending eigenvalues
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Select the top n_components eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Step 5: Project data
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FactorAnalysis():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        """
        Fit the model with X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        #create covariance matrix
        cov_matrix = np.cov(X_centered.T)

        #compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        #sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        #get the first n_components eigenvectors
        self.L = eigenvectors[:, :self.n_components]

        #get the first n_components eigenvalues
        self.eigenvalues = eigenvalues[:self.n_components]

        #explained variance
        self.explained_variance = self.eigenvalues / np.sum(eigenvalues)

        #explained variance ratio
        self.explained_variance_ratio = np.sum(self.explained_variance / np.sum(eigenvalues/np.sum(eigenvalues)))




    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            New data to transform.
        """
        X_centered = X - self.mean_

        # X = L*F
        # L^t*X = L^t*L*F 
        # (L^t*L)^-1 * L^t*X = F

        #compute the factor matrix
        F = np.linalg.inv(self.L.T.dot(self.L)).dot(self.L.T).dot(X_centered.T).T
        return F
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        self.fit(X)
        return self.transform(X)