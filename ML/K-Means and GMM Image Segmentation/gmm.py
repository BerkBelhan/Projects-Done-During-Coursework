import numpy as np

class GMM:
    def __init__(self, K=3, max_iters=50):
        self.K = K
        self.max_iters = max_iters

    def initialize(self, X):#inilizing function for gmm 
        N, D = X.shape

        indices = np.random.choice(N, self.K, replace=False)

        self.means = X[indices]

        self.covariances = np.array([
            np.eye(D) for _ in range(self.K)
        ])

        self.weights = np.ones(self.K) / self.K

    def gaussian(self, X, mean, covariance):#computing gaussian distribution for gmm

        D = X.shape[1]

        # Stabilize covariance matrix
        covariance += 1e-6 * np.eye(D)

        cov_inv = np.linalg.inv(covariance)

        cov_det = np.linalg.det(covariance)

        # Prevent invalid determinant to prevent divide-by-zero, 
        # Bfore it resulted in black images due to NaN values in responsibilities
        if cov_det <= 0:
            cov_det = 1e-6

        norm_const = 1.0 / np.sqrt(
            ((2 * np.pi) ** D) * cov_det
        )

        diff = X - mean

        exponent = np.einsum(
            'ij,jk,ik->i',
            diff,
            cov_inv,
            diff
        )

        return norm_const * np.exp(-0.5 * exponent)

    def expectation(self, X):#computing responsibilities for gmm, which represents the probability of each data point belonging to each cluster

        N = X.shape[0]

        responsibilities = np.zeros((N, self.K))

        for k in range(self.K):

            responsibilities[:, k] = (
                self.weights[k] *
                self.gaussian(
                    X,
                    self.means[k],
                    self.covariances[k]
                )
            )

        row_sums = responsibilities.sum(
            axis=1,
            keepdims=True
        )

        row_sums[row_sums == 0] = 1e-10

        responsibilities /= row_sums

        return responsibilities

    def maximization(self, X, responsibilities):#updating parameters (weights, means, covariances) based on the computed responsibilities to maximize the likelihood of the data under the model

        N, D = X.shape

        Nk = responsibilities.sum(axis=0)

        # Prevent divide-by-zero
        Nk[Nk == 0] = 1e-10

        self.weights = Nk / N

        self.means = (
            responsibilities.T @ X
        ) / Nk[:, np.newaxis]

        self.covariances = []

        for k in range(self.K):#updating covariance matrices for each cluster based on the responsibilities and the difference between data points 

            diff = X - self.means[k]

            cov = (
                responsibilities[:, k][:, np.newaxis]
                * diff
            ).T @ diff

            cov /= Nk[k]

            # Stabilizing covariance
            cov += 1e-5 * np.eye(D)

            self.covariances.append(cov)

        self.covariances = np.array(self.covariances)

    def fit(self, X):

        self.initialize(X)

        for _ in range(self.max_iters):

            responsibilities = self.expectation(X)

            self.maximization(X, responsibilities)

        labels = np.argmax(
            responsibilities,
            axis=1
        )

        return labels, self.means