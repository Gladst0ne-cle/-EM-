import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class GMMResult:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    responsibilities: np.ndarray
    labels: np.ndarray
    log_likelihood_history: list


class GaussianMixtureEM:
    """
    A minimal EM implementation for Gaussian Mixture Model.
    Supports full covariance matrices.
    """

    def __init__(self, n_components=3, max_iter=200, tol=1e-5, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihood_history_ = []

    @staticmethod
    def _multivariate_normal_pdf(x, mean, cov):
        dim = x.shape[1]
        cov = cov + 1e-6 * np.eye(dim)
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** dim) * det_cov)
        diff = x - mean
        expo = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
        return norm_const * np.exp(expo)

    def _initialize_parameters(self, x):
        n_samples, n_features = x.shape
        indices = self.rng.choice(n_samples, self.n_components, replace=False)
        means = x[indices].copy()

        base_cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(n_features)
        covariances = np.array([base_cov.copy() for _ in range(self.n_components)])
        weights = np.ones(self.n_components) / self.n_components
        return means, covariances, weights

    def _e_step(self, x, means, covariances, weights):
        n_samples = x.shape[0]
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = weights[k] * self._multivariate_normal_pdf(
                x, means[k], covariances[k]
            )

        gamma_sum = np.sum(gamma, axis=1, keepdims=True) + 1e-12
        gamma /= gamma_sum
        return gamma

    def _m_step(self, x, gamma):
        n_samples, n_features = x.shape
        nk = np.sum(gamma, axis=0) + 1e-12

        weights = nk / n_samples
        means = (gamma.T @ x) / nk[:, None]

        covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = x - means[k]
            covariances[k] = (gamma[:, k][:, None] * diff).T @ diff / nk[k]
            covariances[k] += 1e-6 * np.eye(n_features)

        return means, covariances, weights

    def _log_likelihood(self, x, means, covariances, weights):
        total = np.zeros(x.shape[0])
        for k in range(self.n_components):
            total += weights[k] * self._multivariate_normal_pdf(x, means[k], covariances[k])
        return np.sum(np.log(total + 1e-12))

    def _check_is_fitted(self):
        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise ValueError("Model is not fitted. Call fit or fit_predict first.")

    def fit(self, x):
        means, covariances, weights = self._initialize_parameters(x)
        history = []

        prev_ll = None
        gamma = None
        for _ in range(self.max_iter):
            gamma = self._e_step(x, means, covariances, weights)
            means, covariances, weights = self._m_step(x, gamma)
            ll = self._log_likelihood(x, means, covariances, weights)
            history.append(ll)

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.means_ = means
        self.covariances_ = covariances
        self.weights_ = weights
        self.log_likelihood_history_ = history
        return self

    def predict_proba(self, x):
        self._check_is_fitted()
        return self._e_step(x, self.means_, self.covariances_, self.weights_)

    def predict(self, x):
        gamma = self.predict_proba(x)
        return np.argmax(gamma, axis=1)

    def score_samples(self, x):
        self._check_is_fitted()
        density = np.zeros(x.shape[0])
        for k in range(self.n_components):
            density += self.weights_[k] * self._multivariate_normal_pdf(
                x, self.means_[k], self.covariances_[k]
            )
        return np.log(density + 1e-12)

    def bic(self, x):
        """
        Bayesian Information Criterion for model selection.
        """
        self._check_is_fitted()
        n_samples, n_features = x.shape
        # Parameters: weights(K-1) + means(K*d) + covariances(K*d*(d+1)/2)
        n_params = (
            (self.n_components - 1)
            + self.n_components * n_features
            + self.n_components * n_features * (n_features + 1) / 2
        )
        ll = np.sum(self.score_samples(x))
        return -2 * ll + n_params * np.log(n_samples)

    def fit_predict(self, x):
        self.fit(x)
        gamma = self.predict_proba(x)
        labels = np.argmax(gamma, axis=1)
        return GMMResult(
            means=self.means_,
            covariances=self.covariances_,
            weights=self.weights_,
            responsibilities=gamma,
            labels=labels,
            log_likelihood_history=self.log_likelihood_history_,
        )


def clustering_accuracy(y_true, y_pred, n_classes):
    """
    Clustering labels are permutation-invariant.
    We brute-force all permutations for small class count.
    """
    best_acc = 0.0
    best_perm = None

    for perm in itertools.permutations(range(n_classes)):
        mapped = np.array([perm[label] for label in y_pred])
        acc = np.mean(mapped == y_true)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, best_perm
