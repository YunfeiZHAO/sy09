import numpy as np
import numpy.linalg as linalg
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin


class AdaptiveKMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, n_init=10, tol=1e-4, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y=None):
        n, p = X.shape
        n_init = self.n_init
        n_clusters = self.n_clusters
        max_iter = self.max_iter
        tol = self.tol

        centers_opt = None
        Vt_opt = None
        partition_opt = None
        d_opt = float('inf')

        for i in range(n_init):
            # Initialisation des centres des classes avec
            # `np.random.choice`
            centers = ...

            # Initialisation des matrices de variance--covariance
            # brutes et normalisées
            Vt = ...
            Vt_norm = ...

            step = tol + 1
            it = 0

            while step > tol and it < max_iter:
                old_centers = centers

                # Calcul d'une nouvelle partition
                dist = np.concatenate([
                    cdist(c[None, :], X, VI=linalg.inv(V))
                    for c, V in zip(centers, Vt_norm)
                ])
                partition = np.argmin(dist, axis=0)

                # Mise à jour des paramètres
                for k in range(n_clusters):
                    # Extraction des individus de class k
                    Xk = ...

                    # Calcul du k-ième centre
                    centers[k, :] = ...

                    # Calcul de la k-ième matrice de
                    # variance-covariance normalisée avec `np.cov` et
                    # `linalg.det`
                    Vt[k] = ...

                step = ((old_centers - centers)**2).sum()
                it += 1

            # Calcul de `d_tot`. On utilisera les instructions
            # permettant de calculer `dist` (voir plus haut).
            d_tot = ...

            # Mise à jour du modèle optimal si besoin
            if d_tot < d_opt:
                centers_opt = centers
                Vt_opt = Vt
                partition_opt = partition
                d_opt = d_tot

        self.labels_ = partition_opt
        self.cluster_centers_ = centers_opt
        self.covars_ = Vt_opt
