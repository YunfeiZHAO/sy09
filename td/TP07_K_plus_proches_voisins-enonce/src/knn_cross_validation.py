from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from knn_validation import accuracy
from sklearn.utils import check_X_y


def knn_cross_validation(X, y, n_folds, n_neighbors_list):
    """Génère les couples nombre de voisins et précisions correspondantes."""

    # Réutiliser l'implémentation de `knn_multiple_validation` en
    # utilisant `KFold` au lieu de `ShuffleSplit`


def knn_cross_validation2(X, y, n_folds, n_neighbors_list):
    # Générer la même sortie de `knn_cross_validation` en utilisant
    # `cross_val_score`
