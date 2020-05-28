import math
import numpy as np


def inertia_factory(feat_metric, sample_metric):
    def feat_inner(u, v):
        return (u[None, :] @ feat_metric @ v[:, None]).item()

    def inertia_point(X, p=None):
        """Calcule l'inertie de `X` par rapport au point `p`. Si le point `p`
        n'est pas fourni prendre le centre de gravité.
        """

        if p is None:
            # Calcul du centre de gravité pondéré par `sample_metric`
            p = np.sum(sample_metric @ X, axis=0)

        # Linéarisation au cas où...
        p = p.squeeze()

        # Centrer par rapport à `p`
        Xp = X - p

        # Utiliser la formule du cours avec la trace.
        inertia = sum(np.diag(Xp.T @ sample_metric @ Xp @ feat_metric))

        return inertia

    def inertia_axe(X, v, p=None):
        """Calcule l'inertie de `X` par rapport à la droite pointée par `p` et
        de direction `v`. Si `p` n'est pas fourni, on prend la droite
        qui passe par l'origine (p=0).
        """

        if p is None:
            p = np.zeros(2)

        # Linéarisation au cas où...
        p = p.squeeze()

        def dist_axe_point(v, p, q):
            """Distance d'un point `q` à une droite repérée par un
            vecteur `v` et un point `p`.
            """

            # Normalisation du vecteur `v`
            v = v.ravel()
            v = v / math.sqrt(feat_inner(v, v))

            # Calculer le vecteur perpendiculaire à la droite
            # d'origine q et dont l'extrémité appartient à la droite.
            q_projq = (p - q) - feat_inner(p - q, v) * v

            # Retourner la norme de `q_projq`
            return feat_inner(q_projq, q_projq)

        # Matrice de toutes les distances au carré à la droite
        dists = [dist_axe_point(v, p, q) for q in X]

        # Somme pondérée en fonction de `sample_metric`
        inertia = sum(np.diag(sample_metric) * np.array(dists))

        return inertia

    return feat_inner, inertia_point, inertia_axe


if __name__ == '__main__':
    # Exemple d'utilisation de `inertia_factory`.
    n, p = 100, 2

    # Matrice M
    feat_metric = np.eye(p)

    # Matrice D_p
    sample_metric = 1 / n * np.eye(n)

    # Création de `inertia_point` et `inertia_axe` qui calculent
    # l'inertie par rapport à un point et l'inertie par rapport à un
    # axe avec les métriques fournies.
    feat_inner, inertia_point, inertia_axe = inertia_factory(feat_metric, sample_metric)
