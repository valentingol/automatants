# automatants

Tous les scipts de mes formations pour l'association d'IA de CentraleSupélec [__Les Automatants__](https://automatants.cs-campus.fr/).

![logo automatants](ressources/logo_automatants.png)

## Formation JAX - CIFAR-10 avec CNN

![logo](./ressources/logo_jax.jpeg)

L'objectif de cette formation est de présenter une pipeline d'entraînement de modèle de deep learning (en l'occurence des CNN séquentiels) pour le jeu de données CIFAR-10 en utilisant le framework [**JAX**](https://github.com/google/jax). C'est un framework développé par Deep-Mind (Google) qui permet de construire des modèles de machine learning de manière performante (compilation XLA) et plus flexible que son homologue Tensorflow, utilisant un framework presque entièrement basé sur les `nd.array` de numpy (mais stocké sur le GPU, ou TPU si disponible). Il fournit également des utilitaires inédits pour le calcul de gradient (per example, backward **et** forward) ainsi qu'un meilleur système de seed (pour la reproductibilité) et un outil pour batcher des opérations compliquées automatiquement et efficacement.

Lien de la doc: <https://jax.readthedocs.io/en/latest/index.html>.

L'objectif est de coder soi-même sa propre pipeline d'entraînement pendant la formation en réécrivant une version de `cnn_cifar10.py` en utilisant les fonctions utilitaires de `utils/`. Ensuite, il faudra améliorer le modèle notamment en limitant l'overfitting !

### Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) est un jeu de données de 60000 images de taille 32x32 RGB labelisées selon 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). L'objectif est tout simplement de prédire le plus souvent possible la classe des images.

### Installation

Cf le README de `formation jax`.

### Quick start

Cf le README de `formation jax`.

## Formation Pytorch - MLP et CNN sur MNIST

L'objectif de la formation est de faire un MLP et un CNN sur [Pytorch](https://pytorch.org/) pour de la classification sur le jeu de données MNIST.

![pytorch](./ressources/logo_pytorch.jpeg)
