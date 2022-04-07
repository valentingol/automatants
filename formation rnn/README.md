# Formation Réseaux de Neurones Récurrents (RNN)

Valentin GOLDITE - 08/04/2022

![alt text](ressources/lstm-vs-gru.png)

## Introduction

Les réseaux de neurones récurrents sont une famille de réseaux de neurones un peu particulier. Les réseaux de neurones classiques prennent en entrée un batch de données qui sont traitées en parralèlle et séparément. En revanche les réseaux de neurones récurrents prennent en entrée un batch de **séquences** de données et chaque séquence est traitée en prenant les données successivement et les sorties précédentes interviennent dans le calcul de la sortie actuelle. Ainsi, ils sont en quelque sorte une généralisation des multi layers perceptrons car ils permettent d'inclure un aspect "séquentiel" (="temporel" le plus souvent) dans le calcul des prédictions.

## Installation

Clonez le repertoire dans un dossier local:

```script
git clone git@github.com:valentingol/automatants.git
```

Créez un nouvel environment virtuel avec `virtualenv`:

```script
python3 -m venv <path to env>
source <path to env>/bin/activate
```

Ou avec `virtualenvwrapper`:

```script
mkvirtualenv <name of env>
```

Puis installez les packages requis avec `pip`:

```script
pip install -r requirements.txt
```

Vous pouvez à présent faire touner le code pour entraîner un LSTM avec tensorflow:

```script
python train_lstm.py
```
