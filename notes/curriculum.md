Le cours sur 2 semaine est centré sur l'analyse prédictive supervisée et non supervisée.
L'environnemnt de travail repose sur les notebook jupyter, sur le language Python 3 et la librairie scikit-learn.
Les elements theoriques sont enseignés le matin et l'apres midi est consacrée aux travaux pratiques (LAB).
Chqaue journée debute par un recapitulatif des elements vus la veille pour permettre aux etudiants de revenir sur les points qui restent a eclaircir.
Les etudiants se verront proposés des lectures pour le lendemain.
Un projet final  basé sur une competition kaggle fera l'objet de presentations le dernier jour.
modèle bayésien naif

Projet, kaggle competition
Envt jupyter notebooks
Python 3, scikit learn, pandas
Office hours
Feedback
Homework ?

# Jour 1: Intro, data science regression lineaire
* AM:
    * Intro, et présentations
    * Présentation du curriculum, du projet (competition kaggle?), des attentes
    * Mise en place de l'envt de travail (jupyter notebook, slack), des outils (python, scikit-learn, ...),
    * Qu'est-ce que la data science?
        * Etapes d'un projet et finalités.
        * DS vs ML vs AI vs RL vs DL.
        * Supervisée vs Non Supervisée,
        * Regression vs classification.
    * Classic Datasets

* PM:
    * LAB: Application a des dataset simples
        * Regression lineaire,
        * Regression logistique,
        * ANOVA:
        * Conditions sur les donnees, cas d'application, p-value, R^2, intervalle de confiance, ...


# Jour 2: Model vs Algo
* AM:
    * Algo du maximum de vraisemblance
    * Importance de la linearité. Tests statistiques associés
    * Impact de la multi-collinearité
    * Régression polynomiale

* PM:
    * Presentation du projet final: compétition Kaggle.
    * LAB: On continue sur les differents type de régression

# Jour 3: Gradient stochastique

* AM:
    * Recap: l'analyse predictive supervisée: Modele ou Algo, minimisation d'une fonction de cout, score, predicteurs et outcome, ...
    * Algo du gradient stochastique, théorie, pratique et applications

* PM:
    * LAB: Gradient stochastique, convergence, parametrisation, visualization, scikit, batch mode, conditions


# Jour 4: Arbres et Forets
* AM:
    * Arbre de decision, forets aleatoires, XGBoost
    * Ensembling et Boosting
    * Score de classification: matrice de confusion, AUC, F1, False positive, ...

* PM:
    * LAB: Random forests et XGBoost

# Jour 5: SVM et overfitting
* AM:
    * Support Vector Machines, different Kernels
    * Overfitting, biais-variance tradeoff, L1 et L2 regularization, ridge et lasso
    * Learning curves: detecter et corriger l'overfitting

* PM:
    * LAB: SVM
    * LAB: Regularization


# Jour 6: Feature engineering et feature selection
* AM:
    * Naive Bayes
    * Feature engineering approche manuelle, brute ou approche bayesienne
    * Feature selection
    * Curse of dimensionality
    * Reduction de dimension

* PM:
    * LAB: feature engineering et selection

# Jour 7: Imbalanced datasets
* AM:
    * Le paradoxe de la precision
    * Methodes de sous-echantillonnage ou de sur-echantillonnage
    * SMOTE

* PM:
    * LAB: sur dataset caravan
    * Point projet

# Jour 8: Non-supervisé
* AM:
    * Kmeans, Nearest Neighbor, T-SNE, ...
* PM:
    * LAB
    * Point projet

# Jour 9: Series temporelles
* AM:
    * Exponential smoothing
    * Modeles linéaires AR, MA, ARIMA
    * Decomposition en trends, saisonnalité et résidus
    * Dynamic time warping
* PM:
    * LAB sur séries temporelles

# Jour 10: Neural Nets with Keras
* AM:
    * Intro au reseaux de neurones
    * Backpropagation, fonction d'activation, ...

* PM:
    * Présentation du projet


Manque
* Text
* Time series
* Neural Nets
* box plot
* missing values
* breiman: black box vs stats
* case studies: splice and image classification
* readings for the night
* Pandas
* Python



------------------------------------------
# Part I: Concepts and applications
------------------------------------------

# IA, DS, ML, DL, RL
Quelle difference ?

# Concepts
* le probleme: prediction, detection, ....
* les outils a disposition: cloud, librairies, python, R, SAS, matlab, excel
* le materiaux brut: les données disponibles que l'on va travailler, acces, confidentialité, frequence, format, ...
* les algos et modeles: regression lineaire, random forest, XGBoost, Neural nets, ...
* Les resultats: les scores obtenus, leur interpretation et leur actionabilité

# Context
A la base on a des variables d'entree, avec ou sans cible
- numerique, categorie,  text, mixes,
- transformer le tout en numerique

# Modele, meta-parametres, algorithme, fonction de cout et score
Arbre de decision ou lin reg

# Classification ou regression supervisée

variable a prédire est continue ou categorie
passer de continue a category avec cut ou qcut

# Non Supervisé
Necessite de definir une distance

- clustering: t-SNE, k-means, nearest neighbor
- topic modeling: base twitter
- reduction de dimension

# Exemple: régression lineaire sur height vs weight vs age
presentation regression lineaire, dataset, ...
les datasets: age vs weight

exemple titanic: classification

exemple Iris non supervisée

# Demo 1:
Cas d'etude classification
approche supervisée basée sur un dataset de verité, ground truth

approche non supervisée, peut on separer les classes sans a priori ?
application au titanic

# Etapes d'un projet d'analyse predictive
Aspects itératifs
- ETL
    - data exploration: outliers, missing values, correlation, ...
    - data transformation: normalization, gaussianization (Box Cox),
    - creation de nouvelles features
- modele(s)
    - selection, optimization, comparaison
    - evaluation
- Mise en production
    - frequence des mises a jours
    - streaming

# 10 fallacies of DS
https://towardsdatascience.com/the-ten-fallacies-of-data-science-9b2af78a1862

# c'est difficile
c'est pas gagné

# Hypotheses sur les donnees pour que les
- iid
- multi-colinearite

- contre exemple ?

# Notion de linearité centrale a l'analyse predictive

- linearite def
- certains modeles s'appliquent: SVM, SGD, Lin Reg, ...
- d'autres non => NN, RF,

# Feature engineering

# Feature selection

# Feature importance

black box model

------------------------------------------
# Part II: Methodology
------------------------------------------

# But: données nouvelles

Les donnees sont "completes"

But est de 1) predire sur des donnees nouvelles 2) selectionner le meilleur modele

Hypothese: les donnees nouvelles sont pareils que les donnees d'entrainement au sens statistique

# train, test, validation
On va selectionner le meilleur modele

- train multiple models
- test to see performance
- select best model
- apply to validation: as proxy for what happens in the real world.

# Cross validation

# Overfitting
- def
- comment detecter
- comment remedier: contraintes sur les parametres

# Application a un cas concret
Notion d'experimentation

1. dataset brut + un modele avec un set de parametres => un resultat
2. autres parametres
3. autres algos
4. nouvelles features => score 2

# Imbalanced dataset
accuracy is wrong
sub sampling
over sampling

# Traiter du texte

transformer le texte en numerique
word2vec, tf-idf

---------------------------------------------------------------------------
# Deep Learning et reseaux de neurones
---------------------------------------------------------------------------
# Intro
- 3 types: feed forward, RNN (LSTMs), CNN
- depth, width, activation function, ...
- quand ?

# Etude de cas: image, object detection, text

# Inception, ResNet, ...

---------------------------------------------------------------------------
# AWS
---------------------------------------------------------------------------

# ML

# SageMaker
# Rekognition
# Comprehend: Keyphrase Extraction, Sentiment Analysis, Entity Recognition, Topic Modeling, and Language Detection

---------------------------------------------------------------------------
# Workshop 1: AWS ML
---------------------------------------------------------------------------

SDK

---------------------------------------------------------------------------
# Workshop 2: AWS SageMaker
---------------------------------------------------------------------------
Avec S3 +
production endpoint

SDK
---------------------------------------------------------------------------
# Workshop 3: AWS Pipeline
---------------------------------------------------------------------------
Pipeline de scoring en temps réel avec image et texte
A partir d’un modèle prédictif, nous construirons un pipeline de prédiction en temps réel sur un jeux de données d’images en utilisant: Sagemaker, S3, Rekognition, Comprehend, AWS Lambda et Kinesis.
