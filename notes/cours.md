
# Generalités

* l'analyse prédictive? modele parametrique (plus simples) ou non parametrique (plus compliqués mais overfitting), erreur(s), loss function, modele, interpretabilité vs flexibilité, prediction vs inference

* Supervisée vs Non Supervisée: fit le modele aux donnees, ground truth, exemples concrets, cluster analysis, relations entre les variables, differents models, differents algo, mais toujours réduction d'une fonction de cout.

* applications: les plus sympa
https://www.datarobot.com/use-cases/

https://towardsdatascience.com/clustering-algorithms-for-customer-segmentation-af637c6830ac
https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
https://towardsdatascience.com/spotifys-this-is-playlists-the-ultimate-song-analysis-for-50-mainstream-artists-c569e41f8118
https://towardsdatascience.com/using-machine-learning-to-simulate-world-cup-matches-959e24d0731
https://towardsdatascience.com/an-examination-of-international-cuisines-through-unsupervised-learning-93c8b56d1ea0


* Classic Datasets: rapidement passer sur les dataset: titanic, iris, housing, income, ...

# Analyse predictive concepts et pratique
* Regression vs classification. variables continue ou quantitative (categories). salaire vs comportement, categorie d'appartenance (arbres) vs temperature, ...,

* Métrique et performance
mesurer la qualité de la prediction avec une metrique: RMSE pour regression, precision pour classification (et beaucoup d'autres), appliquée sur des donnees nouvelles, training vs test dataset,

* bias-variance trade off

# Python et librairies
numpy, pandas, 


# Régression
simple mais interpretable
* Regression lineaire: hypotheses, interpretation,
* Regression logistique
* ANOVA
* p-value, R^2, intervalle de confiance, autres stats
odds ratio
compound : noyade et chaleur...
adj R^2
p 92 potential problems
collinearite: VIF

# Linearité
* Importance de la linearité. Tests statistiques associés
* Impact de la multi-collinearité: sur données artificielles
* Régression polynomiale


# cross validation
K-fold, bootstrap,
bagging: multiple copies of original datasetn iusing the bootstrap

# SGD
convergence, parametrisation, visualization, scikit, batch mode, conditions

# overfitting & regularisation
lasso, ridge
ch 6

Overfitting, biais-variance tradeoff, L1 et L2 regularization, ridge et lasso
Learning curves: detecter et corriger l'overfitting

# feature selection
curse of dimension, dimension reduction
forward, backward, ...

# Tree based models: chapt 8
decision trees, Ensembling
random forest = bagging + bootstrapping the features
different loss, gini, ..
out of bag
boosting

# classification metrics
confusion matrix, ROC, AUC, TP, recall, ...

# SVM
ch 9
rel to logistic regression

# Naive Bayes ?

# data processing
outlier
missing data
leakage

# Imbalanced datasets
Over, under sampling
SMOTE


# Series temporelles
* Exponential smoothing
* Modeles linéaires AR, MA, ARIMA
* Decomposition en trends, saisonnalité et résidus
* Dynamic time warping


# Project management
----------------------------------------------------
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

    * Etapes d'un projet et finalités.
    * DS vs ML vs AI vs RL vs DL.
