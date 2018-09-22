---
layout: slide
title: 4) Biais, variance et gradient stochastique
description: none
transition: slide
permalink: /4-biais-variance-gradient-stochastique
theme: white
---

<section data-markdown>
<div class=centerbox>
<p class=top> Biais, variance et gradient stochastique</p>
<p style ='font-size:28px;'> avec scikit-learn</p>
</div>
</section>

<section>
<div style='float:right;'>
    <h1>Questions ?</h1>

    <div data-markdown>
    <img src=/assets/04/questions_04.gif>
    </div>
</div>

<div data-markdown>
# Cours précédent
* Régression logistique
* Approche statistique ou approche Machine Learning
* Métriques de classification:
    * TP, TN, FP, FN
    * Matrice de confusion
    * ROC-AUC
* One hot encoding
</div>
</section>


<section>
<div style='float:right; width:40%'>
    <div data-markdown>
    ## Lab:
    </div>
</div>

<div style='float:left; width:60%'>
<div data-markdown>
# Aujourd'hui

* Scikit-learn
* Décomposition Biais - Variance
* Gradient Stochastique; Stochastic Gradient Descent
* Outliers, detection and impact
* Skewness: Box cox and Kurtosis

</div>
</div>
</section>

<section>
<div style='float:right;'>
    <div data-markdown>
    <img src=/assets/04/logo-scikit.png style='width:300px;'>

    * [scikit-learn.org](http://scikit-learn.org/stable/)
    * [Eco-système](http://scikit-learn.org/stable/related_projects.html)

    </div>
</div>

<div style='float:left; width:60%'>
<div data-markdown>

# Scikit-learn

Libraries open source pour le machine learning créée 2010 dans le cadre du summer of code de Google
* Librairie basée sur numpy, scipy
* Contributeurs principaux :  Olivier Grisel, Andreas Muller, Gael Varoquaux, Jake Vanderplas => [lien github](github.com)
* Un projet soutenu par
    * [INRIA](http://www.inria.fr),
    * [Telecom ParisTech ](http://www.telecom-paristech.fr/),
    * NYU
* Largement utilisé dans la communauté ML
</div>
</div>
</section>

<section>
<div style='float:right;'>
    <div data-markdown>
    <img src=/assets/04/logo-scikit.png style='width:300px;'>
    </div>
</div>
<div data-markdown>

# Algorithmes et modèles:
Grand choix d'algorithmes et de modèles

## Supervisé
* Classification:
    * SVM, nearest neighbors, random forest, XGBoost, AdaBoost, ...

* Regression:
    * SVM, SGD, ridge et Lasso, regression lineaire, naive bayes, ...

## Non supervisé
* Clustering: Grouper des échantillons *similaires* ou *proches*
    * k-Means, spectral clustering, mean-shift, ...

* Reduction de dimension: Réduire le nombre de variables
    * Applications: Visualization, Performance
    * PCA, feature selection, non-negative matrix factorization.
</div>
</section>

<section>
<div style='float:right;'>
    <div data-markdown>
    <img src=/assets/04/logo-scikit.png style='width:300px;'>
    </div>
</div>
<div style='float:left; width:70%'>
<div data-markdown>

# Scikit learn

Mais aussi :
* Sélectionner les modèles: comparer, valider et choisir les paramètres et modèles
    * But: trouver les paramètres qui offrent les meilleurs performances
    * Modules: grid search, cross validation, metrics.

* Pre-processing: Transformation des variables.
    * But: Transformer les variables brutes pour améliorer leur pertinence et les numériser.
    * Modules: preprocessing, feature extraction.

* Documentation très complete avec de nombreux exemples
* Capable de traiter différents types de données: images, textes, données numeriques

</div>
</div>
</section>

<section>
<div style='float:right;'>
    <div data-markdown>
    <img src=/assets/04/logo-scikit.png style='width:300px;'>
    </div>
</div>
<div data-markdown>
# scikit-learn API
Simple et cohérente

1. Instancier un modèle, par exemple une regression linéaire:
    * ```from sklearn.Linear import LinearRegression```
    * ```mdl = LinearRegression( meta-params, loss function, ...)```
2. Entrainer le modèle
    * ```mdl.fit(X, y)```
3. Obtenir des prédictions sur de nouvelles données
    * ```y_hat = mdl.predict(Nouveaux échantillons}```.
    * ```y_hat = mdl.predict_proba(Nouveaux échantillons)```
4. Obtenir un score
    * ```mdl.score(X,y)```

</div>
</section>

<section data-markdown>
# paramétrer les modèles

* les meta-paramètres du modèle: \\(\alpha, \epsilon, \beta, \gamma, \cdots \\)
* la regularisation: *penalty, l1, l2*
* la fonction de cout: *loss*
* la gestion des itérations: *max_iter, n_iter*
* data pre-processing: *normalize, shuffle, valeurs manquantes*

[![Lin reg params](/assets/04/scikit_linear_reg_params.png)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
[![SGD reg params](/assets/04/scikit_sgd_reg_params.png)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)
</section>

<section data-markdown>
# Demo

[linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
II: Biais - Variance
</p>
</div>
</section>

<section>
<div style='float:right;'>
    <div data-markdown>
    <img src=assets/04/bias-variance-targets.png style='width:500px;'>
    </div>
</div>
<div style='float:left; width:50%'>
<div data-markdown>

# Décomposition Biais - Variance
L'erreur de prédiction peut etre décomposée en 2 termes

$$
\text{Erreur totale} = \text{Erreur du biais} + \text{Erreur de la variance}
$$

**Biais**: la différence entre les predictions du modele et la valeur cible. Le biais mesure la performance du modèle, la distance entre les predictions et les valeurs cibles.

* **Underfitting**: Un biais important indique que le modele n'arrive pas  à comprendre les données qui lui sont fournies

**Variance**: Il s'agit là de la variabilité des prédictions entre différentes *réalisations* du modèle pour un échantillon donné.

* **Overfitting**: Une forte erreur de variance indique que le modèle ne pourra pas extrapoler ses prédictions sur des nouvelles données.
</div>
</div>
</section>


<section >
<div style='float:left; width:50%'>
<div data-markdown>

# Décomposition Bias Variance
L'erreur quadratique est définie par:

$$ \operatorname{MSE}( \hat{y} ) = \frac {1}{n} \sum_{i=1}^{n}( \hat{y_i}-y_i )^2 = \mathbb{E} \big[ (\hat{y} - y)^2   \big] $$


Et on peut réécrire cette équation de la façon suivante:

$$ \operatorname{MSE}( \hat{y} )   =\mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right] + \left(\mathbb{E}(\hat{y})-y\right)^2 $$

Soit
$$ \operatorname{MSE}( \hat{y} )= \operatorname{Var}(\hat{y})+ \operatorname{Bias}(\hat{y},y)^2 $$

avec

$$ \operatorname{Var}(\hat{y}) =  \mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right] $$
$$ \operatorname{Bias}(\hat{y},y)  = \mathbb{E}(\hat{y})-y  $$

</div>
</div>
</section>

<section>
<div data-markdown>
# Le calcul
</div>
$$
\begin{align} \mathbb{E}((\hat{y}-y)^2)&=
 \mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})+\mathbb{E}(\hat{y})-y\right)^2\right]
\\ & =
\mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2 +2\left((\hat{y}-\mathbb{E}(\hat{y}))(\mathbb{E}(\hat{y})-y)\right)+\left( \mathbb{E}(\hat{y})-y \right)^2\right]
\\ & = \mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right]+2\mathbb{E}\Big[(\hat{y}-\mathbb{E}(\hat{y}))(\overbrace{\mathbb{E}(\hat{y})-y}^{\begin{smallmatrix} \text{This is} \\  \text{a constant,} \\ \text{so it can be} \\  \text{pulled out.} \end{smallmatrix}}) \,\Big] + \mathbb{E}\Big[\,\overbrace{\left(\mathbb{E}(\hat{y})-y\right)^2}^{\begin{smallmatrix} \text{This is a} \\  \text{constant, so its} \\  \text{expected value} \\  \text{is itself.} \end{smallmatrix}}\,\Big]
\\ & = \mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right]+2\underbrace{\mathbb{E}(\hat{y}-\mathbb{E}(\hat{y}))}_{=\mathbb{E}(\hat{y})-\mathbb{E}(\hat{y})=0}(\mathbb{E}(\hat{y})-y)+\left(\mathbb{E}(\hat{y})-y\right)^2
\\ & = \mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right]+\left(\mathbb{E}(\hat{y})-y\right)^2
\\ & = \operatorname{Var}(\hat{y})+ \operatorname{Bias}(\hat{y},y)^2
\end{align}

$$
</section>

<section data-markdown>
# En résumé
![underfit - overfit](/assets/04/underfit_overfit.png)
</section>

<section data-markdown>
# En résumé
Pour réduire l'erreur quadratique, il faut minimiser à la fois le **biais** et la **variance**.

* Biais important <-> sous estimation - Underfitting
    * le modele n'est pas bon
    * On obtient de mauvais score

* Variance importante <-> Overfitting
    * Le modele est trop **sensible** au training dataset
    * pouvoir d'extrapolation faible, mauvaises performances sur des nouvelles données.

Mais comment détecter l'overfit ?

</section>

<section data-markdown>
# split: train vs test dataset

* On entraine le modele sur un **dataset de training**
* Mais le but est d'avoir un modele capable de traiter des **données nouvelles**, données qu'il n'a pas vu auparavant
* et on veut surtout éviter **l'Overfit**: *Le modele est trop **sensible** au training dataset*

Donc on va mettre de coté  une partie des données comme données nouvelles: dataset de test.

Typiquement : une répartition  80/20 ou 70/30

<img src=/assets/04/split_train_test.png style='width: 500px;  margin:auto; ' >

Et c'est en évaluant le modele sur les données de test que l'on va pouvoir détecter l'overfit

</section>

<section data-markdown>
# train - test split

> Demo sur iris dataset

</section>

<section>
<div style='float:right;'>
    <div data-markdown>
    [scikit example 1](http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)

    [scikit example 2](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py)
    </div>
</div>
<div style='float:left; width:70%'>
<div data-markdown>




# Comment détecter l'overfit ?
### Courbes d'apprentissages - Learning curves



* On met de coté un set d'apprentissage (20% des données)
* on entraine le modele sur un nombre croissant d'echantillons (10%, 20%, ...)
* Pour chaque réalisation on calcule
    * l'erreur sur le set d'apprentissage
    * l'erreur sur le set de test

* En accroissant le set d'apprentissage, le modele a de plus en plus d'info, le modele apprends le set d'apprentissage. On espere que ca va lui permettre de traiter aussi les données sur le set de test.

Ce que l'on observe:
* avec un set d'apprentissage petit, les 2 erreurs sont grandes
* avec plus de données, l'erreur d'apprentissage décroit
    * si l'erreur sur le set de test ne décroit pas: **overfit**!

Si l'erreur sur le set de test ne décroit pas, alors cela veut dire que le modele n'est pas capable d'extrapoler sur des nouvelles données

Note: si l'erreur ne décroit pas sur le set de training en premier lieu, alors cela veut dire que le modele est mauvais, que l'erreur de biais est forte.

</div>
</div>
</section>


<section data-markdown>
# Illustration

### learning curve

* underfit
* overfit


</section>
<section data-markdown>
# Detecter l'overfit -

Demo sur ames housing avec SGD

</section>


<section data-markdown>
# Validation croisée
Si on a peu de données, le split train - test *gaspille* des données pour le test. Données qui pourraient etre utile pour l'apprentissage du modele.

=> on va alterner le découpage train - test, 80% - 20%,

C'est la validation croisée et plus particulièrement **K-FOLD cross validation**

</section>

<section data-markdown>
# K-fold cross-validation

<img src=/assets/04/k-fold-cross-validation.png style='width: 600px;  margin:auto; float:right;' >

1. Mélanger le dataset
2. Puis découper le dataset en K (5) parties
3. Faire K (5) experiences:
    * apprentissage sur 1,2,3,4 et evaluation sur 5
    * apprentissage sur 1,2,3,**5** et evaluation sur **4**
    * ...
    * apprentissage sur 2,3,4,5 et evaluation sur **1**

La moyenne des scores obtenus ainsi est plus robuste qu'un score obtenu sur un unique découpage.

[K-fold cross validation - scikit-learn](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_validation)

</section>
<section data-markdown>
# Autres méthodes de validation croisée

* **Stratified K-Fold**: (classification) Chaque subset conserve la distribution des classes. Utile lorsque la repartition des classes est déséquilibrée.
* **Leave one out**:  Chaque échantillon est utilisé à son tour comme echantillon de test. Tous les autres sont laissé dans le set d'apprentissage.
* **Shuffle cross validation**: Decoupage aléatoire avec remise en place. rien n'oblige à fixer le découpage au début.


### scikit-learn
* [cross_val_score](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_digits.html#example-exercises-plot-cv-digits-py)
* [cross validation](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_validation)


</section>
<section data-markdown>
# K-fold cross validation
Exercise

On the diabetes dataset, find the optimal regularization parameter alpha.

Bonus: How much can you trust the selection of alpha?

http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold

http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#example-exercises-plot-cv-diabetes-py

</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
III: Gradient Stochastique
</p>
<p class=mitop>
Stochastic Gradient Descent
</p>
</div>
</section>

<section data-markdown>
# Gradient Stochastique

**1951!** Robbins - Monroe: A Stochastic approximation Method

<img src=/assets/04/robins_monroe_1951.png style='width: 500px; margin: auto; float:right; '>

The general idea is to aproximate an unknown function through iterations of unbiased estimates of the function's gradient. Knowing that the expectation of the gradient estimates equal the gradient.

Soit une function \\(  f \\) que l'on souhaite approximer.
Sous certaines conditions sur  \\(  \alpha \\) et \\(\hat{\nabla} f\\) (gradient de f), alors \\(  \\)

$$  {\bf w}_{t+1} = {\bf w}_t - \alpha_t \hat{\nabla} f({\bf x}_t) $$
$$ {\bf w}_t -> f  $$

</section>

<section data-markdown>

Dans notre contexte, la fonction \\(f\\) est une fonction de régression linéaire d'ordre N avec les coefficients \\(w_k \text{with} k \in [0..N]\\)

$$f(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_N x_N $$

On veut donc minimiser l'erreur

$$ e = y - f(X) = t - W^T X $$

Cette equation admet une solution exacte que l'on a vu precedemment

$$\hat{W} = ()() $$

Au lieu de calculer la solution directement on va l'estimer par la methode du gradient:

</section>

<section data-markdown>
# Methode du gradient

Equation

Neanmoins cela necessite de calculer le gradient sur tous les echantillons disponibles a la fois. Pour un dataset grand, c'est couteux et long.

Donc on va utiliser le fait que sous certaines conditions

le gradient peut etre estimé par la moyenne des

The SGD algorithm is low on computation, has good convergence behavior and is applicable to many different situations through its many available variants. The SGD is implemented in a wide variety of languages (python, R, Java, scala, matlab, ...) and platforms (weka, rapidminer, ...) as well as online services big (Amazon ML, Google, Azure) and small (Dataiku, ....)

The idea of iterative stochastic approximation Robbins and Monro in 1951 in a seminal paper titled A Stochastic approximation Method

The literature related to the SGD algorithm is abundant. With the resurgence of depp learning and neural networks, no less than 6000 academic papers about Stochastic Gradient Descent were published in 2016 according to google scholar (from around a few 100s in the early 2000s).



According to the Gauss-Markov theorem, the model fit by the Ordinary Least Squares (OLS) is the least biased estimator of all possible estimators. In other words, it fits the data it has seen better than all possible models.

Calculer la solution a partir de l'equation ci dessus est couteuse en calcul

On va donc approximer la solution de facon iterative

* on choisit un vercteor W_0 pour initialiser l'algo
et a chaque iteration on corrige W par le gradient de la fonction de cout


see [raschka](https://sebastianraschka.com/faq/docs/closed-form-vs-gd.html)
The cost function J(⋅), the sum of squared errors (SSE), can be written as:

The magnitude and direction of the weight update is computed by taking a step in the opposite direction of the cost gradient

where η is the learning rate. The weights are then updated after each epoch via the following update rule:

=> learning rate

~[gradient as ball](/assets/04/gradient_ball.png)

see file:///Users/alexis/amcp/packt-B05028/B05028_07_draft.html

In case of very large datasets, using GD can be quite costly since we are only taking a single step for one pass over the training set – thus, the larger the training set, the slower our algorithm updates the weights and the longer it may take until it converges to the global cost minimum

</section>
<section data-markdown>

La dimenion stichastique consiste a actualiser le W non plus avec l'integralité du gradient sur toutes les donnees mais avec une estimation du gradient echantillon par echantillon.
Cela marche parce que dans notre contexte: E(gradient) = gradient

Instead, we update the weights after each training sample:

et on passe plusieurs fois sur le dataset en shufflant le dataset a chaque fois (epoch)

“stochastic” comes from the fact that the gradient based on a single training sample is a “stochastic approximation” of the “true” cost gradient

it has been shown that SGD almost surely converges to the global cost minimum if the cost function is convex (or pseudo-convex)


## Mini-Batch Gradient Descent (MB-GD)
milieu entre le GD et le SGD
On utilise K echantillons pour estimer le gradient a chaque iteration

Accroitre K entraine

* on converge plus vite et plus directement. moins de zig zag
* on converge moins profondement. le biais est plus important.


</section>

<section data-markdown>
# Outliers, detection and impact

</section>

<section data-markdown>
# Skewness: Box cox and Kurtosis

</section>

<section data-markdown>
# resources
https://towardsdatascience.com/predicting-housing-prices-using-advanced-regression-techniques-8dba539f9abe



* Data Split
    train, test, valid http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/18
    k-fold http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/20
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/21
    * stratified k-fold http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/22
</section>
