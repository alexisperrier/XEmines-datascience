---
layout: slide
title: Biais, variance et gradient stochastique
description: none
transition: slide
permalink: /04-biais-variance-gradient-stochastique
theme: white
---

<section>
<div style='display: table; height: 100px; width: 80%; text-align: center; border: 1px solid   #ccc; margin:auto; margin-top: 50px; box-shadow: 5px 10px 8px #888888; '>
<div  style ='margin: 100px 0 100px 0 ;'>
<p style ='font-size:44px;'> Biais, variance et gradient stochastique</p>
<p style ='font-size:28px;'> avec scikit-learn</p>
</div>
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
* Metrique de classification:
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

<div style='float:left; width:40%'>
<div data-markdown>
# Aujourd'hui

* Scikit-learn
* Biais - Variance
* Stochastic Gradient Descent
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

<div data-markdown>

# Scikit-learn

* Open source, 2010
* Basé sur numpy, scipy, …
* Olivier Grisel, Andreas Muller, Gael Varoquaux, Jake Vanderplas
* [INRIA](http://www.inria.fr), [Telecom ParisTech ](http://www.telecom-paristech.fr/), NYU

### Algorithmes et modèles:
* Classification: Identifying to which category an object belongs to.
    * Applications: Spam detection, Image recognition.
    * Algorithms: SVM, nearest neighbors, random forest, ...

* Regression: Predicting a continuous-valued attribute associated with an object.
    * Applications: Drug response, Stock prices.
    *  Algorithms: SVR, ridge regression, Lasso, ...

* Clustering: Automatic grouping of similar objects into sets.
    * Applications: Customer segmentation, Grouping experiment outcomes
    * Algorithms: k-Means, spectral clustering, mean-shift, ...
* Dimensionality reduction: Reducing the number of random variables to consider.
    * Applications: Visualization, Increased efficiency
    * Algorithms: PCA, feature selection, non-negative matrix factorization.
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
<div data-markdown>

# suite
* Model selection: Comparing, validating and choosing parameters and models.
    * Goal: Improved accuracy via parameter tuning
    * Modules: grid search, cross validation, metrics.

* Preprocessing: Feature extraction and normalization.
    * Application: Transforming input data such as text for use with machine learning algorithms.
    * Modules: preprocessing, feature extraction.

* Fantastic documentation, plein d'exemples
* Input: Images, text, numerique, ...

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
Could not be simpler

1. Instancier un modèle
Par exemple une regression linéaire:
    * ```mdl = linear_model.LinearRegression( meta-params, loss)```
2. Train le modèle
    * ```mdl.fit(X, y)```
3. Faire des predictions sur de nouvelles données
    * ```y_hat = mdl.predict(Some New Data)```.
    * ```y_hat = mdl.predict_proba(Some New Data)```
4. Obtenir un score
    * ```mdl.score(X,y)```

</div>
</section>

<section data-markdown>
# scikit-learn model input

* les meta parametres du modele: \\(\alpha, \epsilon, \beta, \cdots \\) *learning_rate*
* la regularization: *penalty, l1, l2*
* la fonction de cout: *loss*
* gestion des iterations: *max_iter, n_iter*
* data processing: *normalize, shuffle*

[![Lin reg params](/assets/04/scikit_linear_reg_params.png)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
[![SGD reg params](/assets/04/scikit_sgd_reg_params.png)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)
</section>

<section data-markdown>
# Demo

[linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

</section>

<section data-markdown>
# Part II Biais - Variance

</section>

<section data-markdown>
# Bias Variance decomposition
The prediction error of your model can be decomposed in 2 terms:

* The bias
* The Variance

Total Error = Bias Error + Variance Error

</section>

<section data-markdown>
# Bias Variance decomposition

**Error due to Bias**: Error due to bias is taken as the difference between the average prediction of our model and the correct value which we are trying to predict.

Bias measures how far off in general your models' predictions are from the correct value.

**Error due to Variance**: The error due to variance is taken as the variability of a model prediction for a given data point.

The variance is how much the predictions for a given point vary between different realizations of the model.

</section>

<section data-markdown>
# Bias Variance decomposition
![Bias-variance](assets/04/bias-variance-targets.png)
</section>

<section data-markdown>
# Bias Variance and overfitting
![Bias-variance](assets/04/bias_Variance_Under_overfitting.png)
</section>

<section data-markdown>
# Bias Variance decomposition
L'erreur quadritique est définie par:

$$ \operatorname{MSE}( \hat{y} ) = \frac {1}{n} \sum_{i=1}^{n}( \hat{y_i}-y_i )^2  $$

que l'on peut réecrire de la façon suivante:

$$ \operatorname{MSE}( \hat{y} )= \mathbb{E} \big[ (\hat{y} - y)^2   \big]  =\mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right] + \left(\mathbb{E}(\hat{y})-y\right)^2 $$

Soit
$$ \operatorname{MSE}( \hat{y} )= \operatorname{Var}(\hat{y})+ \operatorname{Bias}(\hat{y},y)^2 $$

où

* \\( \operatorname{Var}(\hat{y}) =  \mathbb{E}\left[\left(\hat{y}-\mathbb{E}(\hat{y})\right)^2\right] \\)
* \\( \operatorname{Bias}(\hat{y},y)  = \mathbb{E}(\hat{y})-y  \\)
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
    * L'estimateur, le modele est pas bon
    * Mauvais score

* Variance importante <-> Overfitting
    * Le modele est trop **sensible** au training dataset
    * pouvoir d'extrapolation faible, peu de capacité à etre appliqué sur des nouvelles données.
    * comment détecter l'overfit ?

</section>

<section data-markdown>
# split: train vs test dataset

* On entraine le modele sur un **dataset de training**
* Mais le but est d'avoir un modele capable de traiter des **données nouvelles**, données qu'il n'a pas vu auparavant
* et on veut surtout éviter **l'Overfit**: *Le modele est trop **sensible** au training dataset*

Donc on va mettre de coté  une partie des données comme données nouvelles: dataset de test.
<img src=/assets/04/split_train_test.png style='width: 500px;  margin:auto; ' >

</section>

<section data-markdown>
# train - test split

> Demo sur iris dataset

</section>

<section data-markdown>
# Comment detecter l'overfit - learning curves

* set aside a test set
* train your model on increasing number of training samples
* for each model, calculate the training error and the test error

[scikit example 1](http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)

[scikit example 2](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py)

The learning curve traces the error on training and the error on the validation set as the sample sizes increases.

* For small sample sizes, the model does not have enough data to learn and both the training error and the testing error are high.

* As the sample size increases the learning rate decreases. The model is learning the data. And the testing error is also decreasing. The model becomes a better predictor.

* As you increase the sample size, there comes a point where the training error keeps on decreasing but the testing error stops decreasing and may instead starts increasing.

</section>

<section data-markdown>
# detecter l'overfit -

# TODO
Demo sur ames housing avec SGD

</section>

<section data-markdown>
# Que faire si on a peu de données

test - train split ne laisse pas assez de données pour un bon train du modele

=> validation croisée

k-fold

</section>

<section data-markdown>
# K-fold


# K-fold cross validation

Train, valid, test: You're wasting a lot of data

### [K-fold cross validation](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_validation)

![K-fold cross validation](assets/04/k-fold-cross-validation.png)


1. Split your data into train (80%) / test (20%), leave the test alone

2. Then further split your training set on K subsets (K = 4)
    * train on 1,2,3, validate on 4
    * train on 1,2,4, validate on 3
    * train on 1,3,4, validate on 2
    * train on 2,3,4, validate on 1

The average of the errors obtained on the validations subsets is a better estimation on the performance of your model than if you had just one validation set.
</section>
<section data-markdown>
# cross validation
Many other ways to do [cross validation](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_validation) in Scikit learn.

* Stratified K-Fold: preserving the percentage of samples for each class.
* Leave one out:  Each sample is used once as a test set (singleton) while the remaining samples form the training set.
* Shuffle Split: Random permutation cross-validation iterator. Yields indices.
* cross_val_score, see [this example](http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_digits.html#example-exercises-plot-cv-digits-py)

</section>
<section data-markdown>
# K-fold cross validation
Exercise

On the diabetes dataset, find the optimal regularization parameter alpha.

Bonus: How much can you trust the selection of alpha?

http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold

http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#example-exercises-plot-cv-diabetes-py

</section>

<section>
<div style='display: table; height: 100px; width: 80%; text-align: center; border: 1px solid   #ccc; margin:auto; margin-top: 50px; box-shadow: 5px 10px 8px #888888; '>
<div  style ='margin: 100px 0 100px 0 ;'>
<p style ='font-size:44px;'>Gradient Stochastique - SGD</p>
</div>
</div>
</section>


<section data-markdown>
# Gradient Stochastique aka SGD


The SGD algorithm is low on computation, has good convergence behavior and is applicable to many different situations through its many available variants. The SGD is implemented in a wide variety of languages (python, R, Java, scala, matlab, ...) and platforms (weka, rapidminer, ...) as well as online services big (Amazon ML, Google, Azure) and small (Dataiku, ....)

 The idea of iterative stochastic approximation Robbins and Monro in 1951 in a seminal paper titled A Stochastic approximation Method

The literature related to the SGD algorithm is abundant. With the resurgence of depp learning and neural networks, no less than 6000 academic papers about Stochastic Gradient Descent were published in 2016 according to google scholar (from around a few 100s in the early 2000s).

<section data-markdown>
</section>
The general idea is to aproximate an unknown function through iterations of unbiased estimates of the function's gradient. Knowing that the expectation of the gradient estimates equal the gradient.


Soit une function \\(f, {\bf w}_{t+1} = {\bf w}_t - \alpha_t \hat{\nabla} f({\bf x}_t) \\) given some condition on \\(\alpha_t  \text{and} \hat{\nabla} f\\), alors \\(w_t \text{converges vers} f\\).

Dans notre contexte, la fonction \\(f\\) est une fonction de régression linéaire d'ordre N avec les coefficients \\(w_k \text{with} k \in [0..N]\\)

$$f(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_N x_N $$

On veut donc minimiser l'erreur

$$ e = Y - f(X) = Y - W^T X $$

cette equation admet une solution exacte que l'on a vu precedemment

$$\hat{W} = ()() $$
On peut montrer que cette la solution de cette equation est l'esimateur le moins biaisé de tous le estimqteurs possibles! En d'autre termes il fit les donn´és d'entrainement du mieux possible.

https://towardsdatascience.com/predicting-housing-prices-using-advanced-regression-techniques-8dba539f9abe
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

* Data Split
    train, test, valid http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/18
    k-fold http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/20
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/21
    * stratified k-fold http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/22
</section>
