---
layout: slide
title: 4) Gradient, Stochastique,  Biais, Variance
description: none
transition: slide
permalink: /4bis-biais-variance-gradient-stochastique
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

<section data-markdown>
<div class=centerbox>
<p class=top>
I: Rapide Rappel Scikit-learn
</p>
</div>
</section>


<section >
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

<section>
<div style='float:right; width:40%'>
    <div data-markdown>
    ## Lab: Titanic la suite
    </div>
</div>

<div style='float:left; width:60%'>
<div data-markdown>
# Aujourd'hui

* Gradient et Gradient Stochastique;
* Décomposition Biais - Variance

</div>
</div>
</section>

<section>
<div style='float:right; width:35%;  '>
    <div data-markdown>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:50%;  '>
    <div data-markdown>
# Methode du gradient

Soit une function \\(  f \\) dont on souhaite trouver le minimum.

Pour \\(  \alpha \\)  assez petit et si f est dérivable

alors

$$  {\bf w}_{t+1} = {\bf w}_t - \alpha_t \hat{\nabla} f({\bf w}_t) $$

\\( {\bf w}_t  \\) converge vers le minima de \\(f\\)

On a

$$ f(\mathbf {w}\_{0})\geq f(\mathbf {w}\_{1}) \geq f(\mathbf {w}\_{2})\geq \cdots , $$


=> *fastest  decrease obtained for the direction of the negative gradient of f*
    </div>
</div>
</section>



<section data-markdown>
<img src=/assets/04/700px-Gradient_descent.svg.png width=350px>
<img src=/assets/04/700px-Gradient_ascent_contour.png width=350px>
<img src=/assets/04/Gradient_ascent_surface.png width=350px>

</section>



<section data-markdown>
# En python

On veut trouver le minimum de la fonction  \\( f(x) =   \\)

    cur_x       = 6 # The algorithm starts at x=6
    gamma       = 0.01 # step size multiplier
    max_iters   = 10000 # maximum number of iterations
    iters       = 0 #iteration counter
    precision   = 0.00001
    previous_step_size = 1

    # dérivée de la fonction à minimiser

    fct = lambda x: 4 * x**3 - 9 * x**2
    x = []
    while (previous_step_size > precision) & (iters < max_iters):
        x.append(cur_x)
        prev_x = cur_x
        cur_x -= gamma * fct(prev_x)
        print(cur_x, previous_step_size)
        previous_step_size = abs(cur_x - prev_x)
        iters+=1

    print("Le minimum est {:.4f}", cur_x)

    print("En x = {:.4f}, la valeur de la fonction est {:.4f}  ".format(cur_x, fct(cur_x)) )


</section>

<section data-markdown>
# Stochastic Gradient

The general idea is to aproximate an unknown function through iterations of **unbiased estimates of the function's gradient.**

Knowing that the expectation of the gradient estimates equal the gradient.

On va prendre les erreurs successives entre les vraies valeurs et leur estimée comme estimation du gradient!

Cela marche parce que dans notre contexte: E(gradient) = gradient

<img src=/assets/04/robins_monroe_1951.png style='width: 500px; margin: auto; float:right; '>

</section>


<section data-markdown>
# Fonction de cout, forme générale

On cherche a minimiser la fonction \\( f(x) = w^T x + b   \\)

$$ E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w) $$

where

* \\(L\\) is a loss function that measures model (mis)fit
* \\(R\\) is a regularization term  that penalizes model complexity
* \\(\alpha\\) is a non-negative hyperparameter.

En fonction du choix de \\(L\\) et de \\(R\\) on obtient différent algorithmes

=> http://scikit-learn.org/stable/modules/sgd.html

</section>

<section data-markdown>
# Gradient stochastique
<img src=/assets/04/gradient_ball.png height=300px>

* low on computation,
* has good convergence behavior
* The SGD is implemented in a wide variety of languages (python, R, Java, scala, matlab, ...) and platforms (weka, rapidminer, ...) as well as online services big (Amazon ML, Google, Azure) and small (Dataiku, ....)


The literature related to the SGD algorithm is abundant. With the resurgence of deep learning and neural networks, no less than 6000 academic papers about Stochastic Gradient Descent were published in 2016 according to google scholar (from around a few 100s in the early 2000s).


</section>


<section data-markdown>
# SGD

* learning rate
    * tuning
    * fixe
    * adaptif

* epoch and shuffling

## Mini-Batch Gradient Descent (MB-GD)
milieu entre le GD et le SGD
On utilise K echantillons pour estimer le gradient a chaque iteration

Accroitre K entraine

* on converge plus vite et plus directement. moins de zig zag
* on converge moins profondement. le biais est plus important.

</section>


<section data-markdown>



Instead, we update the weights after each training sample:

et on passe plusieurs fois sur le dataset en shufflant le dataset a chaque fois (epoch)

“stochastic” comes from the fact that the gradient based on a single training sample is a “stochastic approximation” of the “true” cost gradient

it has been shown that SGD almost surely converges to the global cost minimum if the cost function is convex (or pseudo-convex)

</section>


<section data-markdown>
<div class=centerbox>
<p class=top>
III: Biais - Variance
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

La variance mesure la sensibilité du modèle aux données d'apprentissages

* **Overfitting**: Une forte erreur de variance indique que le modèle ne pourra pas extrapoler ses prédictions sur des nouvelles données.
</div>
</div>
</section>


<section >
<div style='float:right; width : 45%;'>
    <div data-markdown>

* **Biais**: L'espérance de  l'erreur de prédiction

* **Variance**: Variance des prédictions.

    </div>
</div>

<div style='float:left; width:45%'>
<div data-markdown>

# Décomposition Biais - Variance
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

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Underftiing


* Ajouter des predicteurs
* Rendre le modele plus complexe
* Attenuer la regularisation.
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Overfitting

* Reduire le nombre de predicteurs
* Utiliser plus de données d'apprentissage
* Accroitre la reguilarisation
* Moyenner plusieurs modèles

    </div>
</div>
</section>



<section data-markdown>
<div class=centerbox>
<p class=top>
IV: K-fold cross validation
</p>
</div>
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
# Overfit - Biais - Variance SGD

Demo sur cars avec SGD

</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
III: Gradient, Stochastique
</p>
<p class=mitop>
Stochastic Gradient Descent
</p>
</div>
</section>



<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

    </div>
</div>
</section>