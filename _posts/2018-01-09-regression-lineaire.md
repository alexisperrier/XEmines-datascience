---
layout: slide
title: 2) Régression linéaire
description: none
transition: slide
permalink: /2-regression-lineaire
theme: white
---

<section data-markdown>
<div class=centerbox>
<p class=top>
Régression Linéaire
</p>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
    <img src=/assets/04/questions_04.gif>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Cours précédent

* Révisions de python
* Différence entre Data Science, Machine Learning et analyse prédictive
* Approche statistique vs approche machine learning
* Déroulement d'un projet de Data science
* Supervisée vs non-supervisée
* Régression vs Classification
* Anaconda, Python et Jupyter

    </div>
</div>
</section>


<section>
<div style='float:right; width:40%'>
    <div data-markdown>
    # Lab

    Régression linéaire sur le dataset *advertising*

    <img src=/assets/02/advertising.png>



    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Régression linéaire

* Régression linéaire
    * OLS, Moindres carrés
    * Modélisation
    * Univariable & multivariables
* Interprétation des résultats
    * Mean Square Error (MSE)
    * P-value, Interval de confiance, \\(R^2\\), \\(R^2_{adj}\\)
    * Confonders et multi-collinearité

* Hypothèses et vérification
    * Linéarité: Définition et tests

* Statsmodel

* Kaggle projet

</div>
</div>
</section>

<section data>

<div style='float:right; width:40%'>
    <div data-markdown>

## Regression: Qualitatif

La variable à prédire est **continue**

* Age, taille, poids,

* nombre d'appels, de clicks, volume de vente, consommation

* Température, Salaire, ...

* Probabilité d'une action

* Temps, délai, retard

</div>
</div>
<hr class='vline' />
<div style='float:left; width:60%'>
<div data-markdown>

## Classification: Quantitatif
La variable à prédire est **discrète**

### Cas binaire
* Achat, resiliation, click
* Survie, maladie, succes examen, admission,
* Positif ou negatif
* Spam, fraude

### Multi class - multinomiale
* Catégories, types (A,B,C),
* Positif, neutre ou negatif
* Espèces de plantes d'animaux, ...
* Pays, planetes

### Ordinale

* Notes, satisfaction, ranking

</div>
</div>
</section>


<section data>

<div style='float:right; width:40%'>
    <div data-markdown>

<img src=/assets/02/height_vs_weight.png style='float:right; width:400px;border:0px'>
</div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
<div data-markdown>

## Taille en fonction de l'age des enfants

On mesure la taille des enfants dans une ecole et leur age.
La taille croit avec l'age. On peut écrire

$$ \text{Taille} = f(\text{Age})  $$


## Regression univariable


On modélise cette fonction par une relation linéaire de la forme:

$$ \hat{\text{T}}\text{aille} = a * \text{Age} + b $$

où \\(\hat{\text{T}}\text{aille}\\) est la taille estimée.


On cherche à connaitre les paramètres \\((a,b)\\) qui donnent la meilleure approximation de la réalité entre la taille et l'age.

Pour trouver ces paramètres on utilise une méthode dite des **moindres carrés**  ou  **Ordinary Least-Squares (OLS)**.

</div>
</div>
</section>


<section data>
<div style='float:right; width:40%'>
    <div data-markdown>
Les résidus  \\( e_i  \\)  représentent une **distance** entre les vrais valeurs  \\( y_i  \\) et leur estimations \\( \hat{y_i}  \\). On veut réduire cette distance.

Pour cela on chercher à minimiser la somme des carrés des résidus (aussi appelé norme  \\( L^2  \\).)

$$  || y - \hat{y} ||^2 =  \sum_{i=0}^n (y_i - (a*x_i + b))^2   $$

<img src=/assets/02/Ordinary_Least_Squares_OLS.jpg>


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Regression linéaire
Nous avons \\( n  \\) échantillons:

* Une variable prédictrice \\( x = [x_1, ... , x_n]  \\)

* Et une variable cible \\( y = [y_1, ... , y_n]  \\)

On veut trouver les *meilleurs*  \\(a\\) et \\(b\\) pour lesquels

$$ \hat{y_i} = a * x_i +b  $$

l'erreur de prédiction \\( e_i  \\)  soit minimale:

$$ e_i = \vert  y_i - \hat{y_i} \vert  = \vert  y_i - (a * x_i +b)\vert  $$
</div>
</div>

</section>

<section data-markdown>
# Petit rappel des normes

### Norme quadratique  \\( L^2  \\)
$$  ||x|| = \sqrt{  x_1^2 + .... + x_n^2  } $$

### Norme \\( L^1  \\) ou norme en valeur absolue
$$  |x| =  |x_1| + .... + |x_n|  $$

### Norme infinie \\( L^\infty  \\)

$$  |x|\_{\infty} = max [ |x_1|, ... , |x_n| ] $$

</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

Cela donne 2 équations à 2 inconnues  dont la solution exacte est:

$$ \hat{\beta} =  (x^T.x)^{-1} x^T y   $$

avec

* \\( \hat{\beta} = \\{ a,b \\}^T \\)

*  \\( x = [x_1, ... , x_n]  \\)

*  \\( y = [y_1, ... , y_n]  \\)

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Fonction de cout

On a ce qu'on appelle une **fonction de cout** \\(L(a,b) \\):

$$ L(a,b)  = || y - \hat{y} ||^2 =  \sum_{i=0}^n [y_i - (a*x_i + b)]^2   $$

C'est fonction quadratique donc convexe.

Par conséquent pour trouver son minima, il faut trouver les valeurs de \\( a \\) et \\( b \\) qui annule la dérivée \\( 0 \\) .

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
On veut trouver les n+1 coefficients

$$ \beta = [\beta_0, \beta_1, ...., \beta_n] $$

qui minimisent la fonction de cout:

$$  L(\beta) = || y - X\beta ||  $$

La solution de cette équation est : $$ \hat{\beta} =  (X^T.X)^{-1} X^T y  $$
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>


# Regression multinomial

## plusieurs predicteurs

On a  maintenant \\(m\\) predicteurs et toujours \\(n\\)  échantillons.

Pour chaque échantillon, on a la modélisation suivante:
$$ \hat{y}_i  = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_m x_m $$
ou plus simplement
$$ \hat{y}  = \beta X  $$

avec

* \\( X = \[ (x_{i,j}) \]  \\) est une matrice de taille  \\(n\\) par \\(m\\)

* \\( y = [y_1, ... , y_n]  \\) vecteur de \\( n\\) échantillons



    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Python

A)

    X, y = make_regression(n_samples=N, n_features=M, noise =10)


B)

    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

C)

    yhat = X[:, 0]* beta[0] + X[:,1] * beta[1]


* ou si \\(M > 2\\):


    yhat = [0 for i in range(N)]

    for k in range(M):
        yhat += X[:, k]* beta[k]


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Regression linéaire


A) N samples avec M variables:

$$ y_i = \sum_k \beta_k * X[i,k]  + \sigma^2 $$

$$ y =  \beta * X  + \sigma^2 $$

B) Regression weights:

$$\quad \hat{\beta} = (X^T . X)^{-1} X^T y $$

C) Prédiction

$$ \hat{y}_i = \sum_k \beta_k * X[i,k]  + \sigma^2 $$



    </div>
</div>
</section>


<section data-markdown>
# Notebook - demo

02 Linear Regression Exact.ipynb

</section>


<section data-markdown>
# Metriques de scoring
## Erreur absolu (MAE) (L1)

Valeur absolue de la difference entre la prédiction et les vraies valeurs


$$  MAE = \sum\_{i=1}^n \| \hat{y\_i} - y\_i \| $$

        e = np.mean( np.abs(y - yhat) )

## Erreur quadratique (MSE) (L2)

$$  MSE = \sum_{i=1}^n (\hat{y_i} - y_i)^2 $$

        ```e = np.mean( (y - yhat)**2 )
        ```

</section>

<section>

<div style='float:right; width:45%;  '>
    <div data-markdown>

Sur un vrai dataset: **Mileage per gallon performances of various cars** disponible sur https://www.kaggle.com/uciml/autompg-dataset

A prédire:
* mpg: continuous

Les variables

* cylinders: multi-valued discrete
* displacement: continuous
* horsepower: continuous
* weight: continuous
* acceleration: continuous

On ne prends pas en compte:

* model year: multi-valued discrete
* origin: multi-valued discrete
* car name: string (unique for each instance)

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Régression linéaire avec Statsmodel

On va estimer les coefficients non plus directement mais avec la méthode OLS.



On aura plus d'information sur les coefficients de régression:

* leur importance relative
* leur fiabilité
* leur impact quantitatif

On utilise la librairie

* [Statsmodel](http://www.statsmodels.org/stable/index.html) librairie Python
pour une approche statistique de l'analyse de données.

* Intégrée avec pandas et numpy


    </div>
</div>
</section>


<section data-markdown>
# statsmodel


![](assets/02/statsmodel_functions.png)

</section>


<section data-markdown>

# Notebook python
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.read_csv('../data/autos_mpg.csv')
    lm = smf.ols(formula='mpg ~ cylinders + displacement + horsepower + weight + acceleration + origin ', data=df).fit()
    lm.summary()

</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Résultats
* **Dep. Variable**: La variable à prédire
* **Model**: Le modèle
* **Method**: La méthode utilisée
* **No. Observations**: Le nombre d'observations / échantillons
* **DF Residuals**: Degré de liberté des résidus = nombre d'échantillons - nombre de variables
* **DF Model**: Nombre de prédicteurs

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
<img src=/assets/02/02_linreg_autompg_01-left.png>
    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Goodness of fit

* **R-squared**: The [coefficient of determination](http://en.wikipedia.org/wiki/Coefficient_of_determination). A statistical measure of how well the regression line approximates the real data points

* **Adj. R-squared**: The above value adjusted based on the number of observations and the degrees-of-freedom of the residuals

* F-statistic: A measure how significant the fit is. The mean squared error of the model divided by the mean squared error of the residuals

* Prob (F-statistic): The probability that you would get the above statistic, given the null hypothesis that they are unrelated

* Log-likelihood: The log of the likelihood function.

* AIC: The [Akaike Information Criterion](http://en.wikipedia.org/wiki/Akaike_information_criterion). Adjusts the log-likelihood based on the number of observations and the complexity of the model.

* BIC: The [Bayesian Information Criterion](http://en.wikipedia.org/wiki/Bayesian_information_criterion). Similar to the AIC, but has a higher penalty for models with more parameters.</td>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
<img src=/assets/02/02_linreg_autompg_01-right.png>
    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
    # Définition

\\(R^2\\) est la proportion des variations de la variable cible qui est prédite grace aux prédicteurs.

On définit  \\(R^2\\) par

$$ R^{2} = 1-{SS\_{\text{res}} \over SS\_{\text{tot}}}   $$

On a

$$ 0 < R^2 < 1$$



    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
    # \\(R^2\\)

Soit la moyenne de la variable cible :

$$ \bar{y} = \frac{1}{n} \sum\_{i} y_{i} $$

et la somme des carrés de la variable cible centrée :

$$ SS\_{\text{tot}} = \sum\_{i} (y\_{i} - \bar{y} )^2 $$

La somme des carrés des résidus :
$$ SS\_{\text{res}} = \sum\_{i}e\_{i}^{2} = \sum\_{i} (y\_{i} - \hat{y}\_{i} )^2  $$


    </div>
</div>
</section>

<section data-markdown>

# R2 does not indicate whether:

* the independent variables are a cause of the changes in the dependent variable;
* omitted-variable bias exists;
* the correct regression was used;
* the most appropriate set of independent variables has been chosen;
* there is collinearity present in the data on the explanatory variables;
* the model might be improved by using transformed versions of the existing set of independent variables;
* there are enough data points to make a solid conclusion.

et surtout
* plus on ajoute de variable plus \\(R^2\\) augmente meme quand les variables ne sont pas vraiment significative.

</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
En accroissant le nombre de predicteurs, on accroit souvent R^2.

Mais \\(R^{2}_{adj}\\) compense la complexité du modele
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# \\(R^2_{adj}\\)

On ajuste pour prendre en compte la complexité du modele:

$$ R^{2}_{adj} = {1-(1-R^{2}){n-1 \over n-p-1}} $$

avec

* \\(p\\) le nombre de prédicteurs
* \\(n\\) le nombre d'échantillons

    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>


# Coefficients et p-value
La deuxième partie des résultats porte sur les coefficients et leur fiabilité.


* **coef**: La valeur estimée des coefficients
* **P &gt; |t|**: la probabilité que l'on observe cette estimation alors qu'en fait le coefficient est nulle (=0) .
* **[95.0% Conf. Interval]**: l'interval de confiance de l'estimation du coefficient.
* **std err**: l'erreur d'estimation
* **t-statistic**: une mesure de l'importance (significant) statistique de chaque coefficient.

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
<img src=/assets/02/02_linreg_autompg_02.png>
    </div>
</div>
</section>



<section>
<div style='float:right;width : 40%;'>
    <div data-markdown>

<img src=/assets/02/p_value.png height=250px >


The p-value represents the probability that the coefficient is actually zero

* Si \\( P_{value} > 0.05 \\) alors il y a plus de 5% de chance que l'hypothèse NULL soit vraie:=> **on ne peut pas la rejeter**.

* si \\( P_{value} < 0.05 \\) a lors il y a moins de 5% de chance pour que l'hypothèse NULL soit vraie: => **on  peut la rejeter**
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# P-value

On a 2 hypothèses:

1. [NULL] ce que l'on observe est du au hasard
2. [ALT] ce que l'on observe n'est pas du au hasard (il y a une relation)

La p-value est la probabilité que ce que l'on observe est du au hasard.

Si la p-value est faible, on rejete l'hypothèse NULL.

Ce qui ne veut pas dire que la valeur du coefficient est la  bonne. (ca serait trop simple) mais simplement que il y a bien une relation entre le predicteur et la variable cible.

</div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
<img src=/assets/02/xkcd_p_value.png style='width:250px'>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
<img src=/assets/02/xkcd_p_value_02.png style='width:300px'>
    </div>
</div>
</section>

<section data-markdown>

# Multinomiale

Que se passe t il quand on filtre certains predicteurs ?

</section>
<section data-markdown>
# Conditions sur les données
Pour qu'une régression linéaire soit possible et fiable, il faut que les données vérifient les conditions suivantes:

* **Linearite**: la relation entre les predicteurs et la cible est lineaire
    * On peut tester avec des scatter plots

* **Normality**: Les variables ont une distribution normale
    * test: [QQ plot](https://en.wikipedia.org/wiki/Q–Q_plot#Interpretation)
    * ou Kolmogorov-Smirnov test
    * correction: log ou box-cox

* **Independence**: no or little multicollinearity between variables
    * test: Correlation matrix

* **Homoscedasticity**: for a given variable the low and high range have the same statistical properties with respect to the residuals
    * test: Chunk data and Check Variance

* All **Confounders** accounted for

http://www.statisticssolutions.com/assumptions-of-linear-regression/

https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/

https://www.kaggle.com/questions-and-answers/57571
What does Linear Relationship means? :- You can check linearity with scatter plots, if there is little to no linearity in the scatter plot between your dependent and Independent variables then the assumption doesn't hold.
Linearity regression assumption requires all variables to be normal, how to check? :- You can check this assumption with a Q-Q plot, if your data deviates substantially from the line on the Q-Q plot, then this assumption doesn't hold.
check for little to no multicollinearity, why is multicollinearity a problem? :- Multicollinearity generally occurs when there are high correlations between two or more predictor variables. A principal danger of such data redundancy is that of overfitting in regression analysis models. The best regression models are those in which the predictor variables each correlate highly with the dependent (outcome) variable but correlate at most only minimally with each other. Such a model is often called "low noise" and will be statistically robust (that is, it will predict reliably across numerous samples of variable sets drawn from the same statistical population).

</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
Correlation
</p>
</div>
</section>

<section data-markdown>
# Rappel pearson coefficient

Etudier la corrélation entre deux ou plusieurs variables aléatoires ou statistiques numériques, c’est étudier l'intensité de la liaison qui peut exister entre ces variables.

Il y a différentes façon de calculer la corrélation de 2 variables.

La plus commune est [Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)

Qui se calcule suivant :

$$r=\frac{\sum\_{i=1}^{n} (x\_{i}-{\bar{x}}) (y\_{i}-{\bar{y}})} { {\sqrt {\sum\_{i=1}^{n}(x\_{i}-{\bar {x}})^{2}}}{\sqrt {\sum\_{i=1}^{n}(y\_{i}-{\bar {y}})^{2}}}}$$

où :

* \\(n\\) nombre d'échantillons

* \\(x\_{i},y\_{i}\\) les échantillons

* \\( \bar{x} = \frac{1}{n} \sum\_{i=1}^{n} x\_{i} \\) la moyenne; de meme pour  \\({\bar {y}}\\)

</section>

<section data-markdown>
# Correlation
![Correlation](/assets/02/pearson-correlation-coefficient-illustration.png)

</section>

<section data-markdown>
# Correlation

On va regarder l'influence de la correlation entre les predicteurs

    df.corr()

Les prédicteurs ```horsepower``` et ```weight``` sont très corrélés, ```displacement``` et ```cylinders``` aussi.

<img src=/assets/02/autompg-correlation.png  height=400>

</section>

<section data-markdown>
# Correlation \\(\neq\\) Causalité
http://www.tylervigen.com/spurious-correlations

<img src=/assets/02/spurious_correlations.png>


</section>

<section data-markdown>

# Regression and Causation:


For regression coefficients to have a causal interpretation we need both that
* the linear regression assumptions hold: linearity, normality,
independence, homoskedasticity
* and that all confounders of, e.g., the relationship between treatment A and Y be in the model.

Not the same thing

Calculating Correlation: easy

Demonstrating and Quantifying Causation: Causal Inference: Not so easy


=> However most common strategy is to find not causality but correlation through linear regression which can be interpreted as causality under strong assumptions on the covariates.

**Works under VERY strong assumptions**

</section>

<section data-markdown>
# Confonders

facteurs potentiels de confusion

https://www.r-bloggers.com/how-to-create-confounders-with-regression-a-lesson-from-causal-inference/

http://www.statisticshowto.com/experimental-design/confounding-variable/

![Confounders](/assets/02/confounder.png)

* Relationship between **ice-cream consumption** and number of **drowning deaths** for a given period

**Confounding: ?**
</section>

<section data-markdown>

![xkcd](/assets/02/xkcd_correlation.png)

</section>

<section data-markdown>
<div class=centerbox>
<p class=top>Récapitulatif</p>
<p class=top></p>
</div>
</section>


<section data-markdown>

# Récapitulatif

* Regression lineaire, simple et explicite
* Attention à ce que les predicteurs soient decorrélés
* R^2 ajusté au lieu de R^2

</section>

<section data-markdown>
# Lab de cette apres midi
Regression lineaire sur le  dataset *advertising*

![advertising](/assets/02/advertisingscatterplots.png)

</section>

<section data-markdown>
# Questions

</section>

<section data-markdown>
# Liens et resources

* [Régression linéaire en python](http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/notebooks/regression_lineaire.html) sur le site de Xavier Dupré

* [OLS sur wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares): tres complet

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
