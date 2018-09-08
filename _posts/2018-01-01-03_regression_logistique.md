---
layout: slide
title: Régression Logistique
description: none
transition: slide
permalink: /03-regression-logistique
theme: white
---


<section data-markdown>
# Régression Logistique
</section>

<section>
<div style='float:right;'>
    <h1>Questions ?</h1>

    <div data-markdown>
    <img src=/assets/03/questions.gif>
    </div>
</div>

<div data-markdown>
# Cours précédent
* Régression linéaire - OLS
* Interpretation
    * p-value
    * R^2

* Correlation
* Statsmodel python
</div>
</section>


<section>

<div style='float:right;'>
    <div data-markdown>
    ## Lab: titanic
    <img src=/assets/03/titanic_photo.jpg style='width:300px; border:0'>
    </div>
</div>

<div data-markdown>
# Programme

* Regression logistique
* odds ratio, log odds ratio
* Maximum de vraisemblance
* encoding categorical values
* Metriques de classification
    * confusion matrix
    * AUC and ROC
* Outliers, detection and impact
* Skewness: Box cox and Kurtosis

</div>

</section>

<section data-markdown>
# More on Classification vs Regression

Why not use linear regression to predict some medical condition such as

* 0: Stroke,
* 1: Epileptic seizure
* 2: Overdose

Encoding it like that and using Linear Regression implies:

* order of the encoding
* equal distance between codes

# In the binary case:

* 0: Stroke,
* 1: Epileptic seizure


Possible to use linear regression as a proxy for a probability

* May end up with results outside the [0,1] range

So classification specific models better!

</section>

<section data-markdown>
# Regression or Classification?
Review the following situations and decide if each one is a regression problem, classification problem, or neither:

* Using the total number of explosions in a movie, predict if the movie is by JJ Abrams or Michael Bay.
* Determine how many tickets will be sold to a concert given who is performing, where, and the date and time.
* Given the temperature over the last year by day, predict tomorrow's temperature outside.
* Using data from four cell phone microphones, reduce the noisy sounds so the voice is crystal clear to the receiving phone.
* With customer data, determine if a user will return or not in the next 7 days to an e-commerce website.

</section>

<section data-markdown>
# Logistic regression

also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier.

Au lieu de prédire la categorie auquelle appartient la variable cible. on va predire la probabilité que cette variable appartienne a la category en question.




$$ P(Y = 1 \bigg{/} X) $$

which we note \\( p(X) \\)

and similarly to Linear Regression we want a **simple linear model** for that probability

$$ P(Y=1 / X) =  p(X) = \beta_0 + \beta_1 X $$

but that still does not give us values between [0, 1]

</section>

<section>

<div style='float:right;'>
<h1> Sigmoid function </h1>
<div data-markdown>
<img src=/assets/03/sigmoid.svg style='width:300px; border:0'>
</div>
</div>

<div data-markdown>

# Logistic regression

So instead we feed the linear model to the sigmoid function

$$ f(z) = \frac{e^{z} }{1 + e^{z}} =  \frac{1 }{1 + e^{-z}} $$

We feed $$ z = P(Y=1 / X) =  p(X) = \beta\_0 + \beta\_1 X $$ to the sigmoid function

$$ p(X) = \frac{e^{(\beta\_0 + \beta\_1 X)} }{1 + e^{(\beta\_0 + \beta\_1 X)}}  $$

because this function shrinks \\( \mathbb{R} \\)  to \\( [0,1] \\)
</div>
</section>


<section data-markdown>
# logistique regression en python
avec statsmodel

### appliqué au *default* dataset

4 colonnes

* student: étudiant?
* balance: compte en banque
* income: revenues

prédiction : va défaulter sur son crédit ou non

Using:

1. default vs balance
2. default vs balance, income and student

* Calculate the probability of default for
    * a student with a credit card balance of \\$1500 and income of \\$40k
    * a non-student, same balance and income

Why is the coefficient for student positive when student is the only factor and negative in the case of multilinomial logistic regression?

</section>


<section data-markdown>
# Evenement et categorie

Question de vocabulaire:
On parle d'evenement le fait que la variable cible appartienne a une categorie.

La probabilité que la variable cible soit dans la categorie 1 = probabilité de l'evenement 1
</section>


<section data-markdown>
# Odds ratio

Aussi appelé  rapport des chances, rapport des cotes1 ou risque relatif rapproché

Comment quantifier l'impact d'une variable predicteur sur la probabilité de la catégorie ?

On a:

$$ p(X) = \frac{e^{\beta\_0 + \beta\_1 X} }{1 + e^{\beta\_0 + \beta\_1 X}}  $$

Le **odds ratio**: est le rapport entre la probabilité de l'evenement sur la probabilité du non evenement.

$$
\frac{p(X)}{ 1 -p(X)} = e^{\beta\_0 + \beta\_1*X}
$$

* Odds ratio  \\( \in [0, +\infty] \\)
* Odds close to 0: low probability of the event happening
* Odds close to \\( +\infty \\) : low probability of the event happening

</section>

<section data-markdown>
# Log-Odds ratio
Si on prends le log du *odds-ratio* on a le log odds ration

$$
log(\frac{p(X)}{ 1 -p(X)}) = \beta\_0 + \beta\_1 X
$$

Increase in \\( \beta\_1 \\) => results in increase in \\( p(X)\\).

Not as direct and linear as in the case of linear regression.
</section>

<section data-markdown>
# Log-Odds ratio: application
Sur le data set default:

* On accroit / décroit le compte en banque de 10k
* On accroit / décroit le revenu de 10k

## Exemple in the default dataset

\\( p(X) = 0.2 \iff  \frac{0.2}{1 -0.2} = 0.25 \\)

* 1/5 people with ods 1/4 will default

\\( p(X) = 0.9 \iff  \frac{0.9}{1 -0.9} = 9 \\)

* 9 out of 10 people (90%)  with ods 9 will default


</section>

<section data-markdown>
# Maximum de vraisemblance
</section>

<section>
<div style='float:right;'>
    <h1>Metriques de classification</h1>

    <div data-markdown>
    <img src=/assets/03/classification_metrics.png style='width:400px;border:0'>
    </div>
</div>

<div data-markdown>
# Metriques de regression
<img src=/assets/03/regression_metrics.png style='width:400px;'>
</div>
</section>


<section>

<div style='float:right;'>

    <div data-markdown>
    <img src=/assets/03/pregnant.jpg style='width:400px;border:0'>
    </div>
</div>


<div data-markdown>
# Metrics
Correctly identified:

* TP = True Positive
* TN = True Negatives

Incorrectly identified:

* FP = False Positive
* FN = False Negatives
## Accuracy

How to you define accuracy?

$$ Accuracy = \frac{ TP + TN  }{TP + FP + TN + FN}   $$

</div>
</section>


<section data-markdown>
# Confusion matrix

![confusion matrix](/assets/03/confusion_matrix.png)

</section>

<section data-markdown>

# ca se complique assez rapidement

<img src=/assets/03/confusion_matrix_wikipedia.png style='border:0'>

</section>

<section data-markdown>
# Confusion matrix
Avec scikit:

        from sklearn.metrics import confusion_matrix
        y_true = [0,0,0,0,0,1,1,1,1,1]
        y_pred = [0,0,0,1,1,0,1,1,1,1]
        confusion_matrix(y_true, y_pred)

</section>
<section >
<div style='float:right; width:30%'>

    <div data-markdown>
    <img src=/assets/03/stats-vs_ml.jpeg>
    </div>
</div>

<div style='float:left;width:50%;'>
<div data-markdown>


# quittons les stats pour rejoindre sur le machine learning

* Statsmodel est dans une approche statistique classique qui favorise l'interpretabilité

* Scikit est dans une approche machine learning plus orientée robustesse et prediction

    * une regression est un modele parmi d'autres

Au niveau du modele, la difference est que scikit ajoute une contrainte sur le modele au niveau de la fonction de cout. cette contrainte est appelé regularization et sert a accroitre la capacité du modele a "marcher" sur des donnees nouvelles. On verra cela en detail dans 2 jours.

On est donc dans une transition de la modelisation statistique vers la modelisation machine learning.
</div>
</div>
</section>

<section data-markdown>

# Regression avec scikit-learn

On va avoir des meta parametres. Par exemple:

* acces a differents algo pour trouver les coefficients et un certain controle sur leur fonctionnement
* differentes façon de traiter le multi-class: ovr, multinomial
* differents mode de regularization

et en output

* un modele que l'on peut appliquer a de nouvelles donnees
* les intervals de confiance
* la ou les categories predites
* les proba de prediction
et surtout un model qye


On n'aura plus:
- les p-value
- le R^2
- les tests statistiques
</section>


<section data-markdown>
# Scikit-learn LogisticRegression

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

![scikit LogisticRegression](/assets/03/scikit-logistic-regression.png)

</section>

<section data-markdown>
# Demo Scikit-learn LogisticRegression

* default dataset
* score
* plot proba hist
* trouver le meilleur threshold (Acc, max P, min Neg, TPR, ...)
Use predict_proba and a different threshold => you should find a different confusion matrix


* QQ plot residuals
* matrice de confusion

Essayer plusieurs regularization L2 et L1 avec differents C

</section>

<section data-markdown>
# AUX and ROC Curve

* TPR = \\( \frac{  TP }{ P}  = \frac{  TP }{ TP + FN} \\) aka **Sensitivity** or **Recall**
* FPR = \\( \frac{  FP }{ N}  = \frac{  FP }{ FP + TN} \\)  aka **Fall-out**

Le TPR / recall et le FPR varient en fonction du seuil. on obtient donc

### Receiver operating characteristic

ROC = TPR vs FPR pour different seuils

        sklearn.metrics.roc_curve returns TPR, FPR

plot to get the ROC Curve

### AUX and ROC Curve
The AUX is Area under the Curve

        sklearn.metrics.roc_auc_score

So what's your best model LR according to AUC?

voir aussi F1-score

</section>

<section data-markdown>
# One hot encoding

Comment traduire les variables quantitative en variables numeriques

Binaires
* est ce un etudiant
* fille / garcon
*

Multinomiales
* liste de villes, pays, destinations,
* tranche d'age
* niveau d'etude
* marques de voiture

Par exemple: Audi, Renault, Ford, Fiat
Si on assigne un numero arbitraire a chqaue marque de voitue on crée une hierarchie
Audi =>1 , Renault => 2, Ford => 3, Fiat => 4

* chien,chat,souris,poulet => {1,2,3,4}
pourquoi le poulet est 4 fois le chien ? ca ne fait pas sense


Mais parfois on peut quand meme assigner un chiffre a chaque categorie, catégories ordonnées

* enfant, jeune, adulte, vieux => {1,2,3,4}
* negatif, neutre, positif => {-1, 0, 1}

</section>

<section data-markdown>
# One hot encoding

[One Hot Encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), ou pandas.get_dummies

Si on a N classes, on crée N-1 variables binaires
par exemple negatif, neutre, positif: est_neutre, est_positif (est_negatif est deduite des 2 autres variables pas besoin de la specifier)


# LabelEncoder
[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) associe un chiffre a chaque classe, on garde l'ordonnancement

</section>

<section data-markdown>
# Recap
* regression logistique
* approche stats vs approche ML
* matrice de confusion
* ROC-AUC
* One hot encoding
</section>

<section data-markdown>

# Resources
* Logistic Regression: Why sigmoid function?
https://github.com/rasbt/python-machine-learning-book/blob/master/faq/logistic-why-sigmoid.md

* scikit-learn documentation: Logistic regression,
http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

* No, Machine Learning is not just glorified Statistics
https://towardsdatascience.com/no-machine-learning-is-not-just-glorified-statistics-26d3952234e3

* on stackexchange When to use One Hot Encoding vs LabelEncoder vs DictVectorizor? https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor


</section>
