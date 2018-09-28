---
layout: slide
title: 3) Régression Logistique
description: none
transition: slide
permalink: /3-regression-logistique
theme: white
---

<section data-markdown>
<div class=centerbox>
<p class=top>
Régression Logistique
</p>
</div>
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
* Statsmodel
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
# Classification ou Régression

Voici une variable à prédire, une condition médicale:

* 0: Attaque cardiaque
* 1: Crise d'épilepsie
* 2: Overdose

Pourquoi ne pas utiliser une regression linéaire pour prédire cette variable ?

L'encodage de la variable (ordre et continuité) implique que

* il y a un ordre entre les catégories: Attaque < Crise < Overdose
* Toutes ces catégories sont équidistantes

# Dans le cas binaire

* 0: Attaque cardiaque
* 1: Crise d'épilepsie

On pourrait utiliser une regression lineaire comme substitut de probabilité mais on obtiendrait peut etre des valeurs en dehors de [0,1]

Donc utiliser des modèles de classification est plus approprié!

</section>

<section data-markdown>
# Regression ou Classification?
Review the following situations and decide if each one is a regression problem, classification problem, or neither:

* Using the total number of explosions in a movie, predict if the movie is by JJ Abrams or Michael Bay.
* Determine how many tickets will be sold to a concert given who is performing, where, and the date and time.
* Given the temperature over the last year by day, predict tomorrow's temperature outside.
* Using data from four cell phone microphones, reduce the noisy sounds so the voice is crystal clear to the receiving phone.
* With customer data, determine if a user will return or not in the next 7 days to an e-commerce website.

</section>

<section data-markdown>
# Régression Logistique


Aussi appelée **logit regression**, **maximum-entropy classification (MaxEnt)** ou log-linear classifier.

Au lieu de prédire la catégorie de la variable cible, on va prédire la probabilité que cette variable appartienne à la catégorie en question :


$$ P(Y = 1 \bigg{/} X) $$

que l'on note \\( p(X) \\)

et comme pour la regression linéaire on vuet avoir un modlèle linéaire simple pour estimer cette probabilité.

$$ P(Y=1 / X) =  p(X) = \beta_0 + \beta_1 X $$

mais pour que \\(p(X)\\) soit une probabilité il faut que ses valeurs soient comprises dans \\( [0, 1] \\) ce qui n
est pas forcement le cas avec la formule ci-dessus.

</section>

<section>

<div style='float:right; width:45%; '>
    <div data-markdown>
    # Fonction Sigmoide
<img src=/assets/03/sigmoid.svg style='width:300px; border:0'>

Cette fonction réduit \\( \mathbb{R} \\)  à l'intervale \\( [0,1] \\)
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Régression logistique

Le modèle linéaire \\(p(X) = \beta\_0 + \beta\_1 X\\)  est pris comme attribut de la fonction sigmoide.

$$ f(z) = \frac{e^{z} }{1 + e^{z}} =  \frac{1 }{1 + e^{-z}} $$

ce qui donne

$$ p(X) = \frac{e^{(\beta\_0 + \beta\_1 X)} }{1 + e^{(\beta\_0 + \beta\_1 X)}}  $$


</div>
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

En utilisant :

1. default vs balance
2. default vs balance, income et student

* calculer la probabilité de default pour

    * Un etudiant avec un solde debiteur de 1500 et un revenude 40000
    * un non etudiant avec le meme solde et meme revenue

Pourquoi est ce que le coefficient relatif a la variable student est positive quand student est la seule variable alors qu'elle est negative dans le cas multinomial ?


</section>


<section data-markdown>
# Catégorie et évènement

Question de vocabulaire:

* un évènement = la variable appartient à la catégorie

</section>


<section data-markdown>
# Odds ratio

Aussi appelé  rapport des chances, **rapport des cotes** ou **risque relatif rapproché**

Comment quantifier l'impact d'une variable predicteur sur la probabilité de la catégorie ?

On a:

$$ p(X) = \frac{e^{\beta\_0 + \beta\_1 X} }{1 + e^{\beta\_0 + \beta\_1 X}}  $$

Le **odds ratio**: est le rapport entre la probabilité de l'évènement sur la probabilité du non évènement.

$$
\frac{p(X)}{ 1 -p(X)} = e^{\beta\_0 + \beta\_1*X}
$$

* Odds ratio  \\( \in [0, +\infty] \\)
* Odds ratio proche de  0: probabilité faible que l'évènement survienne
* Odds ratio s'approchant de  \\( +\infty \\) : probabilité forte que l'évènement survienne

</section>

<section data-markdown>
# Log-Odds ratio
Si on prends le log du *odds-ratio* on a le **log odds ratio**

$$
log(\frac{p(X)}{ 1 -p(X)}) = \beta\_0 + \beta\_1 X
$$

Cela mesure l'influence d'une variable sur la cible.

Si \\( \beta\_1 \\)  augmente, alors  \\( p(X)\\) augmente aussi.

C'est moins direct que dans le cas de la régression linéaire.

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

* 9 out of 10 people (90%)  with odds 9 will default


</section>

<section data-markdown>
# Maximum de vraisemblance
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Métriques de classification
<img src=/assets/03/classification_metrics.png style='width:450px;'>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Métriques de régression
<img src=/assets/03/regression_metrics.png style='width:450px;'>

    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Faux Positifs et Faux Négatifs
<img src=/assets/03/pregnant.jpg style='width:400px;border:0'>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Métrique: Accuracy ou Précision

Correctement identifiés

* TP = True Positive - Vrai positif
* TN = True Negatives  - Vrai négatifs

Incorrectement identifiés

* FP = False Positive
* FN = False Negatives

## Accuracy

On définit la précision par

$$ Accuracy = \frac{ TP + TN  }{TP + FP + TN + FN}   $$

</div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Confusion matrix

![confusion matrix](/assets/03/confusion_matrix.png)
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# classification
![cats and dogs](/assets/03/cats_dogs.png)


    </div>
</div>
</section>


<section data-markdown>

# Ca se complique assez rapidement

<img src=/assets/03/confusion_matrix_wikipedia.png style='border:0'>

[https://en.wikipedia.org/wiki/Confusion_matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
</section>

<section data-markdown>
# Confusion matrix
Avec scikit:

        from sklearn.metrics import confusion_matrix
        y_true = [0,0,0,0,0,1,1,1,1,1]
        y_pred = [0,0,0,1,1,0,1,1,1,1]
        confusion_matrix(y_true, y_pred)

</section>


<section data-markdown>
<div class=centerbox>
<p class=top>
Machine learning
</p>
<p class=mitop>Au revoir les statistiques :)</p>
</div>

</section >

<section >
<div style='float:right; width:30%'>

    <div data-markdown>
    <img src=/assets/03/stats-vs_ml.jpeg>
    </div>
</div>

<div style='float:left;width:50%;'>
<div data-markdown>


# Des stats au machine learning

* Statsmodel est dans une approche **statistique classique** qui favorise l'interprétabilité et l'analyse des prédicteurs

* Scikit-learn est dans une approche **machine learning** plus orientée vers la  robustesse et la prédiction

Au niveau de l'implémentation de la regression logistique dans les 2 librairies, la difference est que scikit ajoute une **contrainte** sur le modele au niveau de la fonction de cout.

Cette contrainte est appelé **régularisation** et sert à accroitre la capacité du modele a extrapoler sur des donnees nouvelles. On verra cela en detail dans 2 jours.

On est donc dans une transition de la modélisation statistique vers la modélisation machine learning.
</div>
</div>
</section>

<section data-markdown>

# Regression avec scikit-learn

On va avoir des meta parametres.

Par exemple:

* accès à differents algorithmes  pour trouver les coefficients + un certain controle sur leur fonctionnement
* différentes façon de traiter le multi-class: ovr, multinomial
* différents mode de régularisation

et en output

* un modele que l'on peut appliquer a de nouvelles donnees
* les intervals de confiance
* la ou les categories prédites
* les probabilités de prédiction (appartenance a la classe)



On n'aura plus:
- les p-value
- les tests statistiques
- le R^2 (pas directement en tout cas)
</section>


<section data-markdown>
# Scikit-learn LogisticRegression

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

![scikit LogisticRegression](/assets/03/scikit-logistic-regression.png)

</section>

<section data-markdown>
![](/assets/03/sklearn_01.png)

</section>

<section data-markdown>
![](/assets/03/sklearn_02.png)

</section>

<section data-markdown>
![](/assets/03/iris_screen_shot_01.png)
</section>

<section data-markdown>
![](/assets/03/iris_screen_shot_02.png)
</section>

<section data-markdown>
# Demo Scikit-learn LogisticRegression

* Iris dataset
* Score
* ROC AUC
* Trouver le meilleur threshold (Acc, max P, min Neg, TPR, ...)
* Use predict_proba and a different threshold => you should find a different confusion matrix


    import pandas as pd
    from sklearn import datasets, metrics
    from sklearn.linear_model import LogisticRegression


    iris = datasets.load_iris()
    clf = LogisticRegression()
    clf.fit(iris.data, iris.target)

    metrics.accuracy_score(y_test, y_hat )


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

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Exemples

* Marque de voiture: Audi, Renault, Ford, Fiat

Si on assigne un numero arbitraire à chaque marque de voiture on crée une hiérarchie:

Audi =>1 , Renault => 2, Ford => 3, Fiat => 4

De meme:

* chien, chat, souris, poulet => {1,2,3,4}

pourquoi le poulet serait *4* fois le chien ? Ca ne fait pas sens.


Parfois on peut quand meme assigner un chiffre à chaque categorie, catégories ordonnées

* enfant, jeune, adulte, vieux => {1,2,3,4}
* negatif, neutre, positif => {-1, 0, 1}


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# One hot encoding et label encoding

Comment traduire les variables quantitative en variables numeriques

Binaires
* Oui / Non ; 1 /0
* Homme / Femme
* Spam / legit
* Action: Achete, enregistre,
* Identification


Multinomiales
* liste de villes, pays, destinations,
* tranche d'age
* niveau d'etude
* marques de voiture

    </div>
</div>
</section>

<section data-markdown>
# One hot encoding

[One Hot Encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), ou pandas.get_dummies

Si on a **N** classes, on crée **N-1** variables binaires:

par exemple la variable ```animal_type: chien, chat, souris, poulet``` sera transformée en 3 variables binaires

* est_ce_chien : 1/0
* est_ce_chat : 1/0
* est_ce_souris : 1/0

La variable *est_ce_poulet* étant redondante et automatiquement déduite des 3 autres.

# LabelEncoder
[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) associe un chiffre a chaque classe, on garde l'ordonnancement

* enfant, jeune, adulte, vieux => {1,2,3,4}

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
