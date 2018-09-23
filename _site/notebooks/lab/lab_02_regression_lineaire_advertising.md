# Lab 02:

Vous allez appliquer la régression linéaire sur le dataset advertising.

Ce dataset est composé de 200 échantillons et des variables suivantes:

* 3 prédicteurs **TV**, **Radio**, **Newspaper**: qui sont les sommes dépensées pour chacun de ces média (k$)
* une variable cible continue: *Sales* qui correspond aux ventes réalisées

Le but de ce TD est de trouver le modèle qui minimise le plus l'erreur quadratique (MSE). L'erreur quadratique est définie comme le carrée de la différence entre les predictions et les vraies valeurs.


# 1. Exploration du dataset

* Chargez le csv dans une dataframe pandas ```df```
* Quelle est la distribution de chaque variable?
* Y a t il des valeurs manquantes ou aberrantes (outliers)
* Quelle est la corrélation entre chacune des variables
* Faire un scatterplot de chaque couple de variable
* Que pouvez vous en déduire ?

# 2. Modeles univariables

Pour chacun des predicteurs (TV, Radio, Newspaper) creer un modèle univariable.

    import statsmodels.formula.api as smf
    lm = smf.ols(formula='Sales ~ Radio ', data=df).fit()

* Determinez le meilleur modèle en fonction
    * de R^2
    * des coefficients
    * des pvalues ```lm.pvalues```
    * de la MSE (utilisez ```lm.fitted_values``` pour les valeurs prédites)

Le coefficient de la regression Sales ~ Radio est plus petit que le coefficient de la regression Sales ~ TV alors que TV est plus corrélé à Sales que Radio, comment expliquer cela ?

Comment modifier les données pour que le coefficient de la regression linéaire reflete l'importance de la variable par rapport aux autres.

Dans le modele ```Sales ~ TV``` que représente l'intercept ?
Si le budget TV est nul, combien d'unité de Sales seront quand meme vendu [95% interval] ?

# 3. modele multi variables

Construisez maintenant le modele a partir des 3 prédicteurs ```Sales ~ Radio + TV + Newspaper```.

* Qu'observez vous en terme d'importance relative des prédicteurs ?
* Si vous augmentez de 50 les sommes allouées au média TV, de combien augmentent les ventes.

* Comment expliquer que le coefficient pour Newspaper est presque nul, légérement négatif, dans le modèle complet tandis qu'il est positif lorsque pris en compte individuellement ?

* Est ce que enlever la variable Newspaper améliore le modèle ? Au niveau R^2, R^2_adj et MSE ?

# 4. modele multiplicatif

Rajoutez au modèle la variable multiplicative ```tv_radio = TV * Radio  ```.

Comment interpreter que cette variable ait une si forte influence sur le modèle ?
