---
jupyter:
  jupytext_format_version: '1.0'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.2
---

# TD 03: Régression logistique sur le Titanic

Vous allez travailler sur le dataset du **Titanic**, un grand classique.

Ce dataset fait l'objet d'une compétition kaggle.

https://www.kaggle.com/c/titanic

Contrairement aux données sur Kaggle qui sont séparées en 2 datasets training et testing, vous aller travailler sur tout le dataset.

Le but du TD est de construire un modele de régression logistique avec la librairie scikit-learn. Le modèle est décrit sur la page http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

Il vous faut trouver le meilleur modèle au sens des métriques:

* Accuracy
* AUC


# Le dataset

Vous avez des information sur 1300 passagers




* survival:	Survival	0 = No, 1 = Yes
* pclass:	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
* sex:	Sex
* Age:	Age in years
* sibsp:	# of siblings / spouses aboard the Titanic
* parch:	# of parents / children aboard the Titanic
* ticket:	Ticket number
* fare:	Passenger fare
* cabin:	Cabin number
* embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


et la variable a prédire est une variable binaire: Survived

# 1. Exploration

Chargez le dataset dans une dataframe .


* Explorez et analysez les différentes variables.

* Regardez (histogram) en particulier la relation entre les variables ```age``` et ```Gender``` avec la cible ```Survived```. Qu'en conclure ?


# 1. Prédicteurs numeriques

En ne prennant en compte que les variables numériques, construisez un modèle de prédiction avec le modele  [sklearn.linear_model.LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

Pour démarrer:

    from sklearn.linear_model import LogisticRegression
    ....


* quel est le score (accuracy et AUC) du modele ?
* matrice de confusion ?


# 2. Valeurs manquantes
Une des caractéristique de ce dataset est l'absence de nombreuses valeurs pour la variable Age.
On va tester 3 methodes de remplacement

1. remplacez les valeurs manquantes par la moyenne de l'age sur tous les autres echantillons
2. Calculer la moyenne pour les femmes et une pour les hommes et remplacer les valeurs manquantes par la moyenne appropriée
3. utiliser la moyenne harmonique

# 3. Variables catégorielles

Certaines variables sont des catégories: notamment "Embarked". "Gender" et "Destination".

Choisissez une des methodes vues dans le cours pour numeriser ces variables : one hot encoding et LabelEncoder.

et ajoutez ces variables en tant que predictuers dans le modèle

# 4. Regularization

Faites varier les parametres du modele
* penalty: l1, l2
* C: 0.1, 1, 10

Qu'observez vous ?

# 5. extraire les infos du noms

Il est possible d'extraire le titre du nom du passager avec une expression régulière.

    df['title'] = df.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

Une fois le titre extrait, traitez cette variable comme une categorie,  l'encoder et l'ajouter en tant que prédicteur.

* Accuracy et AUC ?
* Matrice de confusion

Voit-on une amélioration ?

# 6. prediction sur de nouvelles données

Construisez a la main un dictionnaire  corresponde a un nouveau passager

Par exemple

    {
        'pclass': 1,
        'sex': male,
        'age': 30,
        'fare': 112,
        'sibsp' : 0,
        'parch' : 1,
        'title' : Mr,
        'home.dest': Montreal, PQ,
        'embarked': S
    }

Transformer ce dictionnaire passager en vecteur numerique (de la bonne taille) et calculez sa probabilité de survie.


# 6. SGD classifier

Utilisez maintenant le [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) avec comme fonction de cout : logloss

Quel modele performe le mieux ?
