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

# Lab SVM on Caravan

Vous allez travailler sur le dataset Caravan

[Caravan dataset on kaggle](https://www.kaggle.com/uciml/caravan-insurance-challenge)

* 86 variables
* Fortement déséquilibré
    * No     5474
    * Yes     348

Le but est de prédire si une personne va acheter une poce d'assurance pour une caravane en fonction de son profil.

Les 86 variables sont expliquées sur [cette page](https://www.kaggle.com/uciml/caravan-insurance-challenge/home).

Nous n'allons pas rentrer dans le détail des variables.

Le but de ce lab est d'appliquer les techniques de subsampling et oversampling pour améliorer le score. Nous comparerons aussi différentes métriques de scoring: AUC, F1-score et [Cohen's Kappa](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)

## 1) Data Processing

* Chargez le dataset dans une dataframe pandas
* Convertissez les variables categorielles en numerique avec LabelEncoder
* Extraire les subset de train et test en veillant bien  a melanger (shuffle) le dataset et à stratifier les classes de façon a éviter que les train ou test subset ne contienne que peu ou pas assez de la classe minoritaire.

## 2) Modelisation

En considerant le dataset original déséquilibré, prendre le classifier support vector machine  [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),

Pour un SVC, calculer les scores (AUC, F1-score et Cohen's Kappa ) sur les training et testing subsets. Comparez differents kernel et differentes valeurs pour C et Gamma.

Est-on en presence d'un bon classifier en terme de biais / underfitting et de variance / overfitting ?

## 3) Grid search cross validation
En utilisant [Grid Search CV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) et en définissant une série de parametres
* C
* gamma
* kernel: linear, rbf, ...

Trouvez le meilleur SVC.

Observez vous une amélioration significative par rapport a vos modeles de l'etape 2) ?


## 4) undersampling

Nous allons essayer d'améliorer la performance du classifier en sous échantillonnant la classe majoritaire.

Construisez X et y de facon a ce que \\(n_{Maj} \simeq K n_{Min} \\) avec \\( K \in [1,2] \\).

Trouvez le meilleur classifier avec la methode Grid Search CV et calculer les scores AUC, F1 et Kappa.

## 5) oversampling

En utilisant la technique du bootstrap, sur échantilloner la class minoritaire.

Construisez X et y de facon a ce que \\(n_{Min} \simeq  n_{Maj} / K \\) avec \\( K \in [1,3] \\).

Trouvez le meilleur classifier avec la methode Grid Search CV et calculer les scores AUC, F1 et Kappa.

Si on fait varier K, le rapport entre le nombre d'échantillons par classe, observe t on une valeur seuil de K pour laquelle les performances du classifier sont significativement meilleures.
