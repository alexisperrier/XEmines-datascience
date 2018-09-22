# Notebook d'exploration et d'utilisation de Pandas

Dans ce notebook vous aller étudier le dataset les-arbres.csv disponibles sur https://opendata.paris.fr/explore/dataset/les-arbres/


Ce dataset comprend des informations sur 200k arbres dans Paris.

* Espèces, genres, famille
* Adresse, géolocalisation
* Environnement: rue, jardin, ...
* Hauteur et circonférence
* Arbres remarquables

# But

Ecrire un notebook jupyter d'analyse de ce dataset qui comprenne:

* des cellules explicatives en markdown sur votre démarche d'analyse
* du code python
* des graphs d'illustration

Le but est de livrer une document qui permetrrait à une personne de comprendre et de refaire votre analyse sans connaissance au préalable du dataset.

Le lab est composé de plusieurs parties

## 1. charger le dataset dans une dataframe pandas

* ```df = pd.read_csv()```
* avant cela n'hesitez pas à ouvrir le fichier csv
* attention au séparateur utilisé dans le csv

## 2. exploration:

* statistiques des variables numériques ```df.describe()```
* occurences des catégories ```df['DOMANIALITE'].value_counts()```
* visualisation des variables ```df['HAUTEUR (m)'].hist(bins = 100)```

Cette exploration va vous permettre de trouver les outliers (les valeurs aberrantes) et de les enlever du dataset.

## 3. nature des arbres  par  arrondissement
Vous aller ensuite analyser la nature des arbres par arrondissement, par espèces et par *domanialité*.

* nombre d'arbres
* nombre de variétés d'arbres
* statistiques de la hauteur et de la circonférence des arbres

Observe-t-on des différences significatives entre les arrondissements ?

N'hesitez pas à illustrer vos analyses par des graphes matplotlib: scatterplot, boxplot ou barchart.

## 4. les arbres remarquables

Certains arbres sont taggés comme étant remarquables.
* Qu'est ce qui caractérise ces arbres par rapport aux autres? leur espèce ? leur taille ?
* Comment gérer les valeurs manquantes de cette colonne *remarquable* ? S'agit-il d'une erreur ou peut-on supposer que quand la valeur manque l'arbre n'est en fait pas *remarquable*.

## Les arrondissements les plus verts

On prends ensuite en compte le dataset arrondissement.csv qui contient la superficie de chaque arrondissement *intra muros*. Le but est de voir quel arrondissement a le plus d'arbres par rapport à sa superficie.

Pour joindre les 2 datasets, il faut d'abord faire quelques modifications sur le dataframe des arbres.

* créer une variable booléenne ```dans_paris``` qui indique si l'arrondissement est bien dans Paris
* utiliser cette variable pour supprimer tous les arbres qui ne sont pas dans Paris

Les arrondissements ne sont pas écrits de la meme façon dans les 2 datasets.

D'un coté on a le code postale sous la forme 75112 et de l'autre un texte: PARIS 12E ARRDT.

* écrire une  fonction qui prenne en entrée le texte de l'arrondissement et qui retourne le code postal


    PARIS 11E ARRDT => 75112

* utiliser cette fonction pour créer une nouvelle variable dans le dataset les arbres qui contienne le code postal. Eviter de préférence eviter de faire une boucle sur les 200k rangées du dataframe arbres. Utilisez plutot le pattern ```df[var_cible] = df[var_source].apply(lambda d : fonction(d))``` si vous la connaissez.

* en utilisant ```groupby()``` et ```count()```, créez une nouvelle dataframe qui contienne le nombre d'arbre par arrondissement. Cette dafarme doit avoir 20 rows.

* joindre cette nouvelle dataframe avec la dataframe issue du fichier arrondissement en utilisant ```df.merge()```

Illustrer le ratio

*nombre d'arbres* / *superficie*

 par arrondissement par un graphe (barchart par exemple).
