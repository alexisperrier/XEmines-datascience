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

# Notebook d'exploration et d'utilisation de Pandas

Dans ce notebook vous aller étudier le dataset **les-arbres.csv** et le croiser avec les données **arrondissement.csv** 

Le fichier **les-arbres.csv** est dispo sur le drive:
https://um6p-my.sharepoint.com/:t:/g/personal/alexis_perrier_emines_um6p_ma/EamAeWJzvy5AtjFpV8xrqPgBXNbYiAr2gkS7XppEKfBpug?e=NIe5I5

Il provient de 
https://opendata.paris.fr/explore/dataset/les-arbres/

Le fichier **arrondissement.csv** est disponible sur le drive https://um6p-my.sharepoint.com/:x:/g/personal/alexis_perrier_emines_um6p_ma/ES5JKtIfEvdIlDVHJuwNGe4BOddIOPKRkmDxaL4Dfeb_zg?e=DpLHij


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


A la fin de votre analyse uploadez votre notebook dans le drive 
https://um6p-my.sharepoint.com/:f:/g/personal/alexis_perrier_emines_um6p_ma/EjFRE2wiRbJMqxp5-0QibvIBMOjqWjQINrZCpyo1ge3bnA?e=QuOdQl

**Surtout n'oubliez pas de mettre votre nom dans le nom du fichier**


Le lab est composé de plusieurs parties

## 1. charger le dataset dans une dataframe pandas

* ```df = pd.read_csv()```
* avant cela n'hesitez pas à ouvrir le fichier csv
* attention au séparateur utilisé dans le csv
* utilisez le parametres ```error_bad_line = True ``` si le fichier a du mal a etre ouvert



## 2. exploration:

* Statistiques des variables numériques ```df.describe()```
* Occurences des catégories, par exemple: ```df['DOMANIALITE'].value_counts()```
* Visualisation des variables ```df['HAUTEUR (m)'].hist(bins = 100)```

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
* Comment gérer les valeurs manquantes de cette colonne *remarquable* ? 

Quand la valeur manque, s'agit-il d'une erreur ou peut-on supposer que l'arbre n'est en fait pas *remarquable*.

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




```python
import pandas as pd
```

```python

df = pd.read_csv('les-arbres.csv', sep = ';', error_bad_lines =False)

```

```python
df.shape

```

```python
cd data/
```

```python
pwd
```

```python

```

```python
pwd
```

```python
?pd.read_csv()
```



# Titre

## sous titre

* bullet point

**en gras** et *en italique*

un [lien](http://twitter.com)




```python
df['ARRONDISSEMENT'].describe()
```

```python
df['ARRONDISSEMENT'].value_counts()
```

```python
df.columns
```

```python
print("avant {}".format(df.shape))
condition = (df['HAUTEUR (m)'] < 100) & (df['HAUTEUR (m)'] > 0)
print("apres {}".format(df[condition].shape))


```

```python

df[ df['HAUTEUR (m)'] > 500  ].shape

```

```python
df['REMARQUABLE'].value_counts( dropna = True  )
```

```python
df['REMARQUABLE'].value_counts( dropna = False  )
```

```python
vc = df['LIBELLEFRANCAIS'].value_counts()
mes_arbres = list(vc.head().keys())
```

```python
mes_arbres
```

```python
condition = df['LIBELLEFRANCAIS'].isin(mes_arbres) 
```

```python
df[condition]['LIBELLEFRANCAIS'].value_counts()
```

```python
# prendre seulement les especes qui ont plus de 4000 arbres

vc = df['LIBELLEFRANCAIS'].value_counts()

vc[vc > 4000]

```

```python
mes_arbres = vc[vc > 4000].keys()
mes_arbres
```

```python
df['ARRONDISSEMENT'].value_counts()

condition  = df['ARRONDISSEMENT'].str.contains('PARIS')

df[condition]['ARRONDISSEMENT'].value_counts()
```

```python

```
