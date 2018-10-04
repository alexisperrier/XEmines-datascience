---
layout: slide
title: 6) Texte & Word2Vec
description: none
transition: slide
permalink: /6-text-classification
theme: white
---


<section data-markdown>
<div class=centerbox>
<p class=top> Arbres, Random Forests et XGBoost</p>
<p style ='font-size:28px;'>Ensembling, Bagging, Boosting</p>
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

</div>
</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
I: Text Mining - NLP
</p>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Corpus

## texte brut

* forums, réseaux sociaux (peu structuré)
* plus structuré: discours, news, articles, emails, ...
* plus ou moins long: livres, articles scientifiques, abstracts, ...

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Applications

Simple et directe

* prediction, classification, identification
    * binaire: spam
    * multiclass: sujet du document

Non supervisée

* topic modeling

Avancée: productive

* Résumé
* Traduction automatique
* Chatbots

voir les nouvelles fonctionnalités de gmail

Interpretative

*  Sentiment analysis

    </div>
</div>
</section>



<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Librairies python

* spacy.io
* nltk
* gensim



Nombreuses librairies open source en R, Java, ...

# Resources

* Livre: Speech and Language Processing https://web.stanford.edu/~jurafsky/slp3/

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Cloud

* AWS Comprehend
* Google NLP
* Speech to text

# Arabic

* Stanford NLP: https://nlp.stanford.edu/projects/arabic.shtml
* Deep learning for Arabic NLP https://www.sciencedirect.com/science/article/pii/S1877750317303757


    </div>
</div>
</section>


<section>

<div style='float:right; width:30%;  '>
    <div data-markdown>
# Chomsky
<img src=/assets/06/Syntactic_Structures_Noam_Chomsky_cover.jpg>

1957
    </div>
</div>


<div style='float:left; width:30%;  '>
    <div data-markdown>
# Ferdinand de Saussure
<img src=/assets/06/saussure.jpg>

1916

    </div>
</div>

<div style='float:right; width:30%;  '>
    <div data-markdown>

# Benveniste
<img src=/assets/06/Benveniste-Emile-Problemes-De-Linguistique-Generale-Livre-684848349_L.jpg>
1966

    </div>
</div>

</section>


<section data-markdown>
# Text generation

* https://ml5js.org/docs/lstm-interactive-example

</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Transformations

* lemmatization,

    * la voiture est grande
    * Je suis sur un grand bateau

    * est, suis => etre
    * grande, grand => grand


* tokens, bi-grams
* stopwords: je, tu, il, et, me, sa, son, mais, donc, par, ....


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Numeriser le texte

Comment passer d'un texte libre a une matrice numérique ?

Approche **Bags of words**

## Tf-idf

Pour un mot donné dans un corpus de plusieurs documents

* Fréquence dans un document / frequence des mots dans les autres documents

* tf-idf means term-frequency times inverse document-frequency

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Cosine distance
<img src=/assets/06/cosine_similarity.png>

# Spacy
https://spacy.io/usage/vectors-similarity

    from gensim.models import Word2Vec

    #loading the downloaded model
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

    #the model is loaded. It can be used to perform all of the tasks mentioned above.

    # getting word vectors of a word
    banana = model['banana']

    #performing king queen magic
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))

    #picking odd one out
    print(model.doesnt_match("breakfast cereal dinner lunch".split()))

    #printing similarity index
    print(model.similarity('apple', 'orange'))
    print(model.similarity('car', 'orange'))

see word2vec_demo.py
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
## Word2vec et Glove

* Approche très recente qui associe un vecteur de grande dimension (128, 256, ...) a des milliers de mots

* Comme on a des vecteurs on a une distance entre les mots. Cosine distance

* Corpus original: Wikipedia

* Capture du *sens* du mot

    * Reine - femme = Roi - homme
    * Rabat - capitale = Paris - capitale




    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Word2vec
Word2vec is not a single algorithm but a combination of two techniques – CBOW(Continuous bag of words) and Skip-gram model.

* Skip – gram : to predict the context given a word
* CBOW tends to predict the probability of a word given a context
* word2vec is a "predictive" model, predict word / context + context / word

http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf

### see also Glove

GloVe is a "count-based" model :Dimensionality reduction on the co-occurrence counts matrix.

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
<img src=/assets/06/skip_gram_net_arch.png>

Predictive models learn their vectors in order to improve their predictive ability of Loss(target word | context words; Vectors), i.e. the loss of predicting the target words from the context words given the vector representations. In word2vec, this is cast as a feed-forward neural network and optimized as such using SGD,

    </div>
</div>
</section>




<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

* http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# TF-IDF - sklearn

The most intuitive way to do so is to use a bags of words representation:

Assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).

For each document #i, count the number of occurrences of each word w and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary.

    </div>
</div>
</section>


<section data-markdown>
# Topic modeling

<img src=/assets/06/lsa_decomposition_example_03.png>
</section>


<section data-markdown>
<div class=centerbox>
<p class=top>
II: Lab : text classification sur bbc dataset
</p>
</div>
</section>
