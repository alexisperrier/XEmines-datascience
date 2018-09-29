---
layout: slide
title: 5)  Arbres, Random Forests et XGBoost
description: none
transition: slide
permalink: /5-arbres-random-forest-xgboost
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
* SGD
* Biais Variance

</div>
</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
I: Arbres de décision
</p>
</div>
</section>

<section data-markdown>
# Exemple arbre de décision sur Iris dataset
<img src =/assets/05/L12-tree-iris.png width=900px>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Mais

* **high overfitting** for over-complex trees that do not generalise the data well.
* Decision trees can be **unstable** because small variations in the data might result in a completely different tree being generated.
* no globally optimal decision tree

</div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Avantages

Robustes, rapides et interpretables


* Simple to understand and to interpret. Trees can be visualised.
* Requires little data preparation. (missing values, scaling, dummy variables, ...)
* Can handle both numerical and categorical data.
* Possible to validate a model using statistical tests.
* Uses a white box model. An observed situation can simply be explained by boolean logic.


    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Biais - Variance

## Deep

* Low bias, high variance
* Overfitting

## Shallow (short)

* High bias, low variance
* Underfitting


* Shallow decision trees have high bias and low variance.
* Deep decision trees have low bias and high variance.

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Lab: Controlling the tree

Lab: [Simple Decision tree](https://github.com/alexperrier/gads/blob/master/12_decision_trees/py/L12%20Simple%20Decision%20Tree%20-%20Iris%20dataset.ipynb)


Set these params to control the tree complexity

* **max_depth** (pruning): The maximum depth of the tree

* **min_samples_split**: The minimum number of samples required to split an internal node
* **min_samples_leaf**: The minimum number of samples required to be at a leaf node.
* **max_features**: The number of features to consider when looking for the best split


    </div>
</div>
</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
II: Bootstrap
</p>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Example

* mean de a

1000 fois on sample a

on tire 200 echantillons avec remplacement

    a = [1,2,3,-1,-2,-3,4,-2,-2]

    m = []
    for i in range(1000):
        m.append(np.mean(random.choice(a, size = 200, replace = True)))

    plt.boxplot(m)

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Bootstrap

**Echantilloner avec remplacement**

N samples au total, N est petit (par ex. ~< 10)

* Comment estimer la moyenne de ces echantillons ?
* Est ce que la moyenne arithmetique classique est un bon estimateur ?

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Essayer sur le titanic

sklearn.tree.DecisionTreeClassifier

* Creer les train et test sets
* comme baseline: arbre de decision simple, not pruned, quel accuracy sur le test set ?
* maintenant prendre 20 arbres, en limitant la taille a 2 niveaux
* pour chaque arbre, predire les probas des echantillons du test set
* puis moyenner les proba et utiliser le resultat pour determiner la classe predite.
* quel accuracy sur le test set ?

=> 20 arbres biaisés valent mieux qu'un arbre *non contraint* qui overfit

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Bagging for trees

Bagging stands for Bootstrap Aggregation,


* Generate B different **bootstrapped** training data sets.
* Train a new tree on each training set

The predictions of all the trees are averaged

=> significantly reduces over fitting for deep trees

=> does it also reduce bias for shallow trees ?

    </div>
</div>
</section>

<section data-markdown>
# Bagging Classifier


The key intuition of Bagging is that it reduces the variance of your model class.

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier


A **Bagging classifier** is an ensemble meta-estimator that fits **base classifiers** each on random subsets (bootstrapped) of the original dataset

The final prediction is aggregated from the models individual predictions  to form a final prediction.

* **voting**: most predicted class
* **averaging**: average of predictions (regression) or predicted probabilities (classification)

Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

* base_estimator: The base model, decision tree by default, could also be another simple n

</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
II: Random Forests
</p>
</div>
</section>


<section data-markdown>
# Random Forests

Extension of the bootstrapping to features

* In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set.

* In addition, when splitting a node during the construction of the tree, the split that is picked is the best split among **a random subset of the features**.

=> The bias of the forest usually slightly increases (with respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than compensating for the increase in bias, hence yielding an overall better model.

(see RandomForestClassifier and RandomForestRegressor classes),

A Random Forest is a generalization of Bagging that is specific to DTs. At each branch in the decision tree, Random Forest training also subsamples the features in addition to the training examples. Intuitively, this process further de-correlates the individual trees, which is good for Bagging, since the main limitation of Bagging is that bootstrapping is not the same as drawing fresh samples from the true data distribution.

</section>



<section data-markdown>
# Out Of Bag - OOB

When boostrapping, in each experiment will use only approx. 2/3rd of the available samples.

Which leaves 1/3rd that we can use to estimate the validation error of each tree.

This is called OOB Out of Bag error.

It can be shown that with B sufficiently large, OOB error is virtually equivalent to leave-one-out cross-validation error.

</section>

<section data-markdown>
# Feature importance

* When the *max_features < total number of features*.

    => Some features are left out of the splitting decision in each node.

* Relative Feature importance can be deduced from the delta in MSE associated to the features included vs left out.

</section>


<section data-markdown>
# Titanic

Quelles sont les variables les plus importantes ?

# Cars

Quelles sont les variables les plus importantes ?


etc ...
</section>



<section data-markdown>
<div class=centerbox>
<p class=top>
III: XGBoost
</p>
</div>
</section>




<section data-markdown>
# Adaboost



</section>


<section data-markdown>
# Gradient boosting

Keep an overall predictor that is the (weighted) average of a bunch of models.
Train first model on original training data, and initialize overall predictor as just this single model.
Assess the error of the the overall predictor and modify the training data the focus on areas of high error.
For AdaBoost, this means re-weighting the data points so that poorly modeled data points get higher weight.
For Gradient Boosting, this means redefining the supervised prediction target to be some kind of residual between the ground truth and the overall predictor.
Train a new model on the modified training data, and add to the overall predictor.
Repeat Steps 3 & 4.

A Gradient Boosting will take a different approach. It will start with a (usually) not very deep tree (sometimes a decision stump - a decision tree with only one split) and will model the original target. Then it takes the errors from the first round of predictions, and passes the errors as a new target to a second tree. The second tree will model the error from the first tree, record the new errors and pass that as a target to the third tree. And so forth. Essentially it focuses on modelling errors from previous trees. GB is one of the best algorithms available today and it’s almost always outperforming RF on most datasets I’ve tried.

Notice how RF runs trees in parallel, thus making it possible to parallelize jobs on a multiprocessor machine. GB instead uses a sequential approach.

</section>




<section data-markdown>

Random forests bags models, while boosting iteratively averages them with respect to error. XGBoost extends boosting by imposing regression penalties similar to elastic net.


One can interpret boosting as trying to minimize the bias of the overall predictor. So when you use boosting, you’re incentivized to use shallow decision trees because they have low variance and high bias. Using high variance base models in boosting runs a much higher risk of overfitting than approaches like Bagging.
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
