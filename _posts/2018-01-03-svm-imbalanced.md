---
layout: slide
title: 7) Support Vector Machines, Imbalanced datasets
description: none
transition: slide
permalink: /7-support-vector-machines
theme: white
---


<section data-markdown>
<div class=centerbox>
<p class=top>Support Vector Machines</p>
<p style ='font-size:28px;'>Imbalanced datasets</p>
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
I: Support Vector Machines
</p>
</div>
</section>



<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

# Linearly separable

We have n samples in p dimensions  $$ X = \\{ x_{i}  \\}  \quad i \in [1,n]  $$

which belong to two classes \\( y_i \in \\{-1, +1 \\} \\)


**Def:** The classes are **linearly separable** \\( \iff \\) there is an  hyperplane that fully separates the points according to their classes.


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

## Hyperplane & classification

A hyperplane in 2 dimension is defined by a line equation \\( \beta_0 + \beta_1 X_1 + \beta_2 X_2 = 0  \\)

<img src=/assets/07/hyperplane_3d.png width=300>

In **p dimensions** a hyperplane is defined by: $$\beta_0 + \beta_1 X_1 + ... + \beta_p X_p = 0 \qquad  p \gt 0 $$


    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Linear decision boundary
A classifier that is based on a separating hyperplane leads to a **linear decision boundary**.




    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# New point
We predict the class of a new point \\( x^{\prime} \\)

by calculating
$$ f(x^{\prime}) = \beta\_0+ \sum\_{j=1}^{n} \beta\_j x\_{j}^{\prime}  $$

All points \\( x_i \\) such that

* \\( f(x^{\prime})   < 0 \\) belong to **class -1**
* \\( f(x^{\prime})  > 0 \\) belong to **class +1**

## sign

Note that since both \\( y\_i \\) and \\( \beta\_0 + \sum\_{j=1}^{n} \beta\_j x\_{i,j}  \\) have the same sign:

$$ y\_i ( \beta\_0 + \sum\_{j=1}^{n} \beta\_j x\_{i,j} ) > 0 $$

    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

# Maximal Margin Classifier

We want to find the Hyperplane that will maximise the distance to all the points.
Best separation of the classes


Finding the largest margin \\(M\\) that separates the classes is equivalent to solving:

$$ \max\_{\beta\_i} M \quad  \quad y\_i ( \beta\_0 + \sum\_{j=1}^{n} \beta\_j x\_{i,j} ) > M \quad \forall i \in [1,..,n] $$

with a constraint on the coefficients \\( \sum\_{j=0}^{p}  \beta^2\_j = 1\\)

## M exists

since for all points

$$ y\_i ( \beta\_0 + \sum\_{j=1}^{n} \beta\_j x\_{i,j} ) > 0 $$

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Infinity of Hyperplanes

If our data is linearly separable then there exists an infinity of hyperplanes that can separate it

<img src=/assets/07/hyperplane-classification.png height=200>


=> We need to introduce a margin to separate the classes

<img src=/assets/07/Maximal_Margin_Classifier.png height=250>

    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Only the support vectors
Interestingly, the maximal margin hyperplane depends directly on the support vectors, but not on the other observations.

**A movement to any of the other observations would not affect the separating hyperplane**, provided that the observation’s movement does not cause it to cross the boundary set by the margin.

However changing an observation which is on one of the support vectors impacts the margin a lot => **over fitting**
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Support Vectors

The observations that are on the margin lines are called **support vectors**

They *support* the maximal margin hyperplane in the sense that if these points were moved slightly then the maximal margin hyper- plane would move as well.

<img src=/assets/07/support_vectors.png>

    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

* \\( C\\) is a non negative tuning parameter

$$ \max\_{\beta\_i, \epsilon\_i} M \quad  y\_i ( \beta\_0 + \sum\_{j=1}^{n} \beta\_j x\_{i,j} ) > M - \epsilon\_i  $$
with

* \\( \sum\_{i} \epsilon\_i \leq C  \\)

* \\( \sum\_j  \beta^2\_j = 1 \\)
* \\(   \epsilon\_i > 0 \\)

<img src=/assets/07/simple-svm.gif>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Support Vector Classifier

We allow some points to be either i) wrongly classified or ii) within the margin boundaries?

Add a tuning parameter C that dictates the **severity of the violation of the margin**. The larger C is the more points are within the margin or misclassified

Adding flexibility to the margin makes the classifier less likely to overfit.

We are now able to address datasets that are not **linearly separable**.

<img src=/assets/07/non_linearly_separable.png>

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Demo

07_svm_demo.py

    </div>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
<img src=/assets/07/sgd_loss_svm_hinge.png width=600>

* see scikit-learn doc: [1.5.7. SGD - Mathematical formulation](http://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation)

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# loss functions

### Hinge loss function

$$ L(y_i, x_i) = max(0,1−y_i(w^Tx_i+b) $$

### SGD loss function

$$ E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w) $$

### soft margin SVM tries to minimize

$$ E(w,b) = \frac{1}{n}\sum_{i=1}^{n}  max(0,1−y_i(w^Tx_i+b) + \alpha ||w||^2_2 $$


    </div>
</div>
</section>
<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Kernels

Here are some classic Kernel functions

* Linear Kernel \\( \quad K(x,x_i) = \langle x, x_i \rangle\\)
* Polynomial Kernel (d): \\( \quad K(x,x\_i) =  (1 + \sum\_{j=1}^{p} x\_{j}x\_{i,j} )^d   \\)
* Radial Kernel \\( \quad K(x,x\_i) =   \exp(-\gamma    \sum\_{j=1}^{p} (x\_{j} - x\_{i,j})^2 )   \\)


    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Introducing Kernels

The optimization equation for the linear support vector classifier can be rewritten as such

$$ f(x) =  \beta\_0 + \sum\_{i=1}^{n} \alpha\_i \langle x, x\_i \rangle   $$
$$ f(x) =  \beta\_0 + \sum\_{i=1}^{n} \alpha\_i K(x,x\_i)    $$

with \\( K(x,x\_i) = \langle x, x\_i \rangle \\) the vector dot product.

But we could use a different Kernel function

    </div>
</div>
</section>

<section data-markdown>
<div class=centerbox>
<p class=top>
II: Imbalanced datasets
</p>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

# Strategies

* Collect more data
* Accuracy is not always the best metric
    * Cohen's Kappa
    * F1 score
    * AUC
* Oversample or Undersample
* SMOTE
* Penalized Models
* Decision trees often perform well on imbalanced datasets
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Cas classique: Caravan

[Caravan dataset on kaggle](https://www.kaggle.com/uciml/caravan-insurance-challenge)

Can you predict who would be interested in buying a caravan insurance policy and give an explanation why?

* Identify potential purchasers of caravan insurance policies
* 86 variables
* Highly imbalanced
    * No     5474
    * Yes     348

### simplest classifier

* No one buys
* accuracy = 5474 / (5474+348) = 94% !

Setting all predictions to No gives you a 94% prediction rate!
    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

# SMOTE:
Oversampling method that creates new samples as interpolations of the minority class.

* Paper [Synthetic Minority Over-sampling Technique](http://www.jair.org/papers/paper953.html)
* [SMOTE in python](https://github.com/scikit-learn-contrib/imbalanced-learn)

<img src=/assets/07/SMOTE_R_visualisation_2.png>
<img src=/assets/07/SMOTE_R_visualisation_3.png>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Oversampling - subsampling
The idea is to balance the ratio of minority \\(n\_{min}\\) vs majority \\(n\_{maj}\\) samples.

$$  \frac{ n\_{min} }{n\_{maj}} \ll 1 $$

becomes

$$  \frac{ n\_{min} }{n\_{maj}} \simeq 1 $$

### Subsampling

* Randomly sub sample the majority class.

### Oversampling

* Bootstrap the minority class \\(K\\) times such that \\( K*n\_{min}  \simeq n\_{maj}\\)

    </div>
</div>
</section>

<!-- ------------------------------------------ -->
<!-- ------------------------------------------ -->
<!-- ------------------------------------------ -->
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
