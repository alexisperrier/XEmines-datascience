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
    ## Lab: default sur credit
    </div>
</div>

<div data-markdown>
# Programme

* Regression logistique
* odds ratio, log odds ratio
* Maximum de vraisemblance
* encoding categorical values
* Metrique de classification
* confusion matrix
* AUC and ROC
* Outliers, detection and impact
* Skewness: Box cox and Kurtosis

<img src=/assets/03/titanic_photo.jpg style='width:300px; border:0'>

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
# Logistic regression

Instead of predicting the class, we predict the probability that the outcome belongs to a class i.e.

$$ P(Y = 1/ X) $$

which we note \\( p(X) \\)

and similarly to Linear Regression we want a simple linear model for that probability

$$ P(Y=1 / X) =  p(X) = \beta_0 + \beta_1 X $$

but that still does not give us values between [0, 1]

</section>

<section data-markdown>
# Logistic regression

So instead we feed the linear model to the sigmoid function

$$ f(z) = \frac{e^{z} }{1 + e^{z}} =  \frac{1 }{1 + e^{-z}} $$


We feed \\( z = \beta\_0 + \beta\_1 X \\) to the sigmoid function

$$ p(X) = \frac{e^{\beta\_0 + \beta\_1 X} }{1 + e^{\beta\_0 + \beta\_1 X}}  $$

because this function shrinks \\( \mathbb{R} \\)  to \\( [0,1] \\)
</section>

<section data-markdown>
# Sigmoid function

![sigmoid function](/assets/03/sigmoid.svg)
</section>

<section data-markdown>
# logistique regresison en python



</section>

<section data-markdown>
# applique au default dataset

4 colonnes

* etudiant
* balance
* income

prediction : va defaulter ou non

</section>



<section data-markdown>
# Odds ratio

aussi appelé  rapport des chances, rapport des cotes1 ou risque relatif rapproché

$$ p(X) = \frac{e^{\beta\_0 + \beta\_1 X} }{1 + e^{\beta\_0 + \beta\_1 X}}  $$

This is called the **odds ratio**: ratio of the probability the event happens over the probability it does not happen
$$
\frac{p(X)}{ 1 -p(X)} = e^{\beta\_0 + \beta\_1*X}
$$

* Odds ratio  \\( \in [0, +\infty] \\)
* Odds close to 0: low probability of the event happening
* Odds close to \\( +\infty \\) : low probability of the event happening

</section>

<section data-markdown>
# Log-Odds ratio
This is called the **log-odds ratio**: the log of the odds ratio

$$
log(\frac{p(X)}{ 1 -p(X)}) = \beta\_0 + \beta\_1 X
$$

Increase in \\( \beta\_1 \\) => results in increase in \\( p(X)\\).

Not as direct and linear as in the case of linear regression.

</section>
