--------------------------------------------------------------------
# Day 1: Intro et python
--------------------------------------------------------------------
## Cours
* Intro du cours et presentations
* Data science, applications, ML
http://127.0.0.1:4000/01-what-is-data-science.html#/17 pour l'histoire
http://127.0.0.1:4000/01-what-is-data-science.html#/10 et suivantes

http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/20

* Analyse predictive supervisée, fondements
* non supervisée
* black box vs stats
http://127.0.0.1:4000/01-what-is-data-science.html#/17
Statistical Modeling: The Two Cultures

* Python, conda
http://127.0.0.1:4000/01-what-is-data-science.html#/51
* python ecosystem
http://127.0.0.1:4000/04-statistics-review.html#/8
* numpy
http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/33
http://127.0.0.1:4000/04-statistics-review.html#/7
* pandas
http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/35

>> Engie formation
>> UPEM

## Lab
* jupyter notebook
* rappel de python
UPEM
* numpy
    http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/33
https://github.com/alexisperrier/gads/blob/master/02_research_design_and_pandas/py/L2%20Numpy.ipynb

* pandas & titanic
https://github.com/alexisperrier/gads/blob/master/02_research_design_and_pandas/py/L2%20Pandas.ipynb

* pandas on ozone
https://github.com/alexisperrier/gads/blob/master/02_research_design_and_pandas/py/L2%20Pandas-Lab.ipynb

* Install Python 3.6+ and Anaconda
http://127.0.0.1:4000/01-what-is-data-science.html#/51
* Practice Python syntax, Terminal commands, and Pandas
* iPython Notebook test and Python review

* Exploration of the Online Retail dataset¶
https://github.com/alexisperrier/gads/blob/master/01_what_is_a_data_scientist/py/L1%20Online%20Retail.ipynb
data: https://github.com/alexisperrier/gads/tree/master/01_what_is_a_data_scientist/data

code organization
http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/31

# Before next class
http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/42

# further
https://www.python-course.eu/index.php
http://pandas.pydata.org/
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://github.com/jvns/pandas-cookbook
https://bitbucket.org/hrojas/learn-pandas
Kaggle: https://www.kaggle.com/c/titanic
data types: http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/03-data-types-and-format.html

--------------------------------------------------------------------
# Day 2: Regression lineaire
--------------------------------------------------------------------
## Cours
* Regression vs Classification
http://127.0.0.1:4000/03-statistics-fundamentals.html#/26
    * classification http://127.0.0.1:4000/08-classification-knn.html#/6
* Regression linéaire
http://127.0.0.1:4000/04-statistics-review.html#/16
    * math: http://127.0.0.1:4000/06-scikit-linear-regression.html#/1
    to http://127.0.0.1:4000/06-scikit-linear-regression.html#/17 and after
    * loss function http://127.0.0.1:4000/06-scikit-linear-regression.html#/27
    * RMSE http://127.0.0.1:4000/06-scikit-linear-regression.html#/24
    * OLS http://127.0.0.1:4000/06-scikit-linear-regression.html#/14
    * interpretation voir aussi notebook https://github.com/alexisperrier/gads/blob/master/04_statistics_inference/py/Lesson%204%20-%20Notebook%202%20-%20Linear%20Regression%20for%20Causal%20Inference.ipynb
    * anscombe quartet http://127.0.0.1:4000/04-statistics-review.html#/17
* causation correlation http://127.0.0.1:4000/04-statistics-review.html#/13
    * http://127.0.0.1:4000/04-statistics-review.html#/20
    * http://127.0.0.1:4000/04-statistics-review.html#/21
* hypotheses
    * http://127.0.0.1:4000/04-statistics-review.html#/22
    * http://people.duke.edu/~rnau/testing.htm
    * http://www.statisticssolutions.com/assumptions-of-linear-regression/
* confouding
http://127.0.0.1:4000/04-statistics-review.html#/23
* p-value
http://127.0.0.1:4000/04-statistics-review.html#/33
http://127.0.0.1:4000/04-statistics-review.html#/34
http://127.0.0.1:4000/04-statistics-review.html#/35

* Linearité

* Non Linearité

* Regression polynomiale
    * https://github.com/alexisperrier/gads/blob/master/06_scikit_regression/py/L6%20Polynomial%20regression%20-%20Solutions.ipynb

* Kaggle projet

* correlation
http://127.0.0.1:4000/03-statistics-fundamentals.html#/13
http://127.0.0.1:4000/03-statistics-fundamentals.html#/14
* Confounders

## Lab
* data dictionnaries
http://127.0.0.1:4000/02-experimental-design-and-pandas.html#/30

### regression lineaire
* https://github.com/alexisperrier/gads/blob/master/04_statistics_inference/py/Lesson%204%20-%20Notebook%202%20-%20Linear%20Regression%20for%20Causal%20Inference.ipynb

impact des confounder
https://www.r-bloggers.com/how-to-create-confounders-with-regression-a-lesson-from-causal-inference/

Ordinary Least Squares in Python on datarobot

https://blog.datarobot.com/ordinary-least-squares-in-python

* lab: explorer les donnees du projet Kaggle

## resources
http://www.tylervigen.com/spurious-correlations
https://en.wikipedia.org/wiki/Anscombe%27s_quartet
QQ plot https://en.wikipedia.org/wiki/Q–Q_plot#Interpretation

confoudning: http://www.statisticshowto.com/experimental-design/confounding-variable/
and whole list here
http://127.0.0.1:4000/04-statistics-review.html#/44

L6 Linear Regression Algebra.ipynb:
https://github.com/alexisperrier/gads/blob/master/06_scikit_regression/py/L6%20Linear%20Regression%20Algebra.ipynb

--------------------------------------------------------------------
# Day 3: Regression logistique
--------------------------------------------------------------------
## Cours
* regression not adapted to classifctaion http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/6
* Regression logistique
    * explication http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/10
    * avantage: Advantages of Logistic Regression: Lots of ways to regularize your model, and you don’t have to worry as much about your features being correlated, like you do in Naive Bayes. You also have a nice probabilistic interpretation, and you can easily update your model to take in new data, Use it if you want a probabilistic framework  or if you expect to receive more training data in the future that you want to be able to quickly incorporate into your model.
* odds ratio, log odds ratio
    * http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/13
* Maximum de vraisemblance
    * https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
* Metrique de classification
    *  http://127.0.0.1:4000/08-classification-knn.html#/10
    * http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/19 and following
    * confusion matrix http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/23
    * threshold http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/26
    * AUC and ROC http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/27
* Outliers, detection and impact

* Skewness: Box cox and Kurtosis
    * https://github.com/alexisperrier/gads/blob/master/03_statistics_fundamentals/py/lesson-3-homework.ipynb
    * http://127.0.0.1:4000/03-statistics-fundamentals.html#/32
    * http://127.0.0.1:4000/03-statistics-fundamentals.html#/33
    * normal distribution http://127.0.0.1:4000/03-statistics-fundamentals.html#/29
    box cox http://127.0.0.1:4000/03-statistics-fundamentals.html#/38

## Lab: online retail
https://github.com/alexisperrier/gads/blob/master/03_statistics_fundamentals/py/One%20Hot%20Encoding.ipynb
https://github.com/alexisperrier/gads/blob/master/03_statistics_fundamentals/py/Explore%20Online%20Retail%20Lab%20-%20Solutions.ipynb
https://github.com/alexisperrier/gads/blob/master/03_statistics_fundamentals/py/Normal%20Distribution.ipynb
https://github.com/alexisperrier/gads/blob/master/03_statistics_fundamentals/py/Statistics%20Fundamentals.ipynb

http://127.0.0.1:4000/03-statistics-fundamentals.html#/20
http://127.0.0.1:4000/03-statistics-fundamentals.html#/21

http://127.0.0.1:4000/03-statistics-fundamentals.html#/42

* Classification with Logistic Regression on the Default dataset
https://github.com/alexisperrier/gads/blob/master/09_logistic_regression/py/L9-logistic-regression.py
https://github.com/alexisperrier/gads/blob/master/09_logistic_regression/py/Default%20dataset%20-%20Logistic%20regression.ipynb
https://github.com/alexisperrier/gads/blob/master/09_logistic_regression/py/LR%20with%20Cross%20validation%20and%20Grid%20Search%20on%20Default%20dataset.ipynb
Q: http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/17

admission dataset 2 cont + 1 cat => bnary classfication
* https://github.com/alexisperrier/gads/blob/master/04_statistics_inference/py/Lesson%204%20-%20Notebook%201%20-%20Linear%20Regression.ipynb


## Resources
http://onlinestatbook.com/2/transformations/box-cox.html
http://www.neural.cz/dataset-exploration-boston-house-pricing.html
http://seaborn.pydata.org/tutorial/distributions.html
skewness impact on regression lineaire
https://www.quora.com/How-does-skewness-impact-regression-model
--------------------------------------------------------------------
# Day 4: sklearn, SGD, biais variance
--------------------------------------------------------------------
## Cours
* Scikit
http://127.0.0.1:4000/06-scikit-linear-regression.html#/7
    * related projects http://scikit-learn.org/stable/related_projects.html
    * API http://127.0.0.1:4000/06-scikit-linear-regression.html#/11

* proeprocessing with scikit
    http://127.0.0.1:4000/09-classification-metrics-logistic-regression.html#/31
    * One Hot Encoding http://127.0.0.1:4000/03-statistics-fundamentals.html#/41
    http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

* Biais Variance
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/9
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/11
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/12
* Data Split
    train, test, valid http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/18
    k-fold http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/20
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/21
    * stratified k-fold http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/22


* SGD
## Lab
* model selection with train test validation split
https://github.com/alexisperrier/gads/blob/master/07_bias_variance/py/L7%20train%20validation%20test.ipynb

* k-fold On the diabetes dataset, find the optimal regularization parameter alpha.
http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/23

## Resources
* one hot encoding vs labelencoder

http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor
--------------------------------------------------------------------
# Day 5: Lasso, Ridge, overfitting, Perceptron, Adaboost
--------------------------------------------------------------------
## Cours
* Regularisation

* Lasso Ridge
    * Norm L1, L2, distance de minkowsky http://127.0.0.1:4000/08-classification-knn.html#/20
    * ridge http://127.0.0.1:4000/06-scikit-linear-regression.html#/28
    * lasso http://127.0.0.1:4000/06-scikit-linear-regression.html#/30
    * elastic http://127.0.0.1:4000/06-scikit-linear-regression.html#/31
* Overfitting: detection et remedes
    * polynomial regression: intro to overfit
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/6
    * underfitting overfiffitn biais variance
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/15
    * learning curves
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/33
    http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py
    * remedies
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/38
    http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/39

* Perceptron
* Adaboost
* Boxplot
boxplot http://127.0.0.1:4000/03-statistics-fundamentals.html#/16
http://127.0.0.1:4000/03-statistics-fundamentals.html#/17
http://127.0.0.1:4000/03-statistics-fundamentals.html#/18


* Normalisation

* regression lineaire: overfit
https://towardsdatascience.com/predicting-housing-prices-using-advanced-regression-techniques-8dba539f9abe
Regularized models perform well on this dataset. A note on the bias/variance tradeoff: According to the Gauss-Markov theorem, the model fit by the Ordinary Least Squares (OLS) is the least biased estimator of all possible estimators. In other words, it fits the data it has seen better than all possible models.

It does not necessarily perform well, however, against data that it has not seen. A regularized model penalizes model complexity by limiting the size of the betas. The effect of this is that the model introduces more bias than the OLS model, but becomes more statistically stable and invariant. In other words, it prevents us from overfitting and is better able to generalize to new data.

## Lab

OLS vs Ridge
https://github.com/alexisperrier/gads/blob/master/06_scikit_regression/py/L6%20OLS%20vs%20Ridge.ipynb

Ridge coefficients as a function of the L2 regularization
https://github.com/alexisperrier/gads/blob/master/06_scikit_regression/py/L6%20Ridge%20coefficients%20as%20a%20function%20of%20the%20L2%20regularization.ipynb

* learning curve on housing dataset
https://github.com/alexisperrier/gads/blob/master/07_bias_variance/py/Learning%20Curve%20on%20the%20Housing%20Dataset.ipynb
http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/36

## Resources
https://www.coursera.org/lecture/machine-learning/learning-curves-Kont7
http://www.ultravioletanalytics.com/2014/12/12/kaggle-titanic-competition-part-ix-bias-variance-and-learning-curves/
http://www.astroml.org/sklearn_tutorial/practical.html

Regularization in Logistic Regression: Better Fit and Better Generalization?
By Sebastian Raschka
https://github.com/rasbt/python-machine-learning-book/blob/master/faq/regularized-logistic-regression-performance.md

--------------------------------------------------------------------
# Day 6:
--------------------------------------------------------------------
## Cours
* Naive Bayes
    see http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/
    Advantages of Naive Bayes: Super simple, you’re just doing a bunch of counts. If the NB conditional independence assumption actually holds, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data. And even if the NB assumption doesn’t hold, a NB classifier still often does a great job in practice. A good bet if want something fast and easy that performs pretty well. Its main disadvantage is that it can’t learn interactions between features (e.g., it can’t learn that although you love movies with Brad Pitt and Tom Cruise, you hate movies where they’re together).


* Text
    * NLP
    * NLP in scikit
    http://127.0.0.1:4000/13-nlp.html#/5
    * Tokenize: http://127.0.0.1:4000/13-nlp.html#/7
    * Bag of Words http://127.0.0.1:4000/13-nlp.html#/10
    * tf-idf http://127.0.0.1:4000/13-nlp.html#/17
    * hashing trick http://127.0.0.1:4000/13-nlp.html#/20
    * Naive Bayes and Text Classification https://sebastianraschka.com/Articles/2014_naive_bayes_1.html
* Dataviz
    * matplotlib http://127.0.0.1:4000/05-data-visualization-matplotlib-seaborn.html#/8
    * better fiugures http://127.0.0.1:4000/05-data-visualization-matplotlib-seaborn.html#/13
    * Tufte http://127.0.0.1:4000/05-data-visualization-matplotlib-seaborn.html#/14
    * exemples:
        * space over time https://www.technologyreview.com/s/425120/space-over-time/
    * pie charts http://127.0.0.1:4000/05-data-visualization-matplotlib-seaborn.html#/19
    * seaborn http://127.0.0.1:4000/05-data-visualization-matplotlib-seaborn.html#/25
* plotly: http://127.0.0.1:4000/05-data-visualization-matplotlib-seaborn.html#/31

## Lab
* matplotlib: https://github.com/alexisperrier/gads/blob/master/05_visualization/py/Lesson%205%20Notebook%201%20Matplotlib.ipynb
https://github.com/alexisperrier/gads/blob/master/05_visualization/py/Lesson%205%20Notebook%202%20More%20Matplotlib.ipynb
* tips dataset with seaborn https://github.com/alexisperrier/gads/blob/master/05_visualization/py/Lesson%205%20Notebook%203%20-%20Seaborn.ipynb

* input text in scikit
https://github.com/alexisperrier/gads/blob/master/13-nlp/py/L13%20N1%20Feature%20Extraction.ipynb

* Sentiment prediction on IMDB reviews
https://github.com/alexisperrier/gads/blob/master/13-nlp/py/L13%20N2%20Sentiment%20Prediction.ipynb

* Classification on the Twenty Newsgroup¶
 https://github.com/alexisperrier/gads/blob/master/13-nlp/py/L13%20N3%20Text%20Classification.ipynb

## resources
https://matplotlib.org/
matplotlib by randal olson http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/

--------------------------------------------------------------------
# Day 7: Arbres et Foret
--------------------------------------------------------------------
## Cours
* Arbres de decision
    Advantages of Decision Trees: Easy to interpret and explain (for some people – I’m not sure I fall into this camp). They easily handle feature interactions and they’re non-parametric, so you don’t have to worry about outliers or whether the data is linearly separable (e.g., decision trees easily take care of cases where you have class A at the low end of some feature x, class B in the mid-range of feature x, and A again at the high end).  Plus, random forests are often the winner for lots of problems in classification, they’re fast and scalable, and you don’t have to worry about tuning a bunch of parameters, so they seem to be quite popular these days.
* decision trees
    * http://127.0.0.1:4000/12-decision-trees-random-forests.html#/21
    http://127.0.0.1:4000/12-decision-trees-random-forests.html#/22
    http://127.0.0.1:4000/12-decision-trees-random-forests.html#/25
* Ginir vs MSE classification vs regression http://127.0.0.1:4000/12-decision-trees-random-forests.html#/26
* Boostrapping
* Bagging

    * http://127.0.0.1:4000/12-decision-trees-random-forests.html#/28
    * bagging in scikit: http://127.0.0.1:4000/12-decision-trees-random-forests.html#/31
* OOB
    * http://127.0.0.1:4000/12-decision-trees-random-forests.html#/29
* Random Forest
    * http://127.0.0.1:4000/12-decision-trees-random-forests.html#/27
    http://127.0.0.1:4000/12-decision-trees-random-forests.html#/34
* Ensembling
    * http://127.0.0.1:4000/07-sampling-bias-variance-sgd.html#/27

* Boosting
* XGBoost
https://medium.com/analytics-vidhya/an-end-to-end-guide-to-understand-the-math-behind-xgboost-72c07acb4afb

* Imbalanced dataset
    * http://127.0.0.1:4000/11-svm-unbalanced.html#/21
    * under sample / over sample
    * SMOTE http://127.0.0.1:4000/11-svm-unbalanced.html#/23

## Lab
* bootstrapping https://github.com/alexisperrier/gads/blob/master/07_bias_variance/py/L7%20Boostrapping%20the%20mean.ipynb
* bootstrapping for model selection
Using Scikit's Bootstrap or resample, estimate the distribution of your linear regression coeffficients
* Bagging, Boosting, ...
https://github.com/alexisperrier/gads/blob/master/12_decision_trees/py/L12%20Bagging.ipynb
* Caravan
    * http://127.0.0.1:4000/11-svm-unbalanced.html#/25
    http://127.0.0.1:4000/11-svm-unbalanced.html#/26
    http://127.0.0.1:4000/11-svm-unbalanced.html#/28
    http://127.0.0.1:4000/11-svm-unbalanced.html#/30
     https://github.com/alexisperrier/gads/blob/master/11_svm_imbalanced/py/L11_N3_Imbalanced.py
     RF on caravan: https://github.com/alexisperrier/gads/blob/master/12_decision_trees/py/L12%20Random%20Forests.ipynb
# resources
* Learning from Imbalanced Classes
 http://www.svds.com/learning-imbalanced-classes/
* Combat Imbalanced Classes in Your Machine Learning Dataset https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

--------------------------------------------------------------------
# Day 8: SVM
--------------------------------------------------------------------

=> excellent sur la loss function https://stats.stackexchange.com/questions/74499/what-is-the-loss-function-of-hard-margin-svm


## Cours
* SVM
    * http://127.0.0.1:4000/11-svm-unbalanced.html#/5
    * http://127.0.0.1:4000/11-svm-unbalanced.html#/9
    * http://127.0.0.1:4000/11-svm-unbalanced.html#/12
    http://127.0.0.1:4000/11-svm-unbalanced.html#/24
* Advantages of SVMs: High accuracy, nice theoretical guarantees regarding overfitting, and with an appropriate kernel they can work well even if you’re data isn’t linearly separable in the base feature space. Especially popular in text classification problems where very high-dimensional spaces are the norm. Memory-intensive, hard to interpret, and kind of annoying to run and tune, though, so I think random forests are starting to steal the crown.
* Kernels
http://127.0.0.1:4000/11-svm-unbalanced.html#/17

* How to choose a predictive model https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms
http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/

* Data processing
    * normalize, missing values,
    http://127.0.0.1:4000/08-classification-knn.html#/22
* Curse of dimension
    * illustate with
    * http://127.0.0.1:4000/08-classification-knn.html#/28 https://github.com/alexisperrier/gads/blob/master/08_classification_knn/py/l8_N5_curse.py
    * http://127.0.0.1:4000/08-classification-knn.html#/30

* Dimension reduction  PCA Vs. LDA
    * PCA : http://127.0.0.1:4000/10-unsupervised-learning.html#/17
        http://127.0.0.1:4000/10-unsupervised-learning.html#/18
    * http://127.0.0.1:4000/08-classification-knn.html#/25
    * http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
    * decorrelate with PCA
    http://www.visiondummy.com/2014/05/feature-extraction-using-pca/
    * LDA https://sebastianraschka.com/Articles/2014_python_lda.html

* feature selection

## Lab
* dimension reduction: https://github.com/alexisperrier/gads/blob/master/08_classification_knn/py/L8%20N4%20PCA.py
* chain PCA and predictions
http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py

* How many eigenvalues to retain 80% of the variation in the data?
http://127.0.0.1:4000/10-unsupervised-learning.html#/21
and http://127.0.0.1:4000/10-unsupervised-learning.html#/22 on another dataset

* SVM SVC
    * https://github.com/alexisperrier/gads/blob/master/11_svm_imbalanced/py/L11_N2_SVM.py
    * https://github.com/alexisperrier/gads/blob/master/11_svm_imbalanced/py/L11%20N1%20SVC.py

## resources
The Curse of Dimensionality in classification

http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/

--------------------------------------------------------------------
# Day 9: Time series ARIMA
--------------------------------------------------------------------
## Cours
* Series temporelles
    * http://127.0.0.1:4000/16-time-series.html#/4
    * fourier http://127.0.0.1:4000/16-time-series.html#/7
* most simple predictor http://127.0.0.1:4000/16-time-series.html#/14
* simple forecasting
http://127.0.0.1:4000/17-time-series-forecast.html#/6
* white noise http://127.0.0.1:4000/17-time-series-forecast.html#/9
and test ljung box
* random walk http://127.0.0.1:4000/17-time-series-forecast.html#/22
* is the DJ a random walk http://127.0.0.1:4000/17-time-series-forecast.html#/23
* residual diagnostic  http://127.0.0.1:4000/17-time-series-forecast.html#/13
* metriques http://127.0.0.1:4000/16-time-series.html#/15
* MA, Exp MA http://127.0.0.1:4000/16-time-series.html#/18
* autocorrelation http://127.0.0.1:4000/16-time-series.html#/20
* stationnarity http://127.0.0.1:4000/16-time-series.html#/23
http://127.0.0.1:4000/17-time-series-forecast.html#/3
* stationnarity tests http://127.0.0.1:4000/16-time-series.html#/28
* making a ts stationary http://127.0.0.1:4000/16-time-series.html#/32
* AR http://127.0.0.1:4000/17-time-series-forecast.html#/24
* MA http://127.0.0.1:4000/17-time-series-forecast.html#/26
* ARIMA (pd,d,q)
    * order http://people.duke.edu/~rnau/411arim2.htm
    http://people.duke.edu/~rnau/411arim3.htm
    * http://127.0.0.1:4000/17-time-series-forecast.html#/27
* normality test http://127.0.0.1:4000/17-time-series-forecast.html#/30
* Decomposition
    * components http://127.0.0.1:4000/16-time-series.html#/9
    * ADDITIVE MODEL
: http://127.0.0.1:4000/17-time-series-forecast.html#/15

## Lab
A Simple Time Series Analysis Of The S&P 500 Index
https://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/

* Smoothing and Forecasting https://github.com/alexisperrier/gads/blob/master/16_time_series/py/Smoothing%20and%20Forecast%20101.ipynb

* Dickey Fuller
https://github.com/alexisperrier/gads/blob/master/17_ts2/py/L17%20Time%20Series%20Demo.ipynb

Seasonal ARIMA with Python
* http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

* Time Series Analysis using iPython
 sunpots https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/

* IBM dataset
http://127.0.0.1:4000/17-time-series-forecast.html#/17
* house sales http://127.0.0.1:4000/17-time-series-forecast.html#/18

## Resources
* ts resources :https://datamarket.com/data/list/?q=provider:tsdl

--------------------------------------------------------------------
# Day 10: Projet Kaggle
--------------------------------------------------------------------
