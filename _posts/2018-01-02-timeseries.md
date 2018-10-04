---
layout: slide
title: 8) Time series
description: none
transition: slide
permalink: /8-time-series
theme: white
---

<section data-markdown>
<div class=centerbox>
<p class=top>Time series</p>
<p style ='font-size:28px;'></p>
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
I: Time series, intro
</p>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
<img src=/assets/08/time-series-analysis.png>
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Today
* Metrics
* Simple, MA, EWMA
* Forecasting, train, test
* Stationnarity, definition, Dickey Fuller test
* Decomposition: Trend, cycle, seasonality
* Differencing
* ARIMA
    * autocorrelation et partial autocorrelation
* residuals

    </div>
</div>
</section>



<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Domaines d'application

* **IoT**
* **econometrics**
* mathematical finance / trading / markets
* intelligent transport and trajectory forecasting
* weather forecasting and Climate change research
* earthquake prediction, astronomy
* electroencephalography, control engineering, communications
* **signal processing**

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Time Series [TS]

* A time series is a series of data points listed in time order.
* A sequence taken at successive equally spaced points in time.
* A sequence of **discrete-time data**.

=> The **time interval** is key
    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
<img src=/assets/08/monthly_milk_production.png>

<img src=/assets/08/prediction_with_prophet_01.png>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
<img src=/assets/08/ts-cheese.png>

<img src=/assets/08/TimeSeriesChart_1.jpg>

    </div>
</div>
</section>



<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Ts metrics


Metrics to compare TS techniques

* Mean Absolute Error:

\\( MAE=mean(|e\_i|) \\)

* [Mean Absolute Deviation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mad.html):

\\( MAD = \frac{1}{n} \sum\_{i=1}^{n}  | e\_i |  \\)

* Root mean squared error:

\\( RMSE = \sqrt{  mean(e\_i^2)  } \\)

* Mean absolute percentage error:

\\( MAPE = \frac{1}{n} \sum\_{i=1}^{n} \frac{ | e\_i | }{ y\_i } \\)

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Prévision la plus simple

* Il fera aujourd'hui le meme temps qu'hier

$$ \hat{y}\_{n+1} = y\_{n} $$

## Forecasting error

$$ e\_i=y\_i−\hat{y}\_i $$

    </div>
</div>
</section>



<section data-markdown>
<div class=centerbox>
<p class=top>
II: Modélisation
</p>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Smoothing
<img src=/assets/08/moving-avg-2.png>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Moving Average

Simple smoothing, moyenne sur fenetre

$$  \hat{y}\_{t +1 }   = \frac{1}{n} \sum\_{i=0}^{n-1}y\_{t-i}
= \frac{1}{n} ( y\_{t} + y\_{t-1}+ \cdots + y\_{t-(n-1)}  )
$$


## Avec coefficients

Modèle MA(q): Moving Average d'ordre q

$$  \hat{y}\_{t +1 }   =  \sum\_{i=0}^{q-1} \beta\_{i}  y\_{t-i} $$


    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

<img src=/assets/08/exponential_moving_averages.gif>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Exponential Weighted Moving Average
Introduce a Decay

The EWMA for a series Y may be calculated recursively:

* \\( S\_{1}=Y\_{1} \\)
* \\(  S\_{t}=\alpha  Y\_{t}+(1-\alpha ) S\_{t-1}  \ \ \ \  t>1 \\)


Where:

* The coefficient α represents the degree of weighting decrease, a constant smoothing factor between 0 and 1.
* \\(Y\_t\\) is the value at a time period t.
* \\(S\_t\\) is the value of the EWMA at any time period t.

A higher \\( \alpha \\) discounts older observations faster.

http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html


    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

# ARMA(p,q) model
The notation ARMA(p, q) refers to the model with p autoregressive terms and q moving-average terms. This model contains the AR(p) and MA(q) models,

$$ X\_{t}= c + \varepsilon\_{t} + \sum\_{i=1}^{p} \varphi\_{i} X\_{t-i} + \sum\_{i=1}^{q} \theta\_{i} \varepsilon\_{t-i} $$

# ARIMA(p,d,q)

Difference entre evenements successifs.

On considère le processus différencié (d=1):

$$ y\_{t}'=y\_{t}-y\_{t-1} $$

ou second order differencing (d=2):

$$ \begin{aligned}
y\_{t}^{\*} & = y\_{t}'-y\_{t-1}' \\\
y\_{t}^{\*} & = (y\_{t}-y\_{t-1})-(y\_{t-1}-y\_{t-2}) \\\
y\_{t}^{\*} & = y\_{t} - 2y\_{t-1} + y\_{t-2}
\end{aligned} $$

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>


# Autoregressive model

The notation AR(p) refers to the autoregressive model of order p. The AR(p) model is written

$$ X\_{t}=c + \sum\_{i=1}^{p} \varphi\_{i}X\_{t-i} + \varepsilon\_{t} $$

## polynome associé

$$ AR(z) =  z^{p} - \sum\_{i=1}^{p} \varphi\_{i}z^{p-i} $$



# Moving-average model

The notation MA(q) refers to the moving average model of order q:

$$ X\_{t}=\mu +\varepsilon\_{t}+\sum\_{i=1}^{q}\theta\_{i}\varepsilon\_{t-i} $$

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

* An ARIMA(0,1,0) model (or I(1) model) is given by \\( X\_{t}=X\_{t-1} + \varepsilon\_{t} \\)  which is simply a random walk.

* An ARIMA(0,1,0) with a constant, given by  \\( X\_{t}=X\_{t-1} + c + \varepsilon\_{t} \\) — which is a random walk with drift.

* An ARIMA(0,0,0) model is a white noise model.

* An ARIMA(0,1,1) model without constant is a basic exponential smoothing model

* An ARIMA(0,2,2) model is given by

$$ X\_{t} = 2X\_{t-1} - X\_{t-2} + ( \alpha + \beta -2) \varepsilon\_{t-1} + (1 - \alpha ) \varepsilon\_{t-2} + \varepsilon\_{t} $$

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Stationnarité

* Comment determiner p,d,q

* Quel modelisation appliquer

    </div>
</div>
</section>


<section data-markdown>
<div class=centerbox>
<p class=top>
II: Stationarity
</p>
</div>
</section>


<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Interpretation

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Autocorrelation ACF

as

$$  \hat{R}(k)  = \frac{1}{ (n-k) \sigma^{2} } \sum_{t=1}^{n-k} ( X\_t - \mu )
( X\_{t+k} - \mu ) $$


    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Interpretation

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Partial Autocorrelation PACF

The partial autocorrelation at lag k is the correlation that results after removing the effect of any correlations due to the terms at shorter lags

Given a time series  the partial autocorrelation of lag k, denoted \\( \alpha(k) \\), is the autocorrelation between \\( z\_{t} \\) and  \\( z\_{t+k} \\) with all the linear dependence of \\( z\_{t} \\) on \\( z\_{t+1}, ....,  z\_{t+k -1} \\) removed;

$$ \alpha(1)= \hat{R} (z\_{t+1},z\_{t}) $$


$$ \alpha(k) = \hat{R} ( z\_{t+k}-P\_{t,k}(z\_{t+k} ) , z\_{t} - P\_{t,k}(z\_{t} ) )$$

where \\( P\_{t,k}(x) \\) denotes the projection of \\(x\\) onto the space spanned by \\( x\_{t+1},\dots ,x\_{t+k-1} \\)   </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

<img src=/assets/08/Mean_nonstationary.png height=150>

<img src=/assets/08/Var_nonstationary.png height=150>

<img src=/assets/08/Cov_nonstationary.png height=150>

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>

# Stationarity
Key concept for time series

Do the caracteristics of the TS evolve over time ?

* Trend
* Cycles, Seasonality

A time serie is said to be stationnary if

* **Constant mean**: The mean of the series should be constant with respect to time.

* **Constant variance**: Homoscedasticity: The variance of the series constant with respect to time.

* **Fix lagged covariance**: The covariance of the \\(i^{th}\\) term and the \\( (i+m)^{th}\\) term should only depend on m and not i.

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# PACF - stationnarity
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# ACF - stationnarity

https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

    </div>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>

# KPSS

# Test random walk
    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Test for stationarity

## Dickey Fuller test

* Here the null hypothesis is that the TS is non-stationary.

* The test results comprise of a Test Statistic and some Critical Values for difference confidence levels.

https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line

## Augmented Dickey Fuller test

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
# Transform - Stationary

* difference
* log

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
# Decomposition

    </div>
</div>
</section>



<!-- ------------------------------------------ -->
<!-- ------------------------------------------ -->
<!-- ------------------------------------------ -->
<section>

* Autocorrelation function
https://analysights.wordpress.com/tag/box-pierce-test/

* Test est ce un random noise / walk
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
