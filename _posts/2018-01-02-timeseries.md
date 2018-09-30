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
# Ts metrics

Forecasting error: \\( e\_i=y\_i−\hat{y}\_i \\)


Metrics to compare TS techniques

* Mean Absolute Error: \\( MAE=mean(|e\_i|) \\)
* [Mean Absolute Deviation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mad.html): \\( MAD = \frac{1}{n} \sum\_{i=1}^{n}  | e\_i |  \\)
* Root mean squared error: \\( RMSE = \sqrt{  mean(e\_i^2)  } \\)
* Mean absolute percentage error: \\( MAPE = \frac{100}{n} \sum\_{i=1}^{n} \frac{ | e\_i | }{ y\_i } \\)

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Simplissime

$$ \hat{Y}_{n+1} = Y_{n} $$

* Il fera le meme temps qu'hier

    </div>
</div>
</section>



<section data-markdown>
<div class=centerbox>
<p class=top>
II: Modelisation
</p>
</div>
</section>

<section>
<div style='float:right; width:45%;  '>
    <div data-markdown>
# Exponential Weighted Moving Average
Introduce a Decay

The EWMA for a series Y may be calculated recursively:

* \\( S\_{1}=Y\_{1} \\)
* \\(      t>1, \ \ S\_{t}=\alpha \cdot Y\_{t}+(1-\alpha )\cdot S\_{t-1} \\)


Where:

* The coefficient α represents the degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher α discounts older observations faster.
* Yt is the value at a time period t.
* St is the value of the EWMA at any time period t.

http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html

    </div>
</div>
<hr class='vline' />
<div style='float:left; width:45%;  '>
    <div data-markdown>
# Moving Average

$$ \begin{aligned} SMA = \frac{p\_{M} + p\_{M-1}+ \cdots +p\_{M-(n-1)}}{n} = \frac{1}{n} \sum\_{i=0}^{n-1}p\_{M-i}
\end{aligned} $$

and center

$$ \begin{aligned} SMA = \frac{p\_{M+n/2} + \cdots +   p\_{M+1} +   p\_{M} + p\_{M-1}+ \cdots +p\_{M-(n/2)}}{n} = \frac{1}{n} \sum\_{i=-\frac{n}{2}}^{\frac{n}{2}}p\_{M+i}
\end{aligned} $$

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
# Auto Regressive
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
# ARIMA
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
