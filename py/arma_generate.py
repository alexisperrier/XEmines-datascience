import statsmodels.api as sm
import numpy as np

# Parameters.
ar = np.array([.75, -.25])
ma = np.array([.65, .35])

ar = np.array([])
ma = np.array([])

# Simulate an ARMA process.

np.random.seed(42)

y = sm.tsa.arma_generate_sample(
    ar=np.r_[1, ar],
    ma=np.r_[1, ma],
    nsample=10000,
    sigma=1,
)

fig, ax = plt.subplots(1,1)
plt.plot(y)
plt.show()
# Fit ARMA process on the simulates to check coefficients, ACF and PACF.


# Plot ACF and PACF of estimated model.
sm.tsa.graphics.plot_acf(y, lags=20, zero=True);

sm.tsa.graphics.plot_pacf(y, lags=20, zero=True);

model = sm.tsa.ARMA(y, (1, 2)).fit(trend='c')
model.summary()

model.predict()
