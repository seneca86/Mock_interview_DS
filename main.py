
# %%
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pathlib import Path
from pmdarima.arima import StepwiseContext
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)

# %% Exercise on bayesian inference
candy = pd.DataFrame(index=['1994', '1996'])
candy['prior'] = 1/2, 1/2
candy['likelihood'] = 0.2 * 0.2, 0.14 * 0.1
candy['unnorm'] = candy['prior'] * candy['likelihood']
prob_norm = candy.unnorm.sum()
candy['posterior'] = candy['unnorm'] / prob_norm
candy.head()

# %% Exercise on regression
red_wine = pd.read_csv('.lesson/assets/winequality-red.csv', sep=';')
# %%
# %%
formula = 'quality ~ alcohol'
model = smf.ols(formula, data = red_wine)
results = model.fit()
results.summary()
# %%
inter = results.params['Intercept']
slope = results.params['alcohol']
print(f'{inter=}')
print(f'{slope=}')
# %%
formula = 'quality ~ alcohol + pH + chlorides + density'
model = smf.ols(formula, data = red_wine)
results = model.fit()
results.summary()
inter = results.params['Intercept']
slope = results.params['alcohol']
print(f'{inter=}')
print(f'{slope=}')
# %%
red_wine['category'] = 'red'
white_wine = pd.read_csv('.lesson/assets/winequality-white.csv', sep=';')
white_wine['category'] = 'white'
# %%
wine = pd.concat([white_wine, red_wine])
# %%
wine['red'] = (wine['category'] == 'red') * 1
formula = 'red ~ alcohol + pH + chlorides + density'
model = smf.logit(formula, data=wine)
results = model.fit()
results.summary()
# %%
new = pd.DataFrame([[11, 3.3, 0.06, 1]], columns=['alcohol', 'pH', 'chlorides', 'density'])
y = results.predict(new)
print(f'The chances of this wine being red are {y[0]}')

# %% Exercise 3
flights = pd.read_csv(".lesson/assets/airline-passengers.csv")
flights["Date"] = pd.to_datetime(flights["Month"])
flights["delta"] = flights["Passengers"] - flights["Passengers"].shift(1)
flights.dropna(inplace=True)
# %%
plt.plot(flights["Date"], flights.delta, label="actuals")
plt.legend()
plt.savefig(f"{directory}/flights_actuals.png")
# %%
ix = 3
split = 125
train = flights.iloc[0:split, ix]
test = flights.iloc[split:, ix]
# %%
with StepwiseContext(max_dur=15):
    model = pm.auto_arima(
        train,
        stepwise=True,
        error_action="ignore",
        seasonal=True,
    )
# %%
print(f"{model.summary()}")
preds, conf_int = model.predict(n_periods=flights.shape[0] - split, return_conf_int=True)
# %%
plt.plot(train, label="actuals", color="black")
plt.plot(range(split,split + len(test)), preds, label="autoarima", color="blue")
plt.legend()
plt.savefig(directory + "/autoarima")
plt.clf()

# %%
