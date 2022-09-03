# Live coding case

## Instructions

* Please use `matplotlib` for the plots and store the figures in a directoy

## Libraries and parameters

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from sklearn.inspection import plot_partial_dependence
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
```

## Exercise on regression

* Load the dataset stored in `.lesson/assets/winequality-red.csv`. The separator is a semicolon.

* Build a linear regression where quality is the y variable and alcohol the x variable; print a summary of the results, and the intercept and the slope.

* Build another linear regression with the regressors alcohol, pH, chlorides, and density in order to predict the quality. Print the intercept and slope of alcohol.

* Load the dataset stored in `.lesson/assets/winequality-white.csv`. Concatenate it to the dataset we read in the beginning.

* Build a logistic regression that predicts whether a wine is red or white based on the same regressors we used before. You will need to generate a binary variable named red.

* Estimate the probabilities that a wine is red given its alcohol is 11, its pH is 3.3, its chlorides are 0.06, and its density is 1.

## Exercise on time series

* Read the file `.lesson/assets/airline-passengers.csv`; the separator is a comma

* We will use the time series `Passengers` de-trended. Plot it.

* Split the time series into train (from element 0 until element 125) and test (from element 125 until the end) set.

* Fit an autoarima model to the time series.

* Print the summary of the model, and plot the prediction together with the actuals.

## (Bonus exercise - do only if time permits) Exercise on Bayesian Inference

(1) M&M’s are small candy-coated chocolates that come in a variety of colors. Mars, Inc., which makes M&M’s, changes the mixture of colors from time to time. In 1995, they introduced blue M&M’s.

In 1994, the color mix in a bag of plain M&M’s was 30% Brown, 20% Yellow, 20% Red, 10% Green, 10% Orange, 10% Tan.

In 1996, it was 24% Blue , 20% Green, 16% Orange, 14% Yellow, 13% Red, 13% Brown.

Suppose a friend of mine has two bags of M&M’s, and he tells me that one is from 1994 and one from 1996. He won’t tell me which is which, but he gives me one M&M from each bag. One is yellow and one is green. What is the probability that the yellow one came from the 1994 bag?

Tips:

* Hypothesis A: yellow M&M comes from 1994, green M&M comes from 1996

* Hypothesis B: yellow M&M comes from 1996, green M&M comes from 1994

* Generate a dataframe with two rows representing one hypothesis each

* The first column is the prior probability of the hypothesis

* The second column is the likelihood of the hypothesis

* The third column is the unnormalized posterior (prior times likelihood)

* The fourth column is the (normalized) posterior