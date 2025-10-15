# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

from mapie.regression import SplitConformalRegressor, CrossConformalRegressor


# %% [markdown]
# # 1st way: Split Conformal Prediction
# 
# For the code below, we suppose having access to:
# 
# - `regressor`: a fitted RegressorMixin from sklearn, with a `predict` function
# - `X_conformalize`, `y_conformalize`: the conformalization dataset (one year of data for all cities)
# - `X_test`, `y_test`: the conformalization dataset (one year of data for all cities). We assume `y` is of shape (n_cities, n_years)

# %%
confidence_level = 0.95
mapie_regressor = SplitConformalRegressor(
    estimator=regressor, confidence_level=confidence_level, prefit=True
)
mapie_regressor.conformalize(X_conformalize, y_conformalize)

y_pred, y_pred_interval = mapie_regressor.predict_interval(X_test)

# %%
# visualization of results
n_samples = 5
random_indices = np.random.choice(len(y_test), n_samples, replace=False)

y_test_sample = y_test[random_indices]
y_pred_sample = y_pred[random_indices]
y_interval_sample = y_pred_interval[random_indices]

y_errors = np.stack([
    y_pred_sample.flatten() - y_interval_sample[:, 0],
    y_interval_sample[:, 1] - y_pred_sample.flatten()
])

plt.figure(figsize=(10, 6))
plt.errorbar(
    range(n_samples),
    y_pred_sample.flatten(),
    yerr=y_errors,
    fmt="o",
    capsize=5,
    label="Prediction Intervals"
)
plt.scatter(
    range(n_samples),
    y_test_sample,
    color="red",
    marker="x",
    label="True Values"
)
plt.xlabel("Sampled data points")
plt.ylabel("Predicted values")
plt.title("Predictions with intervals for 5 random points")
plt.xticks(range(n_samples), random_indices)
plt.legend()
plt.show()

# %%
# check if coverage respected
coverage = np.mean(
    (y_test >= y_pred_interval[:, 0]) & (y_test <= y_pred_interval[:, 1])
)
print(f"Empirical coverage: {coverage:.3f}, Target coverage: {confidence_level:.3f}")

# %% [markdown]
# # 2nd way: Cross Conformal Prediction
# 
# It's a different approach useful for small data where the training and conformalization are done together.
# 
# For the code below, we suppose having access to:
# - `X_full`, `y_full`: the full dataset. We keep the last year for testing, the rest for training+conformalization
# 
# Note: it will probably require a specific cross validation setting internally for this to work

# %%
X, X_test, y, y_test = X_full[:, :-1], X_full[:, -1:], y_full[:, :-1], y_full[:, -1:]

mapie_regressor = CrossConformalRegressor(
    estimator=Ridge(),
    confidence_level=0.95,
).fit_conformalize(X, y)

y_pred, y_pred_interval = mapie_regressor.predict_interval(X_test)


