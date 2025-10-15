"""
Analyze energy consumption data
"""


# %%
# Load the dataset

import polars as pl
df = pl.read_parquet("consommation-annuelle-d-electricite-et-gaz-par-commune.parquet")


# %%
# Display the original dataframe
import skrub
skrub.set_config(max_plot_columns=50)
skrub.TableReport(df)

# %%
# sanitize dtypes
df = skrub.Cleaner(datetime_format="%Y-%m-%d", n_jobs=-1).fit_transform(df)


# %%
# Drop columns with more than 50% missing values in the first few years
import datetime
cut_date = datetime.date(2022, 1, 1)

df_clean = skrub.Cleaner(drop_null_fraction=.5, n_jobs=-1,
                         ).fit(df.filter(pl.col("annee") <= pl.lit(cut_date))
                               ).transform(df)


# %%
skrub.TableReport(df_clean)
# %%
# Aggregate by year and commune
df_agg = (
    df_clean
    .group_by(["annee", "code_commune", "filiere"])
    .agg([
        pl.col("conso_totale_mwh").sum(),
        # pl.col("nom_commune").first(),
        pl.col("code_departement").first(),
        pl.col("code_region").first(),
        # pl.col("nom_departement").first(),
        # pl.col("nom_region").first(),
        pl.col("nb_sites").sum(),
        # pl.col("code_epci").first(),
        pl.col("nom_epci").first(),
    ])
)

# %%
# Select on the electricity filiere
df_elec = df_agg.filter(pl.col("filiere") == pl.lit('Electricité')).drop("filiere")

skrub.TableReport(df_elec)
# %%
# Do a baseline prediction with ExtraTrees
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

split_date = datetime.date(2022, 1, 1)

df_elec_train = df_elec.filter(pl.col("annee") < pl.lit(split_date))
df_elec_test = df_elec.filter(pl.col("annee") >= pl.lit(split_date))

# %%
X_train = df_elec_train.drop("conso_totale_mwh")
y_train = df_elec_train["conso_totale_mwh"]

X_test = df_elec_test.drop("conso_totale_mwh")
y_test = df_elec_test["conso_totale_mwh"]

# %%
model = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs=-1)

# %%
# A first baseline, using only numeric columns
model.fit(X_train.select(pl.selectors.numeric()), y_train)
y_pred = model.predict(X_test.select(pl.selectors.numeric()))

# Compute the r2 score
from sklearn.metrics import r2_score, mean_absolute_error
print("R2 score (numeric only):", r2_score(y_test, y_pred))
print("MAE (numeric only):", mean_absolute_error(y_test, y_pred))

# %%
# A second baseline, using all columns
model = skrub.tabular_pipeline(ExtraTreesRegressor(n_estimators=100,
                                                   random_state=0, n_jobs=-1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute the r2 score
print("R2 score (all columns):", r2_score(y_test, y_pred))
print("MAE (all columns):", mean_absolute_error(y_test, y_pred))

# %%
