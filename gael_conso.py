"""
Analyze energy consumption data
"""


# %%
# Load the dataset
#df = pd.read_csv("conso-elec-gaz-annuelle-par-secteur-dactivite-agregee-commune.csv",
#                 sep=";", decimal=",", encoding="latin-1")
import pandas as pd
df = pd.read_parquet("consommation-annuelle-d-electricite-et-gaz-par-commune.parquet")

#df.groupby

# %%
# Display the original dataframe
import skrub
skrub.set_config(max_plot_columns=50)
skrub.TableReport(df)

# %%
# Sanitize dtypes
df = skrub.Cleaner(datetime_format="%Y-%m-%d").fit_transform(df)


# %%
# Drop columns with more than 50% missing values in the first few years
cut_off = pd.to_datetime("2020")
df_clean = skrub.Cleaner(drop_null_fraction=.5,
                         ).fit(df.query("annee <= @cut_off")).transform(df)


# %%
skrub.TableReport(df_clean)
# %%
# Aggregate by year and comune
df_agg = (df_clean
          .groupby(["annee", "code_commune", "filiere"])
          .agg({
              "conso_totale_mwh": "sum",
#              "nom_commune": "first",
              "code_departement": "first",
              "code_region": "first",
#              "nom_departement": "first",
#              "nom_region": "first",
              "nb_sites": "sum",
#              "code_epci": "first",
              "nom_epci": "first",
          })
          .reset_index()
          )

# %%
# Select on the electricity filiere
df_elec = df_agg.query("filiere == 'Electricité'").drop(columns=["filiere"])

skrub.TableReport(df_elec)
# %%
# Do a baseline prediction with ExtraTrees
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

df_elec_train = df_elec.query("annee < '2022-01-01'")
df_elec_test = df_elec.query("annee >= '2022-01-01'")

# %%
X_train = df_elec_train.drop(columns=["conso_totale_mwh"])
y_train = df_elec_train["conso_totale_mwh"]

X_test = df_elec_test.drop(columns=["conso_totale_mwh"])
y_test = df_elec_test["conso_totale_mwh"]

# %%
model = ExtraTreesRegressor(n_estimators=100, random_state=0)

# %%
# A first baseline, using only numeric columns
model.fit(X_train.select_dtypes(include=["number"]), y_train)
y_pred = model.predict(X_test.select_dtypes(include=["number"]))

# Compute the r2 score
from sklearn.metrics import r2_score, mean_absolute_error
print("R2 score (numeric only):", r2_score(y_test, y_pred))
print("MAE (numeric only):", mean_absolute_error(y_test, y_pred))

# %%
