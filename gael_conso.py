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
#              "population": "mean",
#              "surface": "mean",
              "nom_commune": "first",
              "code_departement": "first",
              "code_region": "first",
              "nom_departement": "first",
              "nom_region": "first",
              "nb_sites": "sum",
              "code_epci": "first",
              "nom_epci": "first",
          })
          .reset_index()
          )

# %%
