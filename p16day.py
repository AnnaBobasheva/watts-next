# %% 
import polars as pl
import skrub
from skrub import TableReport
#%%
df = pl.read_parquet("consommation-annuelle-d-electricite-et-gaz-par-commune.parquet")
df = df.lazy()
skrub.TableReport(df.collect())
#%%
df.filter(filiere="Electricité")
df.filter(filiere="Electricité").group_by("operateur", "annee").agg(pl.col("conso_totale_mwh").sum())
nonnull = df.lazy().filter(~pl.col("taux_de_logements_collectifs").is_null()).collect()
first_years = df.lazy().filter(
    pl.col("annee").dt.year() < 2016
).collect()
last_years = df.lazy().filter(
    pl.col("annee").dt.year() >= 2016
).collect()

# %%
cleaner = skrub.Cleaner()
cleaned = cleaner.fit_transform(first_years)

cleaned_last_years = cleaner.transform(last_years)

# %%
df.lazy().group_by("operateur").agg(pl.col(("conso_totale_mwh")).sum()).collect()
# %%
df.top_k_by("conso_totale_mwh")
    # %%
