# %%
import pandas as pd

energy = pd.read_parquet("consommation-annuelle-d-electricite-et-gaz-par-commune.parquet")
energy

# %%
energy["annee"].value_counts()

# %%
energy["filiere"].value_counts()

# %%
from skrub import Cleaner

energy_clean = Cleaner(drop_null_fraction=.5).fit_transform(energy)
energy_clean

# %%
import matplotlib.pyplot as plt

# Filter for Electricity
elec = energy_clean[energy_clean["filiere"] == "Electricité"].copy()
elec_grouped = elec.groupby("annee")["conso_totale_mwh"].sum()

# Filter for Gas
gaz = energy_clean[energy_clean["filiere"] == "Gaz"]
gaz_grouped = gaz.groupby("annee")["conso_totale_mwh"].sum()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(elec_grouped.index, elec_grouped.values, marker="o", label="Électricité")
plt.plot(gaz_grouped.index, gaz_grouped.values, marker="s", label="Gaz")

plt.title("Consommation totale (MWh) par année")
plt.xlabel("Année")
plt.ylabel("Consommation totale (MWh)")
plt.legend()
plt.show()

# %%
elec["annee"] = pd.to_datetime(elec["annee"])
elec_2023 = elec.query('annee.dt.year == 2023')
elec_2023

# %%
import geopandas as gpd

departments = gpd.read_file("../datasets/departements.geojson")
departments.columns
departments.head()

# %%
elec_dept = (
    elec_2023.groupby("code_departement", as_index=False)["conso_totale_mwh"]
    .sum()
)
elec_dept = (
    elec_2023.groupby("code_departement", as_index=False)["conso_totale_mwh"]
    .sum()
)
map_df = departments.merge(
    elec_dept, left_on="code", right_on="code_departement", how="left"
)

fig, ax = plt.subplots(figsize=(8, 8))
map_df.plot(
    column="conso_totale_mwh",
    cmap="YlOrRd",
    legend=True,
    edgecolor="black",
    linewidth=0.5,
    ax=ax,
)

ax.set_title("Consommation totale d'électricité par département (2023)", fontsize=12)
ax.axis("off")

# %%
from ipywidgets import interact, IntSlider

years = sorted(elec["annee"].dt.year.unique())

elec_dept_all = (
    elec.groupby(["annee", "code_departement"], as_index=False)["conso_totale_mwh"]
    .sum()
)
vmin = elec_dept_all["conso_totale_mwh"].min()
vmax = elec_dept_all["conso_totale_mwh"].max()

def plot_year(year):
    subset = elec[elec["annee"].dt.year == year]
    elec_dept = subset.groupby("code_departement", as_index=False)["conso_totale_mwh"].sum()
    map_df = departments.merge(elec_dept, left_on="code", right_on="code_departement", how="left")

    fig, ax = plt.subplots(figsize=(8, 8))
    map_df.plot(
        column="conso_totale_mwh",
        cmap="YlOrRd",
        legend=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        vmin=vmin,   # fixed color scale minimum
        vmax=vmax,   # fixed color scale maximum
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    ax.set_title(f"Consommation totale d'électricité par département ({year})", fontsize=12)
    ax.axis("off")
    plt.show()

interact(
    plot_year,
    year=IntSlider(
        min=min(years),
        max=max(years),
        step=1,
        value=max(years),
        description="Année:",
        continuous_update=False,
    ),
)
