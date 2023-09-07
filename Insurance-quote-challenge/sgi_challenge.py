# %% [markdown]
# Data Set Background
# -	External data set purchased by SGI
# -	Data contains actual quotes issued by SGI brokers from July 2022 to July 2023
# -	Field QUOTEID uniquely identifies each quote
# -	The first four lowest competitors' premiums, SGI’s premium and the number of carrier’s a broker offers on a quote to a client are given
# -	A negative 1 in the data represents a request for the carrier’s default value for a field
# -	Variable named success indicates whether SGI got the business or not
#
# Situation
#
# SGI has just purchased the provided dataset from a 3rd party company.  You have been tasked with investigating and analyzing the dataset to develop and model that can predict quote success for the Alberta Homeowners market – we are interested in the home product only.

# %% [markdown]
# ### importing needed packages

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import os
import numpy as np
import pandas as pd
import plotly.express as px
import pgeocode
import plotly.graph_objects as go
import pyproj
from shapely.geometry import Point
from datetime import datetime as dt
import category_encoders as ce

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import DMatrix

from h2o.grid import H2OGridSearch
from sklearn.model_selection import train_test_split

# %%
import h2o

h2o.init()
from h2o.automl import H2OAutoML

# %%
# The link to the AB_ER_2021.shp file can be found on the province of Alberta's site: https://www.alberta.ca/geographic-geospatial-statistics#jumplinks-0
# %%
df = pd.read_excel("Data/DataSet - Data Scientist Assessment.xlsx", "in")
coldesc = pd.read_excel("Data/ColumnDescription.xlsx", "Sheet1")
ab_shp4 = gpd.read_file("Data/AB_ER_2021.shp")
print("The shape of the original dataframe is:", df.shape)

# %%
df_homeowners = df.loc[df.DwellingType.str.lower() == "homeowners"]

# %%
df_homeowners.loc[
    df_homeowners.NumberOfYearsSinceLastClaim.isna(), "NumberOfYearsSinceLastClaim"
] = -1
df_homeowners.loc[df_homeowners.SecondaryHeat.isna(), "SecondaryHeat"] = "NONE"
df_homeowners.loc[df_homeowners.PoolType.isna(), "PoolType"] = "No Pool"
df_homeowners.loc[df_homeowners.SecondaryHeatDate.isna(), "SecondaryHeatDate"] = "NONE"
df_homeowners.loc[df_homeowners.YearsSinceLastNSF.isna(), "YearsSinceLastNSF"] = -1
df_homeowners.loc[df_homeowners.NonSmoker.isna(), "NonSmoker"] = 0
df_homeowners.loc[df_homeowners.SBUAmount.isna(), "SBUAmount"] = 0
df_homeowners.loc[df_homeowners.EQBldgAmount.isna(), "EQBldgAmount"] = 0
df_homeowners.drop(
    columns=["EQContsAmount", "LivingAreaMeasure", "PrimaryHeatDate"], inplace=True
)
df_homeowners.loc[
    df_homeowners.NumberOfYearsSinceLastWaterClaim.isna(),
    "NumberOfYearsSinceLastWaterClaim",
] = -1
df_homeowners["QuoteDATE"] = pd.to_datetime(
    df_homeowners["QuoteDATE"], format="%d%b%Y:%H:%M:%S.%f"
)
df_homeowners["QuoteDATE"] = df_homeowners["QuoteDATE"].dt.date
df_homeowners["month"] = pd.to_datetime(df_homeowners["EffDATE"]).dt.month
df_homeowners["quarter"] = pd.to_datetime(df_homeowners["EffDATE"]).dt.quarter
df_homeowners["year"] = pd.to_datetime(df_homeowners["EffDATE"]).dt.year
df_homeowners.loc[
    df_homeowners["RequestedDeductible"] == -1, "RequestedDeductible"
] = 500
df_homeowners.loc[df_homeowners["QuotedDeductible"] == -1, "QuotedDeductible"] = 500
df_homeowners.loc[df_homeowners["SBUReqDeduct"] == -1, "SBUReqDeduct"] = 1.0e02
df_homeowners.loc[df_homeowners["SBUReqAmount"] == -1, "SBUReqAmount"] = 1.0e00
df_homeowners.loc[df_homeowners["SBUReqAmount"] == -2, "SBUReqAmount"] = 1.111e07
df_homeowners.loc[
    df_homeowners["RequestedContentsLimit"] == -1, "RequestedContentsLimit"
] = 0.0000e00
df_homeowners.loc[
    df_homeowners["RequestedOutbuildingsLimit"] == -1, "RequestedOutbuildingsLimit"
] = 0.000000e00
df_homeowners.loc[df_homeowners["liability_limit"] == -1, "liability_limit"] = 1000000


df_homeowners.loc[df_homeowners["ExteriorFinish"].isna(), "ExteriorFinish"] = "Unknown"
df_homeowners.loc[
    df_homeowners["ConstructionType"].isna(), "ConstructionType"
] = "OTHER"
df_homeowners.loc[
    df_homeowners["ElectricalWiringType"].isna(), "ElectricalWiringType"
] = "UNKNOWN"
df_homeowners.loc[df_homeowners["RoofType"].isna(), "RoofType"] = "OTHER"


# %%
def get_lat_lon(postal_code):
    nomi = pgeocode.Nominatim("ca")
    location = nomi.query_postal_code(postal_code)
    return [location.latitude], [location.longitude]


gdf = df_homeowners.copy()

uni_pos = list(gdf["PostalCode"].unique())
lat_lon = {
    pc: gpd.points_from_xy(*get_lat_lon(pc[:3] + " " + pc[3:])) for pc in uni_pos
}

gdf["geometry"] = gdf["PostalCode"].map(lat_lon)
gdf["geometry"] = gdf["geometry"].apply(lambda x: x[0])
gdf.set_geometry("geometry")
gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
print("The shape of dataframe with only homeowners is:", df_homeowners.shape)
print("The shape of dataframe with geometry column added is:", gdf.shape)

# %%
print("GIS dataframe as the shape:", ab_shp4.shape)

# plotting the map of Paris where each district has it's own colour.
fig, ax = plt.subplots(1, figsize=(15, 15))
ab_shp4.plot(cmap="tab20", column="ERNAME", ax=ax, categorical=True, legend=True)
ax.set_title(
    "Economic Regions of Alberta", fontdict={"fontsize": "25", "fontweight": "3"}
)

leg = ax.get_legend()
# leg.set_bbox_to_anchor((1.15, 0.5))
ax.set_axis_off()

plt.savefig("er_ab.png", format="png")
plt.show()

# %%
tmp_poly = list(ab_shp4.geometry.unique())
regions = {}
for poly in tmp_poly:
    tmp = list(ab_shp4.loc[ab_shp4["geometry"] == poly]["ERNAME"].unique())[0]
    regions[tmp] = poly


def convert_lat_lon(p):
    # Define the CRS codes for WGS84 (latitude, longitude) and the target CRS (EPSG:3400)
    wgs84_crs = pyproj.CRS("EPSG:4326")  # WGS84 CRS
    target_crs = pyproj.CRS("EPSG:3400")  # Your target CRS

    # Create a transformer to convert between the two CRS
    transformer = pyproj.Transformer.from_crs(wgs84_crs, target_crs, always_xy=True)
    point_wgs84 = p
    point_target_crs = transformer.transform(point_wgs84.y, point_wgs84.x)

    return Point(point_target_crs[0], point_target_crs[1])


# %%
for index, row in gdf.iterrows():
    if row["geometry"].is_empty:
        print(f"Geometry at index {index} is empty {row['PostalCode']} {row['City']}")

gdf.loc[(gdf["geometry"].is_empty) & (gdf["City"] == "EDMONTON"), "geometry"] = gdf.loc[
    ~(gdf["geometry"].is_empty) & (gdf["City"] == "EDMONTON")
]["geometry"].iloc[0]


city = "TSUUT'INA"
nomi = pgeocode.Nominatim("ca")
location = nomi.query_postal_code("t3e 2l9")
point = gpd.points_from_xy([location.latitude], [location.longitude])
gdf.loc[(gdf["geometry"].is_empty) & (gdf["City"] == city), "geometry"] = point


for index, row in gdf.iterrows():
    if row["geometry"].is_empty:
        print(f"Geometry at index {index} is empty {row['PostalCode']} {row['City']}")

# %%
uni_geo = list(gdf.geometry.unique())
for ge in uni_geo:
    gdf.loc[gdf["geometry"] == ge, "geometry"] = convert_lat_lon(ge)


def find_region(point):
    for reg, polygon in regions.items():
        if point.within(polygon):
            return reg
    return None


uni_geo = list(gdf.geometry.unique())
gdf["ERNAME"] = ""
for ge in uni_geo:
    gdf.loc[gdf["geometry"] == ge, "ERNAME"] = find_region(ge)

# %%
merged_df = gdf.merge(ab_shp4, how="right", on="ERNAME")
print("The shape of merged dataframe is:", merged_df.shape)

# %%
tmp = (
    merged_df.groupby(["ERNAME", "geometry_y"])
    .agg(total_success=("success", "sum"), total_quote=("success", "count"))
    .sort_values("total_success", ascending=False)
    .reset_index()
)

tmp["comp_success"] = tmp["total_quote"] - tmp["total_success"]
tmp["conversion_rate"] = tmp["total_success"] / tmp["total_quote"]
tmp["comp_conversion_rate"] = tmp["comp_success"] / tmp["total_quote"]
tmp = gpd.GeoDataFrame(tmp, geometry="geometry_y")

# %%
vmin = tmp["conversion_rate"].min()
vmax = tmp["conversion_rate"].max()
cmap = "viridis"

fig, ax = plt.subplots(1, figsize=(10, 10))
tmp.plot(column="conversion_rate", ax=ax, edgecolor="0.8", linewidth=1, cmap=cmap)

# Add a title
ax.set_title(
    "SGI Conversation Rate Across Alberta's ER",
    fontdict={"fontsize": "14", "fontweight": "3"},
)


# Create colorbar as a legend
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)

tmp.apply(
    lambda x: ax.annotate(
        text=x["ERNAME"], xy=x.geometry_y.centroid.coords[0], ha="center", fontsize=6
    ),
    axis=1,
)
plt.savefig("er_heat_suc.png", format="png")
plt.show()


vmin = tmp["comp_conversion_rate"].min()
vmax = tmp["comp_conversion_rate"].max()
cmap = "viridis"

fig, ax = plt.subplots(1, figsize=(10, 10))
tmp.plot(column="comp_conversion_rate", ax=ax, edgecolor="0.8", linewidth=1, cmap=cmap)

# Add a title
ax.set_title(
    "Competitors Conversation Rate Across Alberta's ER",
    fontdict={"fontsize": "14", "fontweight": "3"},
)


# Create colorbar as a legend
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
# Empty array for the data range
sm._A = []
# Add the colorbar to the figure
cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)

tmp.apply(
    lambda x: ax.annotate(
        text=x["ERNAME"], xy=x.geometry_y.centroid.coords[0], ha="center", fontsize=6
    ),
    axis=1,
)
plt.savefig("er_heat_fail.png", format="png")
plt.show()

# %%
fig = px.bar(
    tmp,
    x="ERNAME",
    y=["comp_conversion_rate", "conversion_rate"],
    title="SGI Conversion Rate Across Alberta's ER",
    labels={"value": "Conversion Rate"},
    opacity=0.9,
    text_auto=".2%",
    orientation="v",
    barmode="group",
    width=900,
    height=600,
)
fig.update_yaxes(tickformat=".2%")
fig.write_image("conversation_rate.png", format="png")
fig.show()

# %%
fig = px.bar(
    tmp,
    x="ERNAME",
    y=["comp_success", "total_success"],
    title="SGI Success Count Across Alberta's ER",
    labels={"value": "Quote Count"},
    opacity=0.9,
    orientation="v",
    text_auto=".2s",
    barmode="group",
    width=900,
    height=600,
)
fig.write_image("conversation_count.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["success"])
    .agg(total_success=("success", "count"))
    .sort_values("total_success", ascending=False)
    .reset_index()
)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="success",
    y="total_success",
    title="SGI Quote Across Alberta's ER",
    labels={"total_success": "Conversion Count"},
    opacity=0.9,
    orientation="v",
    text_auto=".2s",
    color="success",
    barmode="group",
    width=900,
    height=600,
)
fig.write_image("quote_count.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.loc[merged_df["success"] == 1]
    .groupby(["ERNAME", "SGI_Premium_Rank"])
    .agg(total_success=("success", "count"))
    .sort_values("total_success", ascending=False)
    .reset_index()
)

fig = px.bar(
    tmp,
    x="ERNAME",
    y="total_success",
    title="SGI Premium Rank Across Alberta's ER - Converted",
    labels={"total_success": "Conversion Count"},
    opacity=0.9,
    color="SGI_Premium_Rank",
    orientation="v",
    barmode="group",
    width=900,
    height=600,
)
fig.write_image("rank_er_suc.png", format="png")
fig.show()


tmp = (
    merged_df.loc[merged_df["success"] == 0]
    .groupby(["ERNAME", "SGI_Premium_Rank"])
    .agg(total_success=("success", "count"))
    .sort_values("total_success", ascending=False)
    .reset_index()
)

fig = px.bar(
    tmp,
    x="ERNAME",
    y="total_success",
    title="SGI Premium Rank Across Alberta's ER - Lost",
    labels={"total_success": "Conversion Count"},
    opacity=0.9,
    color="SGI_Premium_Rank",
    orientation="v",
    barmode="relative",
    width=900,
    height=600,
)
fig.write_image("rank_er_fail.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.loc[merged_df["success"] == 1]
    .groupby(["SGI_Premium_Rank", "cnt_company_comp"])
    .agg(total_comp=("cnt_company_comp", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

fig = px.bar(
    tmp,
    x="cnt_company_comp",
    y="total_comp",
    title="SGI Premium Rank Against Competitors - Successful Conversion",
    labels={"total_comp": "Number of Occurrences"},
    color="SGI_Premium_Rank",
    opacity=0.9,
    orientation="v",
    barmode="stack",
    width=900,
    height=600,
)
fig.write_image("rank_er_suc_com.png", format="png")
fig.show()

tmp2 = (
    merged_df.loc[merged_df["success"] == 0]
    .groupby(["SGI_Premium_Rank", "cnt_company_comp"])
    .agg(total_comp=("cnt_company_comp", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

fig = px.bar(
    tmp2,
    x="cnt_company_comp",
    y="total_comp",
    title="SGI Premium Rank Against Competitors - Failed Conversion",
    labels={"total_comp": "Number of Occurrences"},
    color="SGI_Premium_Rank",
    opacity=0.9,
    orientation="v",
    barmode="stack",
    width=900,
    height=600,
)
fig.write_image("rank_er_fai_com.png", format="png")
fig.show()

# %%
# Create histogram traces for each dataset
trace1 = go.Histogram(
    x=merged_df.loc[merged_df["success"] == 1]["SGI_Premium_Rank"],
    opacity=0.5,
    name="Won",
)
trace2 = go.Histogram(
    x=merged_df.loc[merged_df["success"] == 0]["SGI_Premium_Rank"],
    opacity=0.5,
    name="Lost",
)
trace3 = go.Histogram(x=merged_df["SGI_Premium_Rank"], opacity=0.5, name="Total")
# Create a figure and add the histogram traces
fig = go.Figure(data=[trace1, trace3])

# Update layout and show the plot
fig.update_layout(
    title="SGI Rank Distribution",
    xaxis_title="Rank",
    yaxis_title="Frequency",
    barmode="overlay",  # Overlay histograms
)
fig.write_image("rank_histo.png", format="png")

fig.show()

# %%
# Create histogram traces for each dataset
trace1 = go.Histogram(
    x=merged_df.loc[merged_df["success"] == 1]["SGI_Premium"], opacity=0.5, name="Won"
)
trace2 = go.Histogram(
    x=merged_df.loc[merged_df["success"] == 0]["SGI_Premium"], opacity=0.5, name="Lost"
)
trace3 = go.Histogram(x=merged_df["SGI_Premium"], opacity=0.5, name="Total")
# Create a figure and add the histogram traces
fig = go.Figure(data=[trace1, trace3])

# Update layout and show the plot
fig.update_layout(
    title="SGI Premium Distribution",
    xaxis_title="Values",
    yaxis_title="Frequency",
    barmode="overlay",  # Overlay histograms
)
fig.write_image("histo.png", format="png")

fig.show()

# %%
# Create histogram traces for each dataset
trace1 = go.Histogram(
    x=merged_df.loc[merged_df["success"] == 1]["BuildingAge"], opacity=0.5, name="Won"
)
trace2 = go.Histogram(
    x=merged_df.loc[merged_df["success"] == 0]["BuildingAge"], opacity=0.5, name="Lost"
)
trace3 = go.Histogram(x=merged_df["BuildingAge"], opacity=0.5, name="Total")
# Create a figure and add the histogram traces
fig = go.Figure(data=[trace1, trace3])

# Update layout and show the plot
fig.update_layout(
    title="Building Age Distribution",
    xaxis_title="Age",
    yaxis_title="Frequency",
    barmode="overlay",  # Overlay histograms
)
fig.write_image("histo_building.png", format="png")

fig.show()

# %%
val1 = len(
    merged_df.loc[(merged_df["BuildingAge"] > 25) & (merged_df["success"] == 1)][
        "success"
    ]
)
val2 = len(
    merged_df.loc[(merged_df["BuildingAge"] <= 25) & (merged_df["success"] == 1)][
        "success"
    ]
)
val3 = len(
    merged_df.loc[(merged_df["BuildingAge"] > 25) & (merged_df["success"] == 0)][
        "success"
    ]
)
val4 = len(
    merged_df.loc[(merged_df["BuildingAge"] <= 25) & (merged_df["success"] == 0)][
        "success"
    ]
)

categories = ["25 and Under", "Over 25"]
values1 = [val2, val1]
values2 = [val4, val3]

trace1 = go.Bar(x=categories, y=values1, name="Won", text=values1)
trace2 = go.Bar(x=categories, y=values2, name="Loss", text=values2)

layout = go.Layout(
    title="Quote Count across Building Age",
    xaxis=dict(title="Age"),
    yaxis=dict(title="Amount"),
    barmode="group",
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.update_traces(textposition="auto", texttemplate="%{text:.2s}")
fig.write_image("build_age_bar.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["YearsSinceLastNSF", "success"])
    .agg(total_comp=("success", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)


tmp["YearsSinceLastNSF"] = tmp["YearsSinceLastNSF"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="YearsSinceLastNSF",
    y="total_comp",
    title="Years Since Last NSF Distribution",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.write_image("yrnsf_bar.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["NumberOfNSF", "success"])
    .agg(total_comp=("success", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)


tmp["NumberOfNSF"] = tmp["NumberOfNSF"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="NumberOfNSF",
    y="total_comp",
    title="Number of NSF Distribution",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.write_image("nsf_bar.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["FireAlarmType", "success"])
    .agg(total_comp=("success", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

tmp["FireAlarmType"] = tmp["FireAlarmType"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="FireAlarmType",
    y="total_comp",
    title="Fire Alarm Type",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.write_image("FireAlarmType.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["BurglarAlarmType", "success"])
    .agg(total_comp=("success", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

tmp["BurglarAlarmType"] = tmp["BurglarAlarmType"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="BurglarAlarmType",
    y="total_comp",
    title="Burglar Alarm Type",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.show()

fig.write_image("BurglarAlarmType.png", format="png")

# %%
tmp = (
    merged_df.groupby(["ElectricalWiringType", "success"])
    .agg(total_comp=("success", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

tmp["ElectricalWiringType"] = tmp["ElectricalWiringType"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="ElectricalWiringType",
    y="total_comp",
    title="Electrical Wiring Type",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.write_image("Elec_w.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["FPC", "success"])
    .agg(total_comp=("success", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

tmp["FPC"] = tmp["FPC"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="FPC",
    y="total_comp",
    title="Fire Protection Code Distribution",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.write_image("FPC.png", format="png")
fig.show()

# %%
tmp = (
    merged_df.groupby(["NonSmoker", "success"])
    .agg(total_comp=("SGI_Premium", "count"))
    .sort_values("total_comp", ascending=False)
    .reset_index()
)

tmp["NonSmoker"] = tmp["NonSmoker"].astype(str)
tmp["success"] = tmp["success"].astype(str)
fig = px.bar(
    tmp,
    x="NonSmoker",
    y="total_comp",
    title="Smoker Distribution",
    labels={"total_comp": "Number of Occurrences"},
    opacity=0.9,
    orientation="v",
    color="success",
    barmode="group",
    text_auto=".2s",
    width=900,
    height=600,
)
fig.write_image("NonSmoker.png", format="png")
fig.show()

# %%
merged_df["EffDATE"] = merged_df["EffDATE"].dt.date
merged_df["month"] = pd.to_datetime(merged_df["EffDATE"]).dt.month
merged_df["quarter"] = pd.to_datetime(merged_df["EffDATE"]).dt.quarter
merged_df["year"] = pd.to_datetime(merged_df["EffDATE"]).dt.year

# %%
# Create histogram traces for each dataset
trace1 = go.Histogram(
    x=merged_df.loc[(merged_df["success"] == 1)]["EffDATE"], opacity=0.5, name="Won"
)
trace2 = go.Histogram(x=merged_df["EffDATE"], opacity=0.5, name="Total")
# Create a figure and add the histogram traces
fig = go.Figure(data=[trace1, trace2])

# Update layout and show the plot
fig.update_layout(
    title="Quote Distribution",
    xaxis_title="Time",
    yaxis_title="Frequency",
    barmode="overlay",  # Overlay histograms
)
fig.write_image("EffDATE.png", format="png")
fig.show()

# %%
for col in [
    "NumberOfClaims10years",
    "NumberOfClaims5years",
    "NumberOfClaims3years",
    "NumberOfYearsSinceLastClaim",
    "NumberOfWaterClaims10years",
    "NumberOfWaterClaims5years",
    "NumberOfWaterClaims3years",
    "NumberOfYearsSinceLastWaterClaim",
    "Occupation",
]:
    tmp = (
        merged_df.groupby([col, "success"])
        .agg(total_comp=("success", "count"))
        .sort_values("total_comp", ascending=False)
        .reset_index()
    )

    tmp[col] = tmp[col].astype(str)
    tmp["success"] = tmp["success"].astype(str)
    fig = px.bar(
        tmp,
        x=col,
        y="total_comp",
        title=f"{col} Distribution",
        labels={"total_comp": "Number of Occurrences"},
        opacity=0.9,
        orientation="v",
        color="success",
        barmode="group",
        text_auto=".2s",
        width=900,
        height=600,
    )
    fig.write_image(f"{col}.png", format="png")
    fig.show()

# %%
# Create histogram traces for each dataset
trace1 = go.Histogram(
    x=merged_df.loc[(merged_df["success"] == 1)]["QuotedLiabilityLimit"],
    opacity=0.5,
    name="Won",
)
trace2 = go.Histogram(x=merged_df["QuotedLiabilityLimit"], opacity=0.5, name="Total")
# Create a figure and add the histogram traces
fig = go.Figure(data=[trace1, trace2])

# Update layout and show the plot
fig.update_layout(
    title="QuotedLiabilityLimit Distribution",
    xaxis_title="Time",
    yaxis_title="Frequency",
    barmode="overlay",  # Overlay histograms
)
fig.write_image("QuotedLiabilityLimit.png", format="png")
fig.show()

# %%
# Create histogram traces for each dataset
trace1 = go.Histogram(
    x=merged_df.loc[(merged_df["success"] == 1)]["InsuredAge"],
    opacity=0.5,
    name="Won",
)
trace2 = go.Histogram(x=merged_df["InsuredAge"], opacity=0.5, name="Total")
# Create a figure and add the histogram traces
fig = go.Figure(data=[trace1, trace2])

# Update layout and show the plot
fig.update_layout(
    title="InsuredAge Distribution",
    xaxis_title="Time",
    yaxis_title="Frequency",
    barmode="overlay",  # Overlay histograms
)
fig.write_image("InsuredAge.png", format="png")
fig.show()

# %% [markdown]
# 1)	Some summary statistics for the data as it relates to quote success.
#     With over 70k quotes made by SGI between the period of July 2022 to July 2023 and converting 3,337 of them the overall conversion rate for SGI during the period of July 2022 to July 2023 is 4.74%. From the exploratory analysis we able to see that SGI has a relatively small portion of the market share with it's highest conversation rate of 8.3% coming from the region of Wood Buffalo -- Cold Lake. The majority of it's conversions happens with the regions of Calgary and Edmonton. This is the case in general for all insurers as most of the quotes are being made from Calgary and Edmonton; likely due to the large population hub within both regions. However, SGI's success rate in these 2 regions are slightly below it's average of 4.74%; 4.54% and 4.18% for Calgary and Edmonton respectively. The rank of the premium directly affected the conversation possibility of a quote, this is evident by the fact that the majority of the quotes were converted when SGI premium was ranked top 2. However, the failed conversions mainly happened when the quote premium were ranked last. This is a clear and consistent trend seen across all economic regions within Alberta. The Alberta market is quite competitive, for every quote won by SGI there is typically 2 to 8 competitors bidding for the conversion as well. In cases where SGI is able to successfully convert a quote there's a clear trend of it providing the lowest premium regardless of the number of competitors. From the overlaid histograms we see that SGI premium distribution for converted quotes is within the range of 800 to 6k with majority of it falling between 1200 to 3000.  Most popular deductable amount is 1000, followed by 2500 Most popular liability limit for converted quote is 2M followed far behind is 3M and 1M respectively. SGI quotes roughly the same amount for 1M liability limit and 3M liability limit, yet, it's conversion of 3M liability limit quotes is almost double that of it's 1M liability limit quotes. Age of the insured and co-insured ranges from 20 to 80 with majority between late 30s early 40s.
#
#     From the SGI Rank Distribution histogram we can see that majority of SGI quotes are ranked bottom 3. With the bottom rank of 5 holding the single largest quote count amongst the ranks. This could explain why majority of the quotes are not converted our pricing may not be as competitive as it should within the province of Alberta. As stated above SGI has best success rate when the premium is ranked amongst the top 2. when we convert the quote we tend to be rank 1 or 2 the opposite is true when lose a quote we tend to be in the bottom rank
#
#     Based on the plots below the slight majority of the successful quotes are of homes that are over 25 years. However, majority of the quotes issued are to homes under 25 years. Thus, SGI intentions are to write good risk, though market competitiveness has made it a challenge to convert more homes under 25 years old. Majority of the claims quoted and won have never had Insufficient fund issue arise. For the small handful of risks with NSF, SGI mainly quotes those risks who had 1 or 2 number of NSF.  I explored several other features about the quote, features such as FPC, BurglarAlarmType, FireAlarmType, NonSmoker, and ElectricalWiringType. I found that when possible SGI aims to write good risks and there is no significant deviation or segmentation from the quotes won versus those lost when comparing the features above.
#
#     There seems to be some seasonality or cyclically behavour to the quotes per day. I believe this likely due to reduced quoting on weekends and holidays. Based on the effective date of the converted quotes we can see the effects of seasonality over the period. SGI success rate was highest during the summer months of 2022 and then saw minor decline over the winter months. This decline was overturned at the beginning of spring 2023 and trend is implying likely another high success rate during the summer of 2023.
#
#     Looking at the claims distribution we see that vast majority of quotes and conversions made by SGI were to homes that have never had a claim.
#
#
# Based on the EDA conducted SGI is writing good risks, with their most successful month being the spring and summer months. Judging by the current market share that SGI has I believe it's fair to assume that SGI is still quite new in market. The keys to success for continuous growth within this market seems to be having competitive premium pricing, i.e. ranking top 2, increasing quoting efforts in the populous regions of Alberta. This might take the form of more marketing dollars to increase public exposure to SGI home insurance serviced.

# %% [markdown]
# 2)	How did you prepare the data to be used in the model?
#     a.	Did the data need to be cleaned? And what did you do?
#         I filtered out the data for only quotes requested by homeowners. I am assuming by the statement "we are interested in the home product only", 'home product' is a reference to only homeowners. I performed some imputations based on the dataset column description with the aim of minimizing data loss. The imputation replaced Null values in columns where nulls had default values assigned to them already. Though, the column 'EQContsAmount' was dropped as it was completely empty. Finally I reformated the column 'QuoteDATE' to allow for easier plotting during the exploratory phase as well creating the columns 'year', 'quarter' and 'month' off of the reformatted 'QuoteDATE' column. The data currently contains 70391 rows and 110 columns. Overall the data was very clean with minimal missing values if needed these rows can be dropped has their impact on the data quality would minor.
#
#     b.	Did you need to manipulate or transform the data for it to be used for the model or analysis?
#         The geographical location of the home will likely affect the premium quoted and thus directly impacting the possibility of a quote conversion. To this end I pulled from the geospatial data from the province of Alberta's website. With this data I was able to chart the SGI's success rate across the economic regions of Alberta. With this plot we can highlight where SGI has great influence and potential areas to focus on to grow the business. Site: https://www.alberta.ca/geographic-geospatial-statistics#jumplinks-0. After the geospatial data had been added to the original dataset, I noticed that a few postal codes could not be mapped on to a region. Using the FSA value, the first 3 characters of the postal code, and city name I imputed these missing values.
#         From the shape file we are able to feature engineer the economic region each quote belongs to and it's respective latitude and longitude coordinate. These new features could be useful in the model as it is intuitive that insurance quote depend on the location. Rather than using the general city description provided, we can leverage the lat and lon proximity distance of each quote and potentially see if there is trend where SGI performs better in certain regions.
#

# %% [markdown]
# ### Model Analysis


# %%
def extract_lat_lon(point):
    return point.x, point.y


merged_df["lat"], merged_df["lon"] = zip(
    *merged_df["geometry_x"].apply(extract_lat_lon)
)

# %%
model_cols = [
    "PostalCode",
    "City",
    "FireHallDistance",
    "HasSingleLimit",
    "BuildingAge",
    "BuildingStyle",
    "ConstructionType",
    "FireAlarmType",
    "BurglarAlarmType",
    "ElectricalType",
    "ElectricalWiringType",
    "ElectricalAge",
    "PrimaryHeat",
    "SecondaryHeat",
    "RoofAge",
    "PlumbingAge",
    "HasInLawApartment",
    "HasBasementApartment",
    "IsLogConstruction",
    "NumberOfFamilies",
    "NumberOfUnits",
    "NumberOfStoreys",
    "LivingArea",
    "BackFlowValve",
    "HasSumpPumpPit",
    "SumpPumpType",
    "IsSumpPumpAlarmed",
    "SumpPumpAuxPower",
    "PoolType",
    "ExteriorFinish",
    "RoofType",
    "FinishedBasementPerc",
    "NumberOfFloaters",
    "NUMFULLBATHS",
    "NUMHALFBATHS",
    "PrimaryHeatApproved",
    "PrimaryHeatOilTank",
    "QuotedDeductible",
    "QuotedLiabilityLimit",
    "SecondaryHeatOilTank",
    "SecondaryHeatApproved",
    "HasSepticSystem",
    "GarageType",
    "PlumbingType",
    "InsuredAge",
    "CoInsuredAge",
    "Occupation",
    "NumberOfMortgages",
    "YearsWithCurrentResidence",
    "YearsPriorInsurance",
    "IsOwnerOccupied",
    "Retired",
    "NumberOfNSF",
    "YearsSinceLastNSF",
    "NonSmoker",
    "FormType",
    "IsDefaultForm",
    "RequestedDeductible",
    "SBUReqDeduct",
    "SBUAmount",
    "SBUReqAmount",
    "liability_limit",
    "BuildingLimit",
    "BuildingValue",
    "RequestedContentsLimit",
    "QuotedContentsLimit",
    "RequestedOutbuildingsLimit",
    "QuotedOutBuildingsLimit",
    "AdditionalLivingExpensesLimit",
    "EQDeduct",
    "EQBldgAmount",
    "NumberOfClaims10years",
    "NumberOfClaims5years",
    "NumberOfClaims3years",
    "NumberOfYearsSinceLastClaim",
    "NumberOfWaterClaims10years",
    "NumberOfWaterClaims5years",
    "NumberOfWaterClaims3years",
    "NumberOfYearsSinceLastWaterClaim",
    "cnt_company_comp",
    "cnt_company_comp_AC",
    "success",
    "FPC",
    "SGI_Premium_Rank",
    "SGI_Premium",
    "Comp1",
    "Comp2",
    "Comp3",
    "Comp4",
    "Comp1_AC",
    "Comp2_AC",
    "Comp3_AC",
    "Comp4_AC",
    "ERNAME",
    "lat",
    "lon",
]

# %%
numeric_columns = list(
    merged_df[model_cols].select_dtypes(include=["int", "float"]).columns
)
non_numeric_columns = [col for col in model_cols if col not in numeric_columns]
drop_cols = list(
    merged_df[non_numeric_columns]
    .isna()
    .sum()[merged_df[non_numeric_columns].isna().sum() > int(merged_df.shape[0] * 0.05)]
    .index
)

non_numeric_columns = [
    col for col in model_cols if col not in numeric_columns + drop_cols
]


# %%
non_numeric_columns

# %%
for i in (
    merged_df[non_numeric_columns]
    .isna()
    .sum()[merged_df[non_numeric_columns].isna().sum() > 0]
    .index
):
    print(i, "\n", merged_df[i].unique())

merged_df[non_numeric_columns].isna().sum()[
    merged_df[non_numeric_columns].isna().sum() > 0
]

# %%
drop_list_2 = list(
    merged_df[numeric_columns]
    .isna()
    .sum()[merged_df[numeric_columns].isna().sum() > int(merged_df.shape[0] * 0.05)]
    .index
)

numeric_columns = [col for col in numeric_columns if col not in drop_list_2]

# %%
merged_df[numeric_columns].isna().sum()[merged_df[numeric_columns].isna().sum() > 0]

# %%
print(len(non_numeric_columns), len(numeric_columns))
model_df = merged_df[numeric_columns + non_numeric_columns]

# %%
df_y = merged_df["success"]
df_x = model_df.copy()
df_x.drop(columns=["success"], inplace=True)

# %%
non_numeric_columns

# %%
binary_col = [
    "PostalCode",
    "City",
    "BuildingStyle",
    "ConstructionType",
    "ElectricalType",
    "ElectricalWiringType",
    "PrimaryHeat",
    "SecondaryHeat",
    "PoolType",
    "ExteriorFinish",
    "RoofType",
    "FormType",
    "ERNAME",
    "FPC",
]
ord_col = ["SGI_Premium_Rank"]

final_df = df_x.copy()
final_df[binary_col + ord_col] = df_x[binary_col + ord_col].astype("category")

final_df.shape

# %%
edit_cols = {}
for i in final_df.columns:
    if (final_df[i].loc[final_df[i] == -1].shape[0]) > 0:
        edit_cols[i] = final_df[i].loc[final_df[i] == -1].shape[0]

# %%
edit_cols

# %% [markdown]
# The first step of my model build was to select which features would be included in the initial list of features to consider as potential model features. Doing this I was able to drop off several features which intuitively would not have made been a good choice as feature. Columns such as 'province', 'brok_no', etc.. These columns provide additional information that do not help determine whether the quote will be converted or not. I also dropped several empty or mostly empty columns - the threshold was more than 5\% of the column is empty we drop it. I noticed that most columns with nan values could be imputed to either Unknown, other or something similar. This approach though a bit general will have little impact on the data quality because the amount of missing values is so small. I finally reformated the reformated the columns with -1 indicating base value for that particular feature and converted the geometry column to latitude and longitude.
#
# I turned all the categorical features to dtype category and used xgboost to determine the most import features amongst the initial list. The data was split 80/20 train and test respectively and I made sure to use stratified sampling with aim of preserving the imbalance class distribution found in the target variable 'success'. The xgboost model performed at 95\% on the test data. From the feature importance list the most important feature as determined by the xgboost model is 'SGI_Premium_Rank'. This not surprising as through out the data analysis it was the only feature that was evidently discriminatory with regards to the success rate. 'SGI_Premium_Rank' had F1 score of 58.82 while the second most important feature, 'HasSlumpPumpPit' had a F1 score of 24.74. The 'SGI_Premium_Rank' is almost 2 and half times more important than the second most important feature.

# %%

X_train, X_test, y_train, y_test = train_test_split(
    final_df, df_y, test_size=0.2, random_state=42, stratify=df_y
)
dtrain = DMatrix(data=X_train, label=y_train, enable_categorical=True)
dtest = DMatrix(data=X_test, label=y_test, enable_categorical=True)

params = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "alpha": 10,
    "learning_rate": 1.0,
}

results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    metrics="auc",
    early_stopping_rounds=10,
    verbose_eval=True,
    seed=42,
)

best_num_boost_round = results.shape[0]

best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=best_num_boost_round,
    evals=[(dtest, "test")],
    early_stopping_rounds=10,
)

y_test_pred = best_model.predict(dtest)
test_accuracy = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))
print("Test Accuracy:", test_accuracy)


# %%
xgb.plot_importance(best_model, max_num_features=12, importance_type="gain")
plt.figure(figsize=(18, 16))
plt.savefig("feat_imp.png", format="png")
plt.show()

# %%
importances = best_model.get_score(importance_type="gain")
top_12 = {}
# Sort the features by importance and print them
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
for indx, imp in enumerate(sorted_importances):
    top_12[imp[0]] = imp[1]
    if indx == 11:
        break

# %%
top_12

# %%
h2o.init()

h2o_df = h2o.H2OFrame(pd.concat([final_df, df_y], axis=1))
features = [k for k, v in top_12.items()]
target = "success"
train_df, test_df = h2o_df.split_frame(ratios=[0.8], seed=42)
train_df[target] = train_df[target].asfactor()
test_df[target] = test_df[target].asfactor()
aml = H2OAutoML(max_runtime_secs=650)
aml.train(x=features, y=target, training_frame=train_df)

# Print the AutoML leaderboard
lb = aml.leaderboard
print(lb)

# Get the best model
best_model = aml.leader

# Make predictions with the best model
predictions = best_model.predict(test_df)

# Shutdown H2O cluster
h2o.cluster().shutdown()


# %%
best_model

# %%
top_cols = [k for k, v in top_12.items()]
merged_df[top_cols].isna().sum()
