#the goal was to create a satellite-based early warning system for agricultural stress in Sicily using MODIS NDVI, CHIRPS precipitation, 
#and MODIS burned-area data (2003–2024). High-risk years were defined using percentile-based thresholds, and a 
#one-year lead prediction framework was implemented. Model performance was evaluated using rolling-origin backtesting to 
#prevent temporal leakage. A regularized logistic regression model was used to estimate probabilistic risk of high agricultural
#stress one year in advance.
#The limitation are of course many, is a toy model. Analysis is limited by the relatively short temporal record and regional-scale aggregation, which constrain predictive power
#and spatial specificity. The model is designed as a probabilistic early-warning indicator rather than a deterministic forecast. 
#Crop-specific responses and management practices are not explicitly modeled. Uncertainty increases for multi-year lead times due 
#to the stochastic nature of Mediterranean precipitation. Future work could integrate crop-level yield data



!pip install earthengine-api geemap

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ee.Authenticate()
ee.Initialize(project="... ") #the name of the project did on https://code.earthengine.google.com/# 

drive.mount("/content/drive")

start_year = 2003
end_year   = 2024

season_start_month = 3   # March
season_end_month   = 9   # September

regions = (
    ee.FeatureCollection("FAO/GAUL/2015/level1")
      .filter(ee.Filter.eq("ADM0_NAME", "Italy"))
      .filter(ee.Filter.eq("ADM1_NAME", "Sicilia"))
)

sicily_geom = regions.geometry()

precip_ic = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
fire_ic   = ee.ImageCollection("MODIS/061/MCD64A1").select("BurnDate")

def mask_ndvi(img):
    qa = img.select("SummaryQA")
    ndvi = img.select("NDVI").multiply(0.0001)
    return ndvi.updateMask(qa.lte(1)).copyProperties(img, ["system:time_start"])

ndvi_ic = (
    ee.ImageCollection("MODIS/061/MOD13Q1")
      .select(["NDVI", "SummaryQA"])
      .map(mask_ndvi)
)


def season_range(year):
    year = ee.Number(year)
    start = ee.Date.fromYMD(year, season_start_month, 1)
    end   = ee.Date.fromYMD(year, season_end_month, 30)
    return start, end

years = ee.List.sequence(start_year, end_year)

ndvi_baseline = (
    ee.ImageCollection(
        years.map(lambda y: ndvi_ic.filterDate(*season_range(y)).mean())
    ).mean()
)

precip_baseline = (
    ee.ImageCollection(
        years.map(lambda y: precip_ic.filterDate(*season_range(y)).sum())
    ).mean()
)


def features_per_year(y):
    start, end = season_range(y)

    ndvi_mean = ndvi_ic.filterDate(start, end).mean()
    ndvi_min  = ndvi_ic.filterDate(start, end).min()
    ndvi_anom = ndvi_mean.subtract(ndvi_baseline)

    precip_sum  = precip_ic.filterDate(start, end).sum()
    precip_anom = precip_sum.subtract(precip_baseline)

    burned = (
        fire_ic.filterDate(start, end)
               .map(lambda img: img.gt(0))
               .sum()
    )

    stack = ee.Image.cat([
        ndvi_mean.rename("ndvi_mean"),
        ndvi_min.rename("ndvi_min"),
        ndvi_anom.rename("ndvi_anomaly"),
        precip_sum.rename("precip_sum"),
        precip_anom.rename("precip_anomaly"),
        burned.rename("burned_pixels")
    ])

    stats = stack.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=sicily_geom,
        scale=250,
        bestEffort=True
    )

    return ee.Feature(None, stats).set("year", y)

table = ee.FeatureCollection(years.map(features_per_year))

ee.batch.Export.table.toDrive(
    collection=table,
    description="sicilia_drought_ndvi_precip_fire",
    fileFormat="CSV"
).start()



df = pd.read_csv("/content/drive/MyDrive/sicilia_drought_ndvi_precip_fire.csv")
df = df.dropna().sort_values("year").reset_index(drop=True)

features = df[
    ["precip_anomaly", "ndvi_anomaly", "ndvi_min", "burned_pixels"]
]

Z = StandardScaler().fit_transform(features)

df["stress_index"] = (
    0.5 * Z[:,0] +
    0.3 * Z[:,1] +
    0.1 * Z[:,2] +
    0.1 * Z[:,3]
)


threshold = df["stress_index"].quantile(0.75)

df["high_risk"] = (df["stress_index"] > threshold).astype(int)
df["high_risk_next_year"] = df["high_risk"].shift(-1)

df_ew = df.dropna().copy()


X_all = df_ew[
    ["precip_anomaly", "ndvi_anomaly", "ndvi_min", "burned_pixels"]
]
y_all = df_ew["high_risk_next_year"]

years_bt, predicted_risk, observed_risk = [], [], []

min_train_years = 8

for i in range(min_train_years, len(df_ew)):
    X_train = X_all.iloc[:i]
    y_train = y_all.iloc[:i]
    X_test  = X_all.iloc[i:i+1]
    y_test  = y_all.iloc[i:i+1]

    model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[0,1]

    years_bt.append(df_ew.iloc[i]["year"])
    predicted_risk.append(prob)
    observed_risk.append(int(y_test.values[0]))

bt = pd.DataFrame({
    "year": years_bt,
    "predicted_risk": predicted_risk,
    "observed_high_risk": observed_risk
})

bt["alert"] = (bt["predicted_risk"] > 0.5).astype(int)
