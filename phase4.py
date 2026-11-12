import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Metro_Interstate_Traffic_Volume.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

print("=== CSE437 PHASE 4 ===")
print("Loading dataset...")
df = pd.read_csv(DATA)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").drop_duplicates().copy()

# Combine simultaneous weather records at each timestamp.
num = ["traffic_volume","temp","rain_1h","snow_1h","clouds_all"]
cat = ["holiday","weather_main","weather_description"]
agg = {c:"mean" for c in num}
agg.update({c:lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan for c in cat})
df = df.groupby("date_time", as_index=False).agg(agg).set_index("date_time")

# Clean sentinel/impossible weather values.
df.loc[df.temp <= 0, "temp"] = np.nan
df.loc[df.rain_1h > 100, "rain_1h"] = np.nan
df.loc[df.snow_1h < 0, "snow_1h"] = np.nan

start = pd.Timestamp("2015-06-11 20:00")
end = pd.Timestamp("2018-09-30 23:00")
df = df.loc[start:end]
idx = pd.date_range(start, end, freq="h")
df = df.reindex(idx)
df.index.name = "date_time"

df["was_missing"] = df.traffic_volume.isna().astype(int)
df["holiday"] = df.holiday.fillna("None")
for c in ["temp","rain_1h","snow_1h","clouds_all"]:
    df[c] = df[c].interpolate(limit_direction="both").ffill().bfill()
for c in ["weather_main","weather_description"]:
    df[c] = df[c].ffill().bfill()
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek
df["how"] = df.index.dayofweek * 24 + df.index.hour

# Traffic: short gaps by interpolation; longer gaps by hour-of-week mean.
short = df.traffic_volume.interpolate(limit=3, limit_direction="both")
df["traffic_volume"] = short
clim = df.groupby("how").traffic_volume.transform("mean")
df["traffic_volume"] = df.traffic_volume.fillna(clim).interpolate(limit_direction="both")
df["is_holiday"] = (df.holiday.astype(str) != "None").astype(int)
df["is_snow"] = (df.snow_1h > 0).astype(int)
df["is_rain"] = (df.rain_1h > 0).astype(int)

# Lag/calendar/Fourier features.
for lag in [1,2,3,24,48,72,168]:
    df[f"lag_{lag}"] = df.traffic_volume.shift(lag)
df["hour_sin"] = np.sin(2*np.pi*df.hour/24)
df["hour_cos"] = np.cos(2*np.pi*df.hour/24)
df["dow_sin"] = np.sin(2*np.pi*df.dow/7)
df["dow_cos"] = np.cos(2*np.pi*df.dow/7)
t = np.arange(len(df))
for k in [1,2]:
    df[f"fourier_sin_{k}"] = np.sin(2*np.pi*k*t/168)
    df[f"fourier_cos_{k}"] = np.cos(2*np.pi*k*t/168)
df = df.dropna()

df.reset_index().to_csv(RESULTS/"traffic_model_ready.csv", index=False)

# Chronological split.
n = len(df)
a, b = int(.70*n), int(.85*n)
train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
print("Model-ready:", df.shape, " Train:",len(train)," Val:",len(val)," Test:",len(test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def mape(y,p):
    y,p=np.asarray(y),np.asarray(p); m=y!=0
    return np.mean(np.abs((y[m]-p[m])/y[m]))*100
def smape(y,p):
    y,p=np.asarray(y),np.asarray(p); d=np.abs(y)+np.abs(p); m=d!=0
    return np.mean(2*np.abs(y[m]-p[m])/d[m])*100
def mase(y,p,ins):
    return np.mean(np.abs(np.asarray(y)-np.asarray(p)))/np.mean(np.abs(np.diff(ins)))
def score(name,y,p,ins):
    return {"model":name,"MAE":mean_absolute_error(y,p),"RMSE":np.sqrt(mean_squared_error(y,p)),
            "MAPE_percent":mape(y,p),"sMAPE_percent":smape(y,p),"R2":r2_score(y,p),"MASE":mase(y,p,ins)}

y=test.traffic_volume.values
ins=train.traffic_volume.values
rows=[]
for name,lag in [("Persistence (lag 1)",1),("Daily naive (lag 24)",24),("Weekly naive (lag 168)",168)]:
    p=test.traffic_volume.shift(lag).bfill().values
    rows.append(score(name,y,p,ins))
cl=train.groupby("how").traffic_volume.mean()
p=test.how.map(cl).fillna(train.traffic_volume.mean()).values
rows.append(score("Hour-of-week climatology",y,p,ins))
pd.DataFrame(rows).to_csv(RESULTS/"baseline_results.csv",index=False)

# XGBoost.
print("Training XGBoost...")
import xgboost as xgb
features=["temp","rain_1h","snow_1h","clouds_all","is_holiday","is_snow","is_rain",
"hour","dow","hour_sin","hour_cos","dow_sin","dow_cos","fourier_sin_1","fourier_cos_1",
"fourier_sin_2","fourier_cos_2","lag_1","lag_2","lag_3","lag_24","lag_48","lag_72","lag_168","was_missing"]
mx=xgb.XGBRegressor(objective="reg:squarederror",n_estimators=800,max_depth=8,learning_rate=.03,
                    subsample=.9,colsample_bytree=.9,min_child_weight=5,reg_lambda=5,random_state=42)
mx.fit(train[features],train.traffic_volume,eval_set=[(val[features],val.traffic_volume)],verbose=False)
px=mx.predict(test[features])
rows.append(score("XGBoost",y,px,ins))
pd.DataFrame(rows).to_csv(RESULTS/"phase4_model_comparison.csv",index=False)

# LSTM.
print("Training LSTM...")
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42); np.random.seed(42)
cols=["traffic_volume","temp","rain_1h","snow_1h","clouds_all","is_holiday","hour_sin","hour_cos","dow_sin","dow_cos","was_missing"]
sc=StandardScaler().fit(train[cols])
z=sc.transform(df[cols]); ti=0
def seq(z,lo,hi,w=168):
    X=[]; Y=[]
    for i in range(max(w,lo),hi):
        X.append(z[i-w:i]); Y.append(z[i,ti])
    return np.asarray(X,np.float32),np.asarray(Y,np.float32)
Xtr,Ytr=seq(z,0,a); Xv,Yv=seq(z,a,b); Xte,Yte=seq(z,b,len(df))
net=Sequential([LSTM(64,return_sequences=True,input_shape=(168,len(cols))),Dropout(.2),LSTM(32),Dropout(.2),Dense(1)])
net.compile(optimizer=tf.keras.optimizers.Adam(.001),loss="mse")
net.fit(Xtr,Ytr,validation_data=(Xv,Yv),epochs=100,batch_size=64,
        callbacks=[EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)],verbose=1)
ps=net.predict(Xte,verbose=0).ravel()*sc.scale_[ti]+sc.mean_[ti]
ys=Yte*sc.scale_[ti]+sc.mean_[ti]
rows.append(score("LSTM",ys,ps,ins))
pd.DataFrame(rows).to_csv(RESULTS/"phase4_model_comparison.csv",index=False)

# SARIMAX.
print("Training SARIMAX (may take several minutes)...")
from statsmodels.tsa.statespace.sarimax import SARIMAX
sx=["temp","rain_1h","snow_1h","clouds_all","is_holiday"]
sm=SARIMAX(train.traffic_volume.astype(float),exog=train[sx].astype(float),
           order=(1,0,1),seasonal_order=(1,1,1,24),
           enforce_stationarity=False,enforce_invertibility=False)
sf=sm.fit(disp=False,maxiter=100)
psar=sf.get_forecast(steps=len(test),exog=test[sx].astype(float)).predicted_mean.values
rows.append(score("SARIMAX",y,psar,ins))

out=pd.DataFrame(rows).sort_values("MAE")
out.to_csv(RESULTS/"phase4_model_comparison.csv",index=False)
pd.DataFrame({"date_time":test.index,"actual":y,"xgboost":px}).to_csv(RESULTS/"xgboost_predictions.csv",index=False)
pd.DataFrame({"date_time":test.index[-len(ys):],"actual":ys,"lstm":ps}).to_csv(RESULTS/"lstm_predictions.csv",index=False)
pd.DataFrame({"date_time":test.index,"actual":y,"sarimax":psar}).to_csv(RESULTS/"sarimax_predictions.csv",index=False)
pd.DataFrame({"feature":features,"importance":mx.feature_importances_}).sort_values("importance",ascending=False).to_csv(RESULTS/"xgboost_feature_importance.csv",index=False)

print("\n=== PHASE 4 FINISHED ===")
print(out.to_string(index=False))
print("\nResults saved in:",RESULTS)
