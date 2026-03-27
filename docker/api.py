"""
api.py — API de prédiction de fin d'orage
==========================================
Expose deux endpoints :
  GET  /health   → vérifie que l'API tourne
  POST /predict  → reçoit un fichier CSV d'éclairs, retourne predictions.csv

Usage (une fois le container lancé) :
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
       -F "file=@mon_fichier_eclairs.csv" \
       --output predictions.csv
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import io

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE AU DÉMARRAGE
# (fait une seule fois quand le container démarre)
# ─────────────────────────────────────────────────────────────────

print("Chargement du modèle...")
model = xgb.Booster()
model.load_model("modele_fin_orage.json")
le           = joblib.load("label_encoder_airport.pkl")
feature_cols = joblib.load("feature_cols.pkl")
HORIZON_MIN  = 30
print("Modèle chargé.")

app = FastAPI(
    title="Meteorage — Prédiction fin d'orage",
    description="Data Battle 2026 — prédit la fin d'une alerte orage",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES (identiques au notebook)
# ─────────────────────────────────────────────────────────────────

def get_strikes_for_alert(df, airport, start_time, end_time, buffer_min=5):
    start = start_time - pd.Timedelta(minutes=buffer_min)
    end   = end_time   + pd.Timedelta(minutes=buffer_min)
    mask  = (
        (df['airport'] == airport) &
        (df['date'] >= start) &
        (df['date'] <= end)
    )
    return df[mask].copy().sort_values('date')


def build_alerts_from_cg(df, end_delay_min=30):
    cg = df[(df['icloud'] == False) & (df['airport_alert_id'].notna())].copy()
    cg['airport_alert_id'] = cg['airport_alert_id'].astype(int)

    alerts = []
    for (airport, alert_id), group in cg.groupby(['airport', 'airport_alert_id']):
        group      = group.sort_values('date')
        start_time = group['date'].iloc[0]
        last_flag  = group[group['is_last_lightning_cloud_ground'] == True]
        end_time   = (
            last_flag['date'].iloc[0] if len(last_flag) > 0
            else group['date'].iloc[-1]
        )
        alerts.append({
            'airport':            airport,
            'airport_alert_id':   alert_id,
            'start_time':         start_time,
            'end_time':           end_time,
            'alert_end_official': end_time + pd.Timedelta(minutes=end_delay_min),
        })
    return pd.DataFrame(alerts)


def compute_features(past_strikes, current_time, airport):
    feat = {'airport': airport}

    feat['month']     = current_time.month
    feat['hour']      = current_time.hour
    feat['month_sin'] = np.sin(2 * np.pi * current_time.month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * current_time.month / 12)
    feat['hour_sin']  = np.sin(2 * np.pi * current_time.hour / 24)
    feat['hour_cos']  = np.cos(2 * np.pi * current_time.hour / 24)
    minutes_since_midnight    = current_time.hour * 60 + current_time.minute
    feat['minute_of_day_sin'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    feat['minute_of_day_cos'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    nan_feats = [
        'time_since_last', 'time_since_last_cg', 'time_since_last_ic',
        'time_since_first', 'time_since_peak',
        'std_interarrival', 'max_interarrival',
        'mean_dist', 'trend_dist', 'dist_recent_vs_mean',
        'angular_spread', 'azimuth_concentration',
        'prop_ic', 'mean_amplitude', 'std_amplitude', 'amplitude_trend',
        'ic_ratio_trend', 'rate_ratio_5_30', 'deceleration',
    ]
    zero_feats = [
        'strike_count_5min', 'strike_count_10min', 'strike_count_30min',
        'rate_5min', 'rate_10min', 'rate_30min',
    ]

    if len(past_strikes) == 0:
        for f in nan_feats:  feat[f] = np.nan
        for f in zero_feats: feat[f] = 0
        return feat

    last = past_strikes.iloc[-1]
    feat['time_since_last'] = (
        current_time - last['date']
    ).total_seconds() / 60.0

    cg = past_strikes[past_strikes['icloud'] == False]
    feat['time_since_last_cg'] = (
        (current_time - cg.iloc[-1]['date']).total_seconds() / 60.0
        if len(cg) > 0 else np.nan
    )
    ic = past_strikes[past_strikes['icloud'] == True]
    feat['time_since_last_ic'] = (
        (current_time - ic.iloc[-1]['date']).total_seconds() / 60.0
        if len(ic) > 0 else np.nan
    )
    feat['time_since_first'] = (
        current_time - past_strikes.iloc[0]['date']
    ).total_seconds() / 60.0

    try:
        tmp = past_strikes.set_index('date').resample('5min').size()
        feat['time_since_peak'] = (
            (current_time - tmp.idxmax()).total_seconds() / 60.0
            if len(tmp) > 0 else np.nan
        )
    except Exception:
        feat['time_since_peak'] = np.nan

    counts = {}
    for w in [5, 10, 30]:
        mask = past_strikes['date'] >= current_time - pd.Timedelta(minutes=w)
        cnt  = int(mask.sum())
        feat[f'strike_count_{w}min'] = cnt
        feat[f'rate_{w}min']         = cnt / w
        counts[w] = cnt

    r1 = counts[5]  / 5
    r2 = (counts[10] - counts[5])  / 5
    r3 = (counts[30] - counts[10]) / 20
    feat['deceleration']    = np.polyfit([0, 1, 2], [r1, r2, r3], 1)[0]
    feat['rate_ratio_5_30'] = counts[5] / (counts[30] + 1e-6)

    if len(past_strikes) >= 2:
        intervals = (
            past_strikes['date'].diff().dt.total_seconds().dropna() / 60.0
        )
        feat['std_interarrival'] = intervals.std()
        feat['max_interarrival'] = intervals.max()
    else:
        feat['std_interarrival'] = np.nan
        feat['max_interarrival'] = np.nan

    feat['mean_dist'] = past_strikes['dist'].mean()
    recent = past_strikes.tail(min(5, len(past_strikes)))
    feat['trend_dist'] = (
        np.polyfit(np.arange(len(recent)), recent['dist'].values, 1)[0]
        if len(recent) >= 2 else 0
    )
    feat['dist_recent_vs_mean']   = (
        recent['dist'].mean() - past_strikes['dist'].mean()
    )
    feat['angular_spread']        = past_strikes['azimuth'].std()
    feat['azimuth_concentration'] = float(np.sqrt(
        np.cos(np.deg2rad(past_strikes['azimuth'])).mean() ** 2 +
        np.sin(np.deg2rad(past_strikes['azimuth'])).mean() ** 2
    ))

    feat['prop_ic']        = past_strikes['icloud'].mean()
    feat['mean_amplitude'] = past_strikes['amplitude'].mean()
    feat['std_amplitude']  = past_strikes['amplitude'].std()
    recent10 = past_strikes.tail(min(10, len(past_strikes)))
    feat['amplitude_trend'] = (
        np.polyfit(np.arange(len(recent10)), recent10['amplitude'].values, 1)[0]
        if len(recent10) >= 2 else 0
    )
    feat['ic_ratio_trend'] = (
        past_strikes.tail(10)['icloud'].mean() /
        (past_strikes['icloud'].mean() + 1e-6)
    )

    return feat


def generate_predictions(df):
    """
    Prend un DataFrame d'éclairs et retourne un DataFrame predictions
    au format attendu par le jury.
    """
    alerts = build_alerts_from_cg(df)
    all_predictions = []

    for _, alert_row in alerts.iterrows():
        airport      = alert_row['airport']
        start        = alert_row['start_time']
        end_cg       = alert_row['end_time']
        end_official = alert_row['alert_end_official']

        # Encoder l'aéroport — si inconnu, lever une erreur claire
        try:
            airport_enc = int(le.transform([airport])[0])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Aéroport inconnu : '{airport}'. "
                       f"Connus : {list(le.classes_)}"
            )

        strikes = get_strikes_for_alert(
            df, airport, start, end_cg,
            buffer_min=5 + HORIZON_MIN
        )

        for t in pd.date_range(start, end_official, freq='1min'):
            past  = strikes[strikes['date'] < t]
            feats = compute_features(past, t, airport)

            X_row = pd.DataFrame([feats]).drop(
                columns=['airport'], errors='ignore'
            )
            X_row['airport_encoded'] = airport_enc
            X_row = X_row[feature_cols]

            dm    = xgb.DMatrix(X_row, feature_names=feature_cols)
            proba = float(model.predict(dm)[0])

            all_predictions.append({
                'airport':                  airport,
                'airport_alert_id':         int(alert_row['airport_alert_id']),
                'prediction_date':          t,
                'predicted_date_end_alert': t,
                'confidence':               round(1.0 - proba, 6),
            })

    return pd.DataFrame(all_predictions)


# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Vérifie que l'API est bien démarrée.
    Retourne simplement {"status": "ok"}.
    """
    return {"status": "ok", "model": "modele_fin_orage.json"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit un fichier CSV d'éclairs (même format que les données Meteorage),
    retourne un fichier predictions.csv au format attendu par le jury.

    Format du CSV d'entrée attendu :
      date, airport, airport_alert_id, lat, lon, dist, azimuth,
      amplitude, icloud, is_last_lightning_cloud_ground, ...
    """
    # Lire le fichier uploadé
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lire le CSV : {e}"
        )

    # Vérifier les colonnes obligatoires
    required_cols = [
        'date', 'airport', 'airport_alert_id',
        'dist', 'azimuth', 'amplitude', 'icloud',
        'is_last_lightning_cloud_ground'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes dans le CSV : {missing}"
        )

    # Générer les prédictions
    try:
        predictions_df = generate_predictions(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des prédictions : {e}"
        )

    # Retourner le CSV en réponse
    output = io.StringIO()
    predictions_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv"
        }
    )