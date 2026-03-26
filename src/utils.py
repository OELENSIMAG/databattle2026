import pandas as pd
import xgboost as xgb
import numpy as np

HORIZON_MIN = 30

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(filepath):
    """
    Charge le fichier CSV contenant les éclairs.
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # # FIX #7 : filtrer les éclairs intra-nuage de Pise en 2016
    # # (système d'enregistrement différent — cf. documentation)
    # mask_pise_2016_ic = (
    #     (df['airport'] == 'Pise') &
    #     (df['date'].dt.year == 2016) &
    #     (df['icloud'] == True)
    # )
    # n_filtered = mask_pise_2016_ic.sum()
    # if n_filtered > 0:
    #     print(f"[load_data] Filtrage Pise 2016 IC : {n_filtered} éclairs supprimés.")
    # df = df[~mask_pise_2016_ic].reset_index(drop=True)

    return df


# =============================================================================
# 2. CONSTRUCTION DES ALERTES À PARTIR DES ÉCLAIRS NUAGE-SOL
# =============================================================================

def build_alerts_from_cg(df, end_delay_min=30):
    """
    Crée une table des alertes à partir des éclairs nuage-sol (icloud=False)
    qui ont un airport_alert_id non nul.

    Pour chaque alerte (aéroport + airport_alert_id), on calcule :
        - start_time        : date du premier éclair nuage-sol de l'alerte
        - end_time          : date du dernier éclair nuage-sol
                              (is_last_lightning_cloud_ground=True)
        - alert_end_official: end_time + end_delay_min  (règle opérationnelle)
        - lightning_count   : nombre total d'éclairs (tous types) dans la fenêtre
        - cg_count          : nombre d'éclairs nuage-sol
    """
    cg_strikes = df[(df['icloud'] == False) & (df['airport_alert_id'].notna())].copy()
    cg_strikes['airport_alert_id'] = cg_strikes['airport_alert_id'].astype(int)

    alerts = []
    for (airport, alert_id), group in cg_strikes.groupby(['airport', 'airport_alert_id']):
        group = group.sort_values('date')
        start_time = group['date'].iloc[0]

        last_cg = group[group['is_last_lightning_cloud_ground'] == True]
        if len(last_cg) == 0:
            end_time = group['date'].iloc[-1]
        else:
            end_time = last_cg['date'].iloc[0]

        buffer = pd.Timedelta(minutes=5)
        mask = (
            (df['airport'] == airport) &
            (df['date'] >= start_time - buffer) &
            (df['date'] <= end_time + buffer)
        )
        all_in_alert = df[mask]

        alerts.append({
            'airport':            airport,
            'airport_alert_id':   alert_id,
            'start_time':         start_time,
            'end_time':           end_time,
            'alert_end_official': end_time + pd.Timedelta(minutes=end_delay_min),
            'lightning_count':    len(all_in_alert),
            'cg_count':           len(group),
        })

    return pd.DataFrame(alerts)


# =============================================================================
# 3. GÉNÉRATION DES ÉCHANTILLONS POUR UNE ALERTE
# =============================================================================

def get_strikes_for_alert(df, airport, start_time, end_time, buffer_min=5):
    """
    Récupère tous les éclairs (tous types) pour un aéroport donné entre
    start_time - buffer_min et end_time + buffer_min.
    """
    start = start_time - pd.Timedelta(minutes=buffer_min)
    end   = end_time   + pd.Timedelta(minutes=buffer_min)
    mask  = (df['airport'] == airport) & (df['date'] >= start) & (df['date'] <= end)
    return df[mask].copy().sort_values('date')


def generate_samples_for_alert(alert_df, df_full, horizon_min=30, freq_min=1, buffer_min=5):
    """
    Pour une alerte donnée (ligne de alerts_df), génère les échantillons
    temporels (un par minute de start_time à alert_end_official).

    FIX #1 : le buffer passé à get_strikes_for_alert est élargi à
              buffer_min + horizon_min pour couvrir toute la fenêtre future
              de chaque instant t de la grille, y compris les derniers instants
              proches de alert_end_official. Sans ça, les labels y sont
              faussement mis à 0 pour les ~horizon_min dernières minutes.
    """
    airport      = alert_df['airport']
    start        = alert_df['start_time']
    end_cg       = alert_df['end_time']
    end_official = alert_df['alert_end_official']

    # Buffer élargi : couvre end_cg + buffer_min + horizon_min
    strikes = get_strikes_for_alert(
        df_full, airport, start, end_cg,
        buffer_min=buffer_min + horizon_min   # ← correction
    )

    times   = pd.date_range(start, end_official, freq=f'{freq_min}min')
    samples = []

    for t in times:
        past   = strikes[strikes['date'] < t]
        future = strikes[
            (strikes['date'] >= t) &
            (strikes['date'] <  t + pd.Timedelta(minutes=horizon_min))
        ]
        y = 1 if len(future) > 0 else 0

        features = compute_features(past, t, airport)
        features['time']             = t
        features['y']                = y
        features['airport_alert_id'] = alert_df['airport_alert_id']
        samples.append(features)

    return pd.DataFrame(samples)


"""
compute_features v2 — Feature engineering enrichi
À remplacer directement dans xgb_storm_corrected.py

Nouvelles features ajoutées par rapport à v1 :
  - Décélération de l'activité (ratio taux récent / taux global)
  - Temps depuis le pic d'activité
  - Tendance amplitude (signal physique de fin d'orage)
  - Ratio IC/CG récent vs global (précurseur fort)
  - Distance récente vs historique (l'orage s'éloigne-t-il ?)
  - Concentration azimutale (orage localisé ou diffus ?)
  - std_amplitude (variabilité de l'intensité)
  - max_interarrival (le plus long silence observé)
  - Heure en minutes depuis minuit (granularité fine vs hour entier)
"""




def compute_features(past_strikes, current_time, airport):
    """
    Calcule les features à partir des éclairs passés et du contexte temporel.
    VERSION 2 — feature engineering enrichi.
    """
    feat = {'airport': airport}

    # ── Contexte temporel cyclique (invariant, calculé dans tous les cas) ──
    feat['month']        = current_time.month
    feat['hour']         = current_time.hour
    feat['month_sin']    = np.sin(2 * np.pi * current_time.month / 12)
    feat['month_cos']    = np.cos(2 * np.pi * current_time.month / 12)
    feat['hour_sin']     = np.sin(2 * np.pi * current_time.hour / 24)
    feat['hour_cos']     = np.cos(2 * np.pi * current_time.hour / 24)
    # Granularité fine : minutes depuis minuit (capte mieux les cycles diurnes)
    minutes_since_midnight = current_time.hour * 60 + current_time.minute
    feat['minute_of_day_sin'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    feat['minute_of_day_cos'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    # ── Cas sans aucun éclair passé ────────────────────────────────────────
    if len(past_strikes) == 0:
        nan_features = [
            'time_since_last', 'time_since_last_cg', 'time_since_last_ic',
            'time_since_first', 'time_since_peak',
            'std_interarrival', 'max_interarrival',
            'mean_dist', 'trend_dist', 'dist_recent_vs_mean',
            'angular_spread', 'azimuth_concentration',
            'prop_ic', 'mean_amplitude', 'std_amplitude', 'amplitude_trend',
            'ic_ratio_trend', 'rate_ratio_5_30', 'deceleration',
        ]
        for f in nan_features:
            feat[f] = np.nan

        zero_features = [
            'strike_count_5min', 'strike_count_10min', 'strike_count_30min',
            'rate_5min', 'rate_10min', 'rate_30min',
        ]
        for f in zero_features:
            feat[f] = 0

        return feat

    # ── Features temporelles de base ──────────────────────────────────────
    last = past_strikes.iloc[-1]
    feat['time_since_last'] = (
        current_time - last['date']
    ).total_seconds() / 60.0

    last_cg = past_strikes[past_strikes['icloud'] == False]
    feat['time_since_last_cg'] = (
        (current_time - last_cg.iloc[-1]['date']).total_seconds() / 60.0
        if len(last_cg) > 0 else np.nan
    )

    last_ic = past_strikes[past_strikes['icloud'] == True]
    feat['time_since_last_ic'] = (
        (current_time - last_ic.iloc[-1]['date']).total_seconds() / 60.0
        if len(last_ic) > 0 else np.nan
    )

    feat['time_since_first'] = (
        current_time - past_strikes.iloc[0]['date']
    ).total_seconds() / 60.0

    # ── Temps depuis le PIC d'activité ────────────────────────────────────
    # On découpe en fenêtres de 5min et on trouve celle avec le plus d'éclairs
    try:
        tmp = past_strikes.set_index('date').resample('5min').size()
        if len(tmp) > 0:
            peak_time = tmp.idxmax()
            feat['time_since_peak'] = (
                current_time - peak_time
            ).total_seconds() / 60.0
        else:
            feat['time_since_peak'] = np.nan
    except Exception:
        feat['time_since_peak'] = np.nan

    # ── Comptages & taux par fenêtre ──────────────────────────────────────
    counts = {}
    for window in [5, 10, 30]:
        mask = past_strikes['date'] >= current_time - pd.Timedelta(minutes=window)
        cnt  = int(mask.sum())
        feat[f'strike_count_{window}min'] = cnt
        feat[f'rate_{window}min']         = cnt / window
        counts[window] = cnt

    # ── Décélération : pente sur les 3 fenêtres ───────────────────────────
    # rate_5min, rate entre 5 et 10, rate entre 10 et 30
    r1 = counts[5]  / 5
    r2 = (counts[10] - counts[5])  / 5
    r3 = (counts[30] - counts[10]) / 20
    feat['deceleration'] = np.polyfit([0, 1, 2], [r1, r2, r3], 1)[0]
    # valeur négative = l'activité décroît → bon signal de fin d'orage

    # ── Ratio taux récent / taux global ───────────────────────────────────
    feat['rate_ratio_5_30'] = counts[5] / (counts[30] + 1e-6)
    # proche de 0 → l'activité récente est faible par rapport au passé

    # ── Inter-arrivées ────────────────────────────────────────────────────
    if len(past_strikes) >= 2:
        intervals = (
            past_strikes['date'].diff().dt.total_seconds().dropna() / 60.0
        )
        feat['std_interarrival'] = intervals.std()
        feat['max_interarrival'] = intervals.max()
        # grand max_interarrival = long silence déjà observé → fin probable
    else:
        feat['std_interarrival'] = np.nan
        feat['max_interarrival'] = np.nan

    # ── Spatiales ─────────────────────────────────────────────────────────
    feat['mean_dist'] = past_strikes['dist'].mean()

    recent = past_strikes.tail(min(5, len(past_strikes)))
    if len(recent) >= 2:
        slope = np.polyfit(np.arange(len(recent)), recent['dist'].values, 1)[0]
        feat['trend_dist'] = slope
        # positif → éclairs de plus en plus loin → orage qui s'éloigne
    else:
        feat['trend_dist'] = 0

    # Distance récente vs historique
    feat['dist_recent_vs_mean'] = recent['dist'].mean() - past_strikes['dist'].mean()
    # positif → éclairs récents plus loin que la moyenne → orage s'éloigne

    feat['angular_spread'] = past_strikes['azimuth'].std()

    # Concentration azimutale : proche de 1 = orage très directionnel
    feat['azimuth_concentration'] = float(
        np.sqrt(
            np.cos(np.deg2rad(past_strikes['azimuth'])).mean() ** 2 +
            np.sin(np.deg2rad(past_strikes['azimuth'])).mean() ** 2
        )
    )

    # ── Physiques ─────────────────────────────────────────────────────────
    feat['prop_ic']       = past_strikes['icloud'].mean()
    feat['mean_amplitude'] = past_strikes['amplitude'].mean()
    feat['std_amplitude']  = past_strikes['amplitude'].std()

    # Tendance amplitude sur les 10 derniers éclairs
    recent10 = past_strikes.tail(min(10, len(past_strikes)))
    if len(recent10) >= 2:
        feat['amplitude_trend'] = np.polyfit(
            np.arange(len(recent10)),
            recent10['amplitude'].values,
            1
        )[0]
        # négatif → amplitude décroissante → orage qui faiblit
    else:
        feat['amplitude_trend'] = 0

    # Ratio IC/CG récent vs global
    recent_ic  = past_strikes.tail(10)['icloud'].mean()
    global_ic  = past_strikes['icloud'].mean()
    feat['ic_ratio_trend'] = recent_ic / (global_ic + 1e-6)
    # < 1 → proportion IC récente plus faible → signal de fin d'orage

    return feat


def prepare_dataset(df, alerts_df, horizon_min=30, freq_min=1, buffer_min=5):
    """
    Parcourt toutes les alertes et construit le dataset d'entraînement.
    horizon_min est transmis jusqu'à generate_samples_for_alert qui l'utilise
    à la fois pour définir la cible y ET pour élargir le buffer de récupération
    des éclairs (fix #1).
    """
    all_samples = []
    for idx, alert_row in alerts_df.iterrows():
        samples = generate_samples_for_alert(
            alert_row, df,
            horizon_min=horizon_min,
            freq_min=freq_min,
            buffer_min=buffer_min
        )
        all_samples.append(samples)
        if (idx + 1) % 10 == 0:
            print(f"Alertes traitées : {idx+1}/{len(alerts_df)}")
    return pd.concat(all_samples, ignore_index=True)


# =============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE
# =============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle XGBoost avec early stopping (API native Booster).
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    params = {
        'objective':        'binary:logistic',
        'max_depth':        6,
        'learning_rate':    0.05,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'eval_metric':      'logloss',
        'seed':             42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    return model

