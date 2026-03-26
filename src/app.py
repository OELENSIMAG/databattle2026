"""
app.py — Démo Streamlit : Prédiction de fin d'orage
====================================================
Lance avec : streamlit run app.py

Prérequis dans le même dossier :
  - modele_fin_orage.json         (XGBoost sauvegardé)
  - label_encoder_airport.pkl     (LabelEncoder)
  - segment_alerts_all_airports_train.csv  (données)

Install :
  pip install streamlit xgboost pandas numpy folium streamlit-folium
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
import folium
from pathlib import Path
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / 'generated'
TRAIN_DATA_PATH = BASE_DIR.parent / 'data' / 'data_train_databattle2026' / 'segment_alerts_all_airports_train.csv'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Meteorage — Prédiction fin d'orage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé — thème sombre industriel
st.markdown("""
<style>
    /* Fond général */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Titres */
    h1, h2, h3 { color: #58a6ff; font-family: 'Courier New', monospace; }

    /* Métriques */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }

    /* Boutons */
    .stButton > button {
        background-color: #1f6feb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover { background-color: #388bfd; }

    /* Tableaux */
    .stDataFrame { border: 1px solid #30363d; border-radius: 6px; }

    /* Info boxes */
    .info-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .prob-display {
        font-size: 3.5rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        text-align: center;
        padding: 10px;
    }
    .confidence-high   { color: #3fb950; }
    .confidence-medium { color: #d29922; }
    .confidence-low    { color: #f85149; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COORDONNÉES DES AÉROPORTS (pour la carte)
# ─────────────────────────────────────────────────────────────────────────────

AIRPORT_COORDS = {
    'Ajaccio':  (41.9236,  8.8029),
    'Bastia':   (42.5527,  9.4837),
    'Biarritz': (43.4683, -1.5240),
    'Bron':     (45.7294,  4.9389),
    'Nantes':   (47.1532, -1.6107),
    'Pise':     (43.6950, 10.3990),
}


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ET DU MODÈLE (mis en cache)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = GENERATED_DIR / 'modele_fin_orage.json'
    label_encoder_path = GENERATED_DIR / 'label_encoder_airport.pkl'
    feature_cols_path = GENERATED_DIR / 'feature_cols.pkl'

    missing_files = [
        str(path) for path in [model_path, label_encoder_path, feature_cols_path]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            'Fichiers modèle introuvables: ' + ', '.join(missing_files)
        )

    model = xgb.Booster()
    model.load_model(str(model_path))
    le           = joblib.load(label_encoder_path)
    feature_cols = joblib.load(feature_cols_path)
    return model, le, feature_cols


@st.cache_data
def load_data():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset introuvable: {TRAIN_DATA_PATH}')

    df = pd.read_csv(TRAIN_DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # Filtrer Pise 2016 IC
    mask = (
        (df['airport'] == 'Pise') &
        (df['date'].dt.year == 2016) &
        (df['icloud'] == True)
    )
    df = df[~mask].reset_index(drop=True)
    return df


@st.cache_data
def build_alerts_cached(df_hash):
    """Reconstruit la table des alertes depuis les données."""
    df = load_data()
    cg = df[(df['icloud'] == False) & (df['airport_alert_id'].notna())].copy()
    cg['airport_alert_id'] = cg['airport_alert_id'].astype(int)

    alerts = []
    for (airport, alert_id), group in cg.groupby(['airport', 'airport_alert_id']):
        group = group.sort_values('date')
        start = group['date'].iloc[0]
        last_flag = group[group['is_last_lightning_cloud_ground'] == True]
        end = last_flag['date'].iloc[0] if len(last_flag) > 0 else group['date'].iloc[-1]
        alerts.append({
            'airport':            airport,
            'airport_alert_id':   alert_id,
            'start_time':         start,
            'end_time':           end,
            'alert_end_official': end + pd.Timedelta(minutes=30),
            'duration_min':       (end - start).total_seconds() / 60,
        })
    return pd.DataFrame(alerts)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (même fonction que xgb_storm_corrected.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(past_strikes, current_time, airport, airport_encoded):
    feat = {}

    # Contexte temporel (calculé dans tous les cas)
    feat['month']     = current_time.month
    feat['hour']      = current_time.hour
    feat['month_sin'] = np.sin(2 * np.pi * current_time.month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * current_time.month / 12)
    feat['hour_sin']  = np.sin(2 * np.pi * current_time.hour / 24)
    feat['hour_cos']  = np.cos(2 * np.pi * current_time.hour / 24)
    minutes_since_midnight = current_time.hour * 60 + current_time.minute
    feat['minute_of_day_sin'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    feat['minute_of_day_cos'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)
    feat['airport_encoded'] = airport_encoded

    nan_features = [
        'time_since_last', 'time_since_last_cg', 'time_since_last_ic',
        'time_since_first', 'time_since_peak',
        'std_interarrival', 'max_interarrival',
        'mean_dist', 'trend_dist', 'dist_recent_vs_mean',
        'angular_spread', 'azimuth_concentration',
        'prop_ic', 'mean_amplitude', 'std_amplitude', 'amplitude_trend',
        'ic_ratio_trend', 'rate_ratio_5_30', 'deceleration',
    ]
    zero_features = [
        'strike_count_5min', 'strike_count_10min', 'strike_count_30min',
        'rate_5min', 'rate_10min', 'rate_30min',
    ]

    if len(past_strikes) == 0:
        for f in nan_features:  feat[f] = np.nan
        for f in zero_features: feat[f] = 0
        return feat

    # Temporelles de base
    last = past_strikes.iloc[-1]
    feat['time_since_last'] = (current_time - last['date']).total_seconds() / 60.0

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

    # Temps depuis le pic
    try:
        tmp = past_strikes.set_index('date').resample('5min').size()
        feat['time_since_peak'] = (
            (current_time - tmp.idxmax()).total_seconds() / 60.0
            if len(tmp) > 0 else np.nan
        )
    except Exception:
        feat['time_since_peak'] = np.nan

    # Comptages & taux
    counts = {}
    for w in [5, 10, 30]:
        mask = past_strikes['date'] >= current_time - pd.Timedelta(minutes=w)
        cnt  = int(mask.sum())
        feat[f'strike_count_{w}min'] = cnt
        feat[f'rate_{w}min']         = cnt / w
        counts[w] = cnt

    # Décélération
    r1 = counts[5]  / 5
    r2 = (counts[10] - counts[5])  / 5
    r3 = (counts[30] - counts[10]) / 20
    feat['deceleration']   = np.polyfit([0, 1, 2], [r1, r2, r3], 1)[0]
    feat['rate_ratio_5_30'] = counts[5] / (counts[30] + 1e-6)

    # Inter-arrivées
    if len(past_strikes) >= 2:
        intervals = past_strikes['date'].diff().dt.total_seconds().dropna() / 60.0
        feat['std_interarrival'] = intervals.std()
        feat['max_interarrival'] = intervals.max()
    else:
        feat['std_interarrival'] = np.nan
        feat['max_interarrival'] = np.nan

    # Spatiales
    feat['mean_dist'] = past_strikes['dist'].mean()
    recent = past_strikes.tail(min(5, len(past_strikes)))
    feat['trend_dist'] = (
        np.polyfit(np.arange(len(recent)), recent['dist'].values, 1)[0]
        if len(recent) >= 2 else 0
    )
    feat['dist_recent_vs_mean'] = recent['dist'].mean() - past_strikes['dist'].mean()
    feat['angular_spread']      = past_strikes['azimuth'].std()
    feat['azimuth_concentration'] = float(np.sqrt(
        np.cos(np.deg2rad(past_strikes['azimuth'])).mean() ** 2 +
        np.sin(np.deg2rad(past_strikes['azimuth'])).mean() ** 2
    ))

    # Physiques
    feat['prop_ic']        = past_strikes['icloud'].mean()
    feat['mean_amplitude'] = past_strikes['amplitude'].mean()
    feat['std_amplitude']  = past_strikes['amplitude'].std()
    recent10 = past_strikes.tail(min(10, len(past_strikes)))
    feat['amplitude_trend'] = (
        np.polyfit(np.arange(len(recent10)), recent10['amplitude'].values, 1)[0]
        if len(recent10) >= 2 else 0
    )
    feat['ic_ratio_trend'] = (
        past_strikes.tail(10)['icloud'].mean() / (past_strikes['icloud'].mean() + 1e-6)
    )

    return feat

# ─────────────────────────────────────────────────────────────────────────────
# SCORE DE CONFIANCE (Option B + trajectoire)
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(past_strikes, current_time, proba, alert_start):
    """
    Retourne un score de confiance entre 0 et 1, et un message explicatif.

    4 composants :
      1. Durée de l'alerte (contexte disponible)
      2. Zone de probabilité (le modèle est-il tranché ?)
      3. Typicité de l'orage (ratio IC/CG, nombre d'éclairs)
      4. Trajectoire apparente (l'orage s'éloigne-t-il ?)
    """
    scores   = []
    messages = []

    # ── 1. Durée de l'alerte ─────────────────────────────────────────────
    duration = (current_time - alert_start).total_seconds() / 60.0
    if duration >= 30:
        scores.append(1.0)
        messages.append(f"✓ Alerte longue ({duration:.0f} min) — contexte riche")
    elif duration >= 15:
        scores.append(0.6)
        messages.append(f"~ Alerte de durée moyenne ({duration:.0f} min)")
    else:
        scores.append(0.2)
        messages.append(f"⚠ Alerte courte ({duration:.0f} min) — peu de contexte")

    # ── 2. Zone de probabilité ───────────────────────────────────────────
    if proba < 0.05 or proba > 0.90:
        scores.append(1.0)
        messages.append("✓ Probabilité très tranchée")
    elif proba < 0.15 or proba > 0.75:
        scores.append(0.7)
        messages.append("~ Probabilité assez claire")
    elif 0.35 <= proba <= 0.65:
        scores.append(0.2)
        messages.append("⚠ Zone d'incertitude — modèle hésite")
    else:
        scores.append(0.5)

    # ── 3. Typicité de l'orage ───────────────────────────────────────────
    if len(past_strikes) == 0:
        scores.append(0.3)
        messages.append("⚠ Aucun éclair observé — contexte vide")
    else:
        n_cg = (past_strikes['icloud'] == False).sum()
        n_ic = (past_strikes['icloud'] == True).sum()
        total = len(past_strikes)

        if n_cg >= 5:
            scores.append(0.9)
            messages.append(f"✓ Suffisamment d'éclairs CG ({n_cg})")
        elif n_cg >= 2:
            scores.append(0.6)
            messages.append(f"~ Peu d'éclairs CG ({n_cg}) — pattern moins connu")
        else:
            scores.append(0.2)
            messages.append(f"⚠ Très peu d'éclairs CG ({n_cg}) — orage atypique")

    # ── 4. Trajectoire apparente ─────────────────────────────────────────
    if len(past_strikes) >= 5:
        recent = past_strikes.tail(10)

        # Vitesse d'éloignement
        trend_dist = 0
        if len(recent) >= 3:
            trend_dist = np.polyfit(
                np.arange(len(recent)), recent['dist'].values, 1
            )[0]

        # Cohérence de direction
        azimuth_std = recent['azimuth'].std() if len(recent) >= 3 else 180

        # Distance actuelle
        mean_dist_recent = recent['dist'].mean()

        if trend_dist > 0.3 and azimuth_std < 45:
            scores.append(1.0)
            messages.append(
                f"✓ Orage s'éloigne de façon régulière "
                f"(+{trend_dist:.1f} km/éclair, direction stable)"
            )
        elif trend_dist > 0:
            scores.append(0.7)
            messages.append(f"~ Orage s'éloigne légèrement")
        elif trend_dist < -0.3:
            scores.append(0.1)
            messages.append(
                f"⚠ Orage se rapproche ({trend_dist:.1f} km/éclair) — risque élevé"
            )
        elif azimuth_std > 90:
            scores.append(0.3)
            messages.append("⚠ Trajectoire chaotique — orage diffus imprévisible")
        else:
            scores.append(0.5)
            messages.append(f"~ Orage stationnaire (dist. moy. {mean_dist_recent:.1f} km)")
    else:
        scores.append(0.4)
        messages.append("~ Trop peu d'éclairs pour estimer la trajectoire")

    # ── Score final ───────────────────────────────────────────────────────
    final_score = np.mean(scores)

    if final_score >= 0.70:
        level = "ÉLEVÉE"
        css   = "confidence-high"
        emoji = "🟢"
    elif final_score >= 0.45:
        level = "MODÉRÉE"
        css   = "confidence-medium"
        emoji = "🟡"
    else:
        level = "FAIBLE"
        css   = "confidence-low"
        emoji = "🔴"

    return final_score, level, css, emoji, messages


# ─────────────────────────────────────────────────────────────────────────────
# CARTE FOLIUM
# ─────────────────────────────────────────────────────────────────────────────

def make_map(airport, strikes_so_far):
    lat, lon = AIRPORT_COORDS.get(airport, (45.0, 5.0))

    m = folium.Map(
        location=[lat, lon],
        zoom_start=9,
        tiles='CartoDB dark_matter',
    )

    # Aéroport
    folium.Marker(
        [lat, lon],
        popup=f"✈ {airport}",
        icon=folium.Icon(color='blue', icon='plane', prefix='fa'),
    ).add_to(m)

    # Cercle 30km
    folium.Circle(
        [lat, lon], radius=30000,
        color='#1f6feb', fill=False,
        weight=1, opacity=0.5,
        tooltip="Zone de surveillance (30 km)"
    ).add_to(m)

    # Éclairs
    for _, row in strikes_so_far.iterrows():
        color  = '#f85149' if not row['icloud'] else '#d29922'
        radius = max(3, min(8, abs(row['amplitude']) / 10))
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=(
                f"{'CG' if not row['icloud'] else 'IC'} | "
                f"{row['date'].strftime('%H:%M:%S')} | "
                f"{row['dist']:.1f} km | "
                f"{row['amplitude']:.0f} kA"
            ),
        ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# BARRE DE PROBABILITÉ COLORÉE
# ─────────────────────────────────────────────────────────────────────────────

def proba_color(p):
    if p > 0.50:
        return "#f85149"   # rouge
    elif p > 0.20:
        return "#d29922"   # orange
    elif p > 0.10:
        return "#e3b341"   # jaune
    else:
        return "#3fb950"   # vert


def render_proba_bar(proba):
    color = proba_color(proba)
    pct   = int(proba * 100)
    st.markdown(f"""
    <div class="info-box">
        <div style="font-size:0.85rem; color:#8b949e; margin-bottom:6px;">
            PROBABILITÉ D'UN ÉCLAIR DANS LES 30 PROCHAINES MINUTES
        </div>
        <div class="prob-display" style="color:{color};">{pct}%</div>
        <div style="background:#21262d; border-radius:4px; height:14px; margin-top:8px;">
            <div style="
                width:{pct}%;
                background:{color};
                height:14px;
                border-radius:4px;
                transition: width 0.5s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# APP PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="font-size:2rem;">⚡</span>
        <div>
            <h1 style="margin:0; font-size:1.6rem;">
                METEORAGE — Prédiction de fin d'orage
            </h1>
            <p style="color:#8b949e; margin:0; font-size:0.85rem;">
                Outil d'aide à la décision · Data Battle 2026
            </p>
        </div>
    </div>
    <hr style="border-color:#30363d; margin-bottom:16px;">
    """, unsafe_allow_html=True)

    # ── Chargement ───────────────────────────────────────────────────────
    try:
        model, le, feature_cols = load_model()
        df        = load_data()
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e}")
        st.info("Place `modele_fin_orage.json`, `label_encoder_airport.pkl` "
                "et `segment_alerts_all_airports_train.csv` dans le même dossier que app.py")
        st.stop()


    alerts = build_alerts_cached(len(df))

    # ── SIDEBAR ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres")
        st.markdown("---")

        # Sélection aéroport
        airports_dispo = sorted(alerts['airport'].unique())
        airport = st.selectbox("✈️ Aéroport", airports_dispo)

        # Filtrer les alertes de cet aéroport
        alerts_ap = alerts[alerts['airport'] == airport].copy()
        alerts_ap['label'] = alerts_ap.apply(
            lambda r: (
                f"Alerte #{int(r['airport_alert_id'])} — "
                f"{r['start_time'].strftime('%d/%m/%Y %H:%M')} "
                f"({r['duration_min']:.0f} min)"
            ),
            axis=1
        )

        alert_label = st.selectbox(
            "🌩️ Alerte",
            alerts_ap['label'].tolist(),
        )
        alert_row = alerts_ap[alerts_ap['label'] == alert_label].iloc[0]

        st.markdown("---")

        # Vitesse de replay
        speed = st.select_slider(
            "⏩ Vitesse du replay",
            options=[0.5, 1., 1.5, 2.0, 3.0],
            value=0.5,
            format_func=lambda x: f"{x}s / minute",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#8b949e;">
        <b>Légende carte</b><br>
        🔴 Éclair nuage-sol (CG)<br>
        🟡 Éclair intra-nuage (IC)<br>
        🔵 Zone de surveillance 30km
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        run_btn = st.button("▶ Lancer le replay", use_container_width=True)

    # ── LAYOUT PRINCIPAL ──────────────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1], gap="medium")

    # Récupérer les éclairs de cette alerte
    buffer = pd.Timedelta(minutes=35)
    mask_ap = (
        (df['airport'] == airport) &
        (df['date'] >= alert_row['start_time'] - pd.Timedelta(minutes=5)) &
        (df['date'] <= alert_row['alert_end_official'] + buffer)
    )
    strikes_alert = df[mask_ap].copy().sort_values('date').reset_index(drop=True)

    # Grille temporelle (1 minute par pas)
    times = pd.date_range(
        alert_row['start_time'],
        alert_row['alert_end_official'],
        freq='1min'
    )

    airport_enc = int(le.transform([airport])[0])

    # ── Affichage statique initial ────────────────────────────────────────
    with col_left:
        st.markdown(f"""
        <div class="info-box">
            <div style="color:#8b949e; font-size:0.8rem;">ALERTE SÉLECTIONNÉE</div>
            <div style="font-size:1.1rem; font-weight:bold; margin-top:4px;">
                {airport} — #{int(alert_row['airport_alert_id'])}
            </div>
            <div style="color:#8b949e; font-size:0.85rem; margin-top:4px;">
                Début : {alert_row['start_time'].strftime('%d/%m/%Y %H:%M')} |
                Dernier CG : {alert_row['end_time'].strftime('%H:%M')} |
                Fin règle 30min : {alert_row['alert_end_official'].strftime('%H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        proba_placeholder    = st.empty()
        conf_placeholder     = st.empty()
        metrics_placeholder  = st.empty()
        timeline_placeholder = st.empty()

    with col_right:
        map_placeholder    = st.empty()
        table_placeholder  = st.empty()

    # ── Replay ────────────────────────────────────────────────────────────
    if run_btn:
        proba_history    = []
        time_history     = []
        strikes_shown    = pd.DataFrame()
        lift_time_model  = None

        for t in times:
            past = strikes_alert[strikes_alert['date'] < t].copy()

            # ── Probabilité du modèle ─────────────────────────────────
            feats = compute_features(past, t, airport, airport_enc)
            X_row = pd.DataFrame([feats])[feature_cols]  # ordre garanti
            dm    = xgb.DMatrix(X_row, feature_names=feature_cols)
            proba = float(model.predict(dm)[0])

            # ── Score de confiance ────────────────────────────────────
            conf_score, conf_level, conf_css, conf_emoji, conf_msgs = (
                compute_confidence(past, t, proba, alert_row['start_time'])
            )

            proba_history.append(proba)
            time_history.append(t)

            # Éclairs apparus à cette minute
            new_strikes = strikes_alert[
                (strikes_alert['date'] >= t - pd.Timedelta(minutes=1)) &
                (strikes_alert['date'] <  t)
            ]
            if len(new_strikes) > 0:
                strikes_shown = pd.concat(
                    [strikes_shown, new_strikes]
                ).drop_duplicates()

            # ── Mise à jour colonne gauche ────────────────────────────
            with col_left:
                # Probabilité
                with proba_placeholder.container():
                    render_proba_bar(proba)
                    st.markdown(
                        f"<div style='text-align:center; color:#8b949e; "
                        f"font-size:0.8rem;'>t = {t.strftime('%H:%M')}</div>",
                        unsafe_allow_html=True
                    )

                # Confiance
                with conf_placeholder.container():
                    st.markdown(f"""
                    <div class="info-box">
                        <div style="color:#8b949e; font-size:0.8rem;">
                            SCORE DE CONFIANCE
                        </div>
                        <div class="{conf_css}" style="font-size:1.3rem;
                             font-weight:bold; margin:6px 0;">
                            {conf_emoji} {conf_level}
                            <span style="font-size:0.9rem; margin-left:8px;">
                                ({conf_score:.0%})
                            </span>
                        </div>
                        <div style="font-size:0.78rem; color:#8b949e;
                             line-height:1.6;">
                            {'<br>'.join(conf_msgs)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Métriques
                with metrics_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Éclairs CG",
                              int((past['icloud'] == False).sum()))
                    m2.metric("Éclairs IC",
                              int((past['icloud'] == True).sum()))
                    m3.metric("Dist. moy.",
                              f"{past['dist'].mean():.1f} km"
                              if len(past) > 0 else "—")

                # Timeline
                with timeline_placeholder.container():
                    if len(proba_history) > 1:
                        df_plot = pd.DataFrame({
                            'Heure':       time_history,
                            'Probabilité': proba_history,
                        }).set_index('Heure')
                        st.line_chart(
                            df_plot,
                            color="#1f6feb",
                            height=150,
                        )

            # ── Mise à jour colonne droite ────────────────────────────
            with col_right:
                # Carte
                with map_placeholder.container():
                    if len(strikes_shown) > 0:
                        m = make_map(airport, strikes_shown)
                        st_folium(m, width=420, height=320,
                                  returned_objects=[],
                                  key=f"map_{t.strftime('%Y%m%d%H%M')}")

                # Tableau des derniers éclairs
                with table_placeholder.container():
                    if len(strikes_shown) > 0:
                        display = strikes_shown.tail(8)[[
                            'date', 'icloud', 'dist', 'azimuth', 'amplitude'
                        ]].copy()
                        display['date']    = display['date'].dt.strftime('%H:%M:%S')
                        display['Type']    = display['icloud'].map(
                            {True: '☁ IC', False: '⚡ CG'}
                        )
                        display['dist']    = display['dist'].round(1)
                        display['azimuth'] = display['azimuth'].round(0)
                        display['amplitude'] = display['amplitude'].round(0)
                        display = display.rename(columns={
                            'date': 'Heure', 'dist': 'Dist (km)',
                            'azimuth': 'Az (°)', 'amplitude': 'Amp (kA)'
                        })[['Heure', 'Type', 'Dist (km)', 'Az (°)', 'Amp (kA)']]
                        st.dataframe(
                            display,
                            hide_index=True,
                            use_container_width=True,
                            height=200,
                        )

            time.sleep(speed)

        # ── Résumé final ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Résumé de l'alerte")

        res_col1, res_col2, res_col3 = st.columns(3)

        # Probabilité minimale atteinte
        proba_min = min(proba_history)
        t_min     = time_history[proba_history.index(proba_min)]

        res_col1.metric(
            "Probabilité minimale atteinte",
            f"{proba_min:.0%}",
            f"à {t_min.strftime('%H:%M')}",
        )

        # Moment règle 30min
        res_col2.metric(
            "Fin selon règle 30min",
            alert_row['alert_end_official'].strftime('%H:%M'),
            f"Dernier CG : {alert_row['end_time'].strftime('%H:%M')}",
        )

        # Gain potentiel à seuil 10%
        seuil_demo = 0.10
        lift_times = [
            t for t, p in zip(time_history, proba_history)
            if t > alert_row['end_time'] and p < seuil_demo
        ]
        if lift_times:
            lift = lift_times[0]
            gain = (alert_row['alert_end_official'] - lift).total_seconds() / 60.0
            if gain > 0:
                res_col3.metric(
                    f"Gain potentiel (seuil {seuil_demo:.0%})",
                    f"{gain:.0f} min",
                    f"Modèle < {seuil_demo:.0%} dès {lift.strftime('%H:%M')}",
                    delta_color="normal",
                )
            else:
                res_col3.metric(
                    f"Gain potentiel (seuil {seuil_demo:.0%})",
                    "Aucun",
                    "Modèle plus tardif que la règle",
                    delta_color="off",
                )
        else:
            res_col3.metric(
                f"Gain potentiel (seuil {seuil_demo:.0%})",
                "—",
                "Proba jamais descendue sous 10%",
                delta_color="off",
            )

        # Courbe finale complète
        st.markdown("#### Évolution de la probabilité — alerte complète")
        df_final = pd.DataFrame({
            'Heure':       time_history,
            'Probabilité': proba_history,
        }).set_index('Heure')
        st.line_chart(df_final, color="#1f6feb", height=200)

        st.success("✅ Replay terminé")

    else:
        # Affichage initial avant le replay
        with col_left:
            with proba_placeholder.container():
                st.markdown("""
                <div class="info-box" style="text-align:center; padding:32px;">
                    <div style="color:#8b949e;">
                        Appuie sur <b>▶ Lancer le replay</b> pour démarrer
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            with map_placeholder.container():
                # Carte vide centrée sur l'aéroport
                lat, lon = AIRPORT_COORDS.get(airport, (45.0, 5.0))
                m = folium.Map(
                    location=[lat, lon],
                    zoom_start=9,
                    tiles='CartoDB dark_matter'
                )
                folium.Marker(
                    [lat, lon],
                    popup=f"✈ {airport}",
                    icon=folium.Icon(color='blue', icon='plane', prefix='fa'),
                ).add_to(m)
                folium.Circle(
                    [lat, lon], radius=30000,
                    color='#1f6feb', fill=False,
                    weight=1, opacity=0.5,
                ).add_to(m)
                st_folium(m, width=420, height=320, returned_objects=[])


if __name__ == '__main__':
    main()