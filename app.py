"""
AURUM - Delhi Parking Forecast
Single-file Streamlit application combining the dashboard and forecasting model.
Run with: streamlit run app.py
Required data files:
  data/occupancy_birmingham.csv
  data/parking_sessions.csv
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# Configuration and Delhi parking locations (formerly aurum/config.py
# and aurum/places.py)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "models" / "occupancy_model.joblib"
IST = ZoneInfo("Asia/Kolkata")
WINDOW = (7.0, 17.5)
SLOT_STEP = 0.5
MAX_RATIO = 0.99

@dataclass(frozen=True)
class Place:
    id: str
    name: str
    area: str
    capacity: int
    archetype: str
    lat: float
    lon: float
    kind: str

PLACES = (
    Place("cp", "Connaught Place Inner Circle", "Central Delhi", 1200, "BHMNCPLDH01", 28.6315, 77.2167, "Commercial"),
    Place("citywalk", "Select Citywalk", "Saket", 1500, "BHMBRCBRG01", 28.5284, 77.2193, "Mall"),
    Place("promenade", "DLF Promenade", "Vasant Kunj", 1800, "Bull Ring", 28.5413, 77.1565, "Mall"),
    Place("khan", "Khan Market", "Lodhi Road", 600, "BHMBCCSNH01", 28.6003, 77.2270, "Market"),
    Place("indiagate", "India Gate Lawns", "Rajpath", 900, "BHMBCCTHL01", 28.6129, 77.2295, "Leisure"),
    Place("nehru", "Nehru Place Business Hub", "South Delhi", 1000, "BHMEURBRD01", 28.5491, 77.2533, "Office"),
    Place("ndls", "New Delhi Railway Station", "Paharganj", 1100, "BHMBRTARC01", 28.6429, 77.2192, "Transit"),
    Place("igi", "IGI Airport Terminal 3", "Aerocity", 3500, "BHMMBMMBX01", 28.5562, 77.1000, "Transit"),
    Place("mandapam", "Bharat Mandapam", "Pragati Maidan", 2500, "BHMNCPNST01", 28.6177, 77.2431, "Events"),
    Place("hauzkhas", "Hauz Khas Village", "South Delhi", 450, "Others-CCCPS133", 28.5535, 77.1930, "Leisure"),
    Place("qutub", "Qutub Minar", "Mehrauli", 500, "BHMBCCPST01", 28.5245, 77.1855, "Heritage"),
    Place("redfort", "Red Fort - Chandni Chowk", "Old Delhi", 700, "Others-CCCPS105a", 28.6562, 77.2410, "Heritage"),
)
BY_ID = {p.id: p for p in PLACES}

@dataclass(frozen=True)
class Prediction:
    place_id: str
    ratio: float
    vehicles: int
    low: int
    high: int
    capacity: int
    in_window: bool

    @property
    def status(self) -> str:
        if self.ratio < 0.5:
            return "Calm - easy parking"
        if self.ratio < 0.75:
            return "Moderately busy"
        if self.ratio < 0.9:
            return "Very busy"
        return "Near capacity"

# ---------------------------------------------------------------------
# Forecasting model (combined from aurum/model.py)
# ---------------------------------------------------------------------
def load_occupancy(path: Path = DATA / "occupancy_birmingham.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Upload the data folder from the project ZIP.")
    d = pd.read_csv(path, parse_dates=["LastUpdated"])
    d = d[(d.Occupancy >= 0) & (d.Capacity > 0)].copy()
    d["ratio"] = (d.Occupancy / d.Capacity).clip(0, 1.05)
    d["t"] = d.LastUpdated.dt.hour + d.LastUpdated.dt.minute / 60
    d["dow"] = d.LastUpdated.dt.dayofweek
    d["lot"] = d.SystemCodeNumber.astype("category")
    return d

def behaviour_modifiers(path: Path = DATA / "parking_sessions.csv") -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Upload the data folder from the project ZIP.")
    b = pd.read_csv(path)

    def ratio(mask: pd.Series) -> float:
        numerator = b.loc[mask, "duration_hours"].mean()
        denominator = b.loc[~mask, "duration_hours"].mean()
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return 1.0
        return round(float(numerator / denominator), 3)

    return {
        "rain": ratio(b.weather == "Rainy"),
        "holiday": ratio(b.is_holiday == 1),
        "peak_event": ratio(b.is_exam_day == 1),
    }

def _regressor():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, categorical_features=[2], random_state=7
    )

def _frame(t, dow, lot_code):
    return pd.DataFrame({"t": t, "dow": dow, "lot": lot_code})

def train_model() -> dict:
    d = load_occupancy()
    codes = d.lot.cat.codes
    cut = d.LastUpdated.quantile(0.8)
    train_mask, test_mask = d.LastUpdated <= cut, d.LastUpdated > cut
    probe = _regressor().fit(
        _frame(d.t[train_mask], d.dow[train_mask], codes[train_mask]),
        d.ratio[train_mask],
    )
    predicted = np.clip(
        probe.predict(_frame(d.t[test_mask], d.dow[test_mask], codes[test_mask])), 0, 1
    )
    metrics = {
        "holdout_mae": round(float(mean_absolute_error(d.ratio[test_mask], predicted)), 4),
        "holdout_r2": round(float(r2_score(d.ratio[test_mask], predicted)), 4),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "lots": int(d.lot.nunique()),
    }
    full = _regressor().fit(_frame(d.t, d.dow, codes), d.ratio)
    residuals = d.ratio - np.clip(full.predict(_frame(d.t, d.dow, codes)), 0, 1)
    bundle = {
        "model": full,
        "lots": list(d.lot.cat.categories),
        "sigma": round(float(residuals.std()), 4),
        "mods": behaviour_modifiers(),
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle

class Forecaster:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.lots = bundle["lots"]
        self.sigma = bundle["sigma"]
        self.mods = bundle["mods"]
        self.metrics = bundle["metrics"]

    @classmethod
    def load(cls):
        if not MODEL_PATH.exists():
            return cls(train_model())
        try:
            return cls(joblib.load(MODEL_PATH))
        except Exception:
            # Retrain if the saved model is incompatible with the installed sklearn version.
            return cls(train_model())

    def multiplier(self, rain=False, holiday=False, peak_event=False):
        value = 1.0
        for enabled, key in ((rain, "rain"), (holiday, "holiday"), (peak_event, "peak_event")):
            if enabled:
                value *= self.mods[key]
        return value

    def _base(self, place_id: str, times: np.ndarray, dow: int) -> np.ndarray:
        place = BY_ID[place_id]
        if place.archetype not in self.lots:
            raise ValueError(
                f"Training data does not contain parking archetype '{place.archetype}'."
            )
        lot_code = self.lots.index(place.archetype)
        clipped_times = np.clip(times, *WINDOW)
        return np.clip(
            self.model.predict(_frame(clipped_times, np.full(len(clipped_times), dow),
                                      np.full(len(clipped_times), lot_code))),
            0, MAX_RATIO,
        )

    def curve(self, place_id: str, day: date, times=None, **conditions):
        place = BY_ID[place_id]
        ts = np.asarray(
            times if times is not None else np.arange(WINDOW[0], WINDOW[1] + 1e-9, SLOT_STEP),
            dtype=float,
        )
        ratios = np.minimum(
            self._base(place_id, ts, day.weekday()) * self.multiplier(**conditions), MAX_RATIO
        )
        half_range = 1.28 * self.sigma * place.capacity
        results = []
        for t, ratio in zip(ts, ratios):
            vehicles = round(float(ratio) * place.capacity)
            results.append(Prediction(
                place_id, float(ratio), vehicles,
                max(0, round(vehicles - half_range)),
                min(place.capacity, round(vehicles + half_range)),
                place.capacity, WINDOW[0] <= t <= WINDOW[1],
            ))
        return results

    def predict(self, place_id: str, when: datetime, **conditions):
        hour = when.hour + when.minute / 60
        return self.curve(place_id, when.date(), times=[hour], **conditions)[0]

    def week_matrix(self, place_id: str, **conditions):
        times = np.arange(WINDOW[0], WINDOW[1] + 1e-9, SLOT_STEP)
        multiplier = self.multiplier(**conditions)
        return np.array([
            np.minimum(self._base(place_id, times, weekday) * multiplier, MAX_RATIO)
            for weekday in range(7)
        ])

@st.cache_resource(show_spinner="Training/loading parking forecast model...")
def get_forecaster():
    return Forecaster.load()

# ---------------------------------------------------------------------
# Streamlit dashboard (combined from aurum/dashboard.py)
# ---------------------------------------------------------------------
GOLD = "#c9a96a"
GOLD_LIGHT = "#e8d3a2"
MUTED = "#9a9080"
BACKGROUND = "#0b0a09"
FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;500"
    "&family=Jost:wght@300;400&display=swap"
)
CSS = """
<style>
@import url('__FONT_URL__');
html, body, [class*="css"] {font-family:'Jost',sans-serif;font-weight:300;}
h1,h2,h3 {font-family:'Cormorant Garamond',serif !important;font-weight:300 !important;}
h1 {font-size:3.2rem !important;color:#e8d3a2;}
[data-testid="stMetricValue"] {font-family:'Cormorant Garamond',serif;font-size:2.7rem;color:#e8d3a2;}
[data-testid="stMetricLabel"] {letter-spacing:.18em;text-transform:uppercase;font-size:.7rem;color:#c9a96a;}
section[data-testid="stSidebar"] {border-right:1px solid rgba(201,169,106,.25);}
</style>
"""

def style_chart(fig, title=""):
    fig.update_layout(
        title=title, paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font={"color": MUTED, "family": "Jost"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h"},
        xaxis={"gridcolor": "rgba(201,169,106,.12)"},
        yaxis={"gridcolor": "rgba(201,169,106,.12)"},
    )
    return fig

def slot_label(value: float) -> str:
    return f"{int(value):02d}:{'30' if value % 1 else '00'}"

def main():
    st.set_page_config(page_title="AURUM - Delhi Parking Forecast", page_icon="🅿️", layout="wide")
    st.markdown(CSS.replace("__FONT_URL__", FONT_URL), unsafe_allow_html=True)

    try:
        forecaster = get_forecaster()
    except Exception as exc:
        st.error("The forecasting model could not be loaded.")
        st.exception(exc)
        st.stop()

    today = datetime.now(IST).date()
    st.title("AURUM")
    st.caption("DELHI - PARKING INTELLIGENCE")

    with st.sidebar:
        st.header("Forecast settings")
        selected_place = st.selectbox(
            "Destination (type to search)", PLACES,
            format_func=lambda p: f"{p.name} - {p.area}",
        )
        selected_day = st.date_input(
            "Date", today, min_value=today, max_value=today + timedelta(days=365)
        )
        time_slots = [
            WINDOW[0] + i * SLOT_STEP
            for i in range(int((WINDOW[1] - WINDOW[0]) / SLOT_STEP) + 1)
        ]
        selected_time = st.select_slider(
            "Time", options=time_slots, value=13.0, format_func=slot_label
        )
        st.markdown("**Conditions**")
        conditions = {
            "rain": st.checkbox("Rain"),
            "holiday": st.checkbox("Public holiday"),
            "peak_event": st.checkbox("Peak-event day"),
        }

    selected_datetime = datetime.combine(selected_day, datetime.min.time()).replace(
        hour=int(selected_time), minute=int((selected_time % 1) * 60)
    )
    prediction = forecaster.predict(selected_place.id, selected_datetime, **conditions)
    curve = forecaster.curve(selected_place.id, selected_day, **conditions)
    labels = [slot_label(WINDOW[0] + i * SLOT_STEP) for i in range(len(curve))]

    tab_forecast, tab_week, tab_locations, tab_model = st.tabs(
        ["Forecast", "Weekly pattern", "Delhi locations", "Model details"]
    )

    with tab_forecast:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected vehicles", f"{prediction.vehicles:,}")
        c2.metric("Occupancy", f"{prediction.ratio:.0%}")
        c3.metric("80% estimated range", f"{prediction.low:,} - {prediction.high:,}")
        quietest_index = min(
            range(len(curve)),
            key=lambda i: curve[i].ratio if labels[i] >= "09:00" else float("inf"),
        )
        c4.metric("Quietest after 09:00", labels[quietest_index])

        st.markdown(
            f"**{prediction.status}** · Indicative capacity: {prediction.capacity:,} bays"
        )
        fig = go.Figure()
        fig.add_scatter(x=labels, y=[p.high for p in curve], line={"width": 0},
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=labels, y=[p.low for p in curve], fill="tonexty",
                        fillcolor="rgba(201,169,106,.15)", line={"width": 0},
                        name="Estimated 80% range", hoverinfo="skip")
        fig.add_scatter(x=labels, y=[p.vehicles for p in curve],
                        line={"color": GOLD_LIGHT, "width": 2.5}, name="Forecast")
        fig.add_vline(x=slot_label(selected_time), line_color=GOLD, line_dash="dot")
        fig.add_hline(y=selected_place.capacity, line_color=MUTED, line_dash="dash",
                      annotation_text="Capacity")
        st.plotly_chart(style_chart(
            fig, f"{selected_place.name} · {selected_day:%A %d %b %Y}"
        ), use_container_width=True)

    with tab_week:
        matrix = forecaster.week_matrix(selected_place.id, **conditions) * 100
        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=labels,
            y=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            zmin=0, zmax=100,
            colorscale=[[0, "#13110e"], [0.5, "#7a6238"], [1, GOLD_LIGHT]],
            colorbar={"title": "Occupancy %"},
        ))
        st.plotly_chart(style_chart(fig, "Weekly parking occupancy pattern"),
                        use_container_width=True)

    with tab_locations:
        location_rows = []
        for place in PLACES:
            ratio = forecaster.predict(place.id, selected_datetime, **conditions).ratio
            location_rows.append({
                "Location": place.name, "Area": place.area, "Type": place.kind,
                "Capacity": place.capacity, "Forecast occupancy": f"{ratio:.0%}",
                "Expected vehicles": round(ratio * place.capacity),
            })
        locations_df = pd.DataFrame(location_rows).sort_values(
            "Forecast occupancy", ascending=False
        )
        st.dataframe(locations_df, use_container_width=True, hide_index=True)
        map_df = pd.DataFrame([
            {"lat": p.lat, "lon": p.lon,
             "size": 10 + 30 * forecaster.predict(p.id, selected_datetime, **conditions).ratio}
            for p in PLACES
        ])
        st.map(map_df, latitude="lat", longitude="lon", size="size")

    with tab_model:
        metrics = forecaster.metrics
        a, b, c = st.columns(3)
        a.metric("Holdout R²", f"{metrics['holdout_r2']:.2f}")
        b.metric("Holdout MAE", f"{metrics['holdout_mae'] * 100:.1f} percentage points")
        c.metric("Training rows", f"{metrics['train_rows']:,}")
        st.write("Parking-lot behaviour multipliers:", forecaster.mods)
        st.caption(
            "The model is trained on occupancy records from 30 car parks in Birmingham, UK. "
            "Delhi locations borrow patterns from similar car parks and use indicative capacities. "
            "Forecast coverage is 07:00-17:30. These are estimates, not live parking availability."
        )
        if st.button("Retrain forecasting model"):
            with st.spinner("Training model..."):
                try:
                    bundle = train_model()
                    get_forecaster.clear()
                    st.success(f"Model retrained. Holdout MAE: {bundle['metrics']['holdout_mae']:.4f}")
                    st.rerun()
                except Exception as exc:
                    st.error("Model retraining failed.")
                    st.exception(exc)

    st.caption(
        "AURUM parking forecast demonstration · Synthetic/borrowed location patterns; "
        "connect validated Delhi sensor data before operational use."
    )

if __name__ == "__main__":
    main()
