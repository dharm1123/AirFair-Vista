\import streamlit as st

st.set_page_config(
    page_title='AirFair Vista',
    page_icon='✈️',
    layout='wide',
    initial_sidebar_state='expanded'
)

import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "backend"))

from preprocessor import (
    AIRLINES, SOURCES, DESTINATIONS, STOPS, MONTHS, WEEKDAYS,
    AIRLINE_MEAN_PRICE, SOURCE_FREQ, DEST_FREQ, SOURCE_MEAN_PRICE,
    PRICE_MIN, PRICE_MAX, PRICE_AVG, PRICE_MED,
    AIRLINE_MEAN, STOPS_MEAN, ROUTE_MEAN, HOUR_MEAN, MONTH_MEAN, WEEKDAY_MEAN,
    DURATION_LOOKUP, AVG_SPEED_BY_STOPS, CITY_COORDS,
    VALID_ROUTES, VALID_DESTINATIONS, VALID_AIRLINE_STOPS,
    MAX_PAX_BY_STOPS, INDIAN_HOLIDAYS,
    MODEL_PATH, MODEL_FEATURES,
    haversine_km, is_indian_holiday, predict_duration,
    get_validation_errors, build_features, build_feature_matrix,
)

@st.cache_resource(show_spinner='Loading model...')
def load_model():
    if os.path.exists(MODEL_PATH):
        pkl = joblib.load(MODEL_PATH)
        return pkl['model'], True
    return None, False

model, MODEL_LOADED = load_model()


def predict_price(airline, source, destination, stops,
                  dep_hour, journey_month, journey_weekday,
                  journey_day, duration_hours, passengers):
    if MODEL_LOADED:
        try:
            X = build_features(airline, source, destination,
                               dep_hour, journey_month, journey_weekday,
                               journey_day, duration_hours)
            raw = model.predict(X)[0]
            if raw < 15:
                raw = np.expm1(raw)
            return round(float(raw) * passengers, 2)
        except Exception as e:
            st.warning(f'Model error: {e} → fallback estimator.')
    base = (
        AIRLINE_MEAN.get(airline, PRICE_AVG) * 0.30 +
        STOPS_MEAN.get(stops, PRICE_AVG) * 0.25 +
        ROUTE_MEAN.get((source, destination), PRICE_AVG) * 0.20 +
        HOUR_MEAN.get(dep_hour, PRICE_AVG) * 0.10 +
        MONTH_MEAN.get(journey_month, PRICE_AVG) * 0.10 +
        WEEKDAY_MEAN.get(journey_weekday, PRICE_AVG) * 0.05
    )
    return round(base * (1 + max(0, duration_hours - 2) * 0.04) * passengers, 2)


def batch_predict_app(combos: list, passengers: int = 1) -> list:
    from preprocessor import batch_predict
    return batch_predict(model if MODEL_LOADED else None, combos, passengers)


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER CSS  — theme-aware, works in both Light and Dark mode
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  STYLES  — theme-aware CSS + JS runtime patcher
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800;900&display=swap');

:root {
    --primary: #0052cc;
    --primary-dark: #003a99;
    --primary-light: #e8f0fe;
    --accent: #f5a623;
    --success: #22c55e;
    --danger: #ef4444;
    --bg-main: #f0f4ff;
    --bg-card: #ffffff;
    --bg-sidebar: #09122c;
    --text-primary: #0f172a;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --border: #e2e8f0;
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.10);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --radius-xl: 24px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp { background: var(--bg-main); }

/* ── SELECTBOX: border/radius only — NO background forcing ─────── */
/* Background is handled entirely by JS to avoid light/dark conflicts */
div[data-baseweb="select"] > div:first-child {
    border-radius: 8px !important;
    min-height: 42px !important;
    transition: border-color 0.15s ease !important;
}
div[data-baseweb="select"] > div:first-child:hover {
    box-shadow: 0 0 0 3px rgba(0,82,204,0.12) !important;
}
div[data-baseweb="menu"] {
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,82,204,0.14) !important;
}
li[role="option"] { font-size: 0.88rem !important; }

/* ── LABELS — CSS fallback for all widget label types ─────── */
.stSelectbox label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-bottom: 4px !important;
}
.stSlider label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stDateInput"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
/* Radio & checkbox labels */
.stRadio label, .stRadio > label,
.stCheckbox label,
[data-testid="stRadioGroup"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
/* Streamlit modern widget label wrapper */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] > div,
[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* ── SIDEBAR (always dark — standalone block) ─────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"] section,
[data-testid="stSidebarContent"] {
    background: var(--bg-sidebar) !important;
    background-color: var(--bg-sidebar) !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] span { color: #cbd5e1 !important; -webkit-text-fill-color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div:first-child {
    background-color: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] [class*="singleValue"],
[data-testid="stSidebar"] div[data-baseweb="select"] [class*="placeholder"],
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill: #94a3b8 !important; }
[data-testid="stSidebar"] div[data-baseweb="menu"] {
    background: #1e2d50 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] li[role="option"] {
    color: #e2e8f0 !important;
    background: #1e2d50 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}
[data-testid="stSidebar"] li[role="option"]:hover {
    background: rgba(0,82,204,0.35) !important;
    color: #93c5fd !important;
    -webkit-text-fill-color: #93c5fd !important;
}
/* Fix toggles in sidebar */
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] [data-testid="stToggleLabel"] {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
}
/* Fix divider color in sidebar */
[data-testid="stSidebar"] hr {
    border-top: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── SLIDER ─────────────────────────────────────────────────────── */
.stSlider > div > div > div > div { background: var(--primary) !important; }

/* ── PAGE HEADER ─────────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #09122c 0%, #0d1f5c 40%, #0052cc 100%);
    border-radius: var(--radius-xl); padding: 36px 40px 32px;
    margin-bottom: 28px; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,82,204,0.25);
}
.page-header h1 { font-family: 'Syne', sans-serif; font-size: 2.1rem; font-weight: 900; margin: 0 0 6px; color: #fff; letter-spacing: -0.5px; }
.page-header h1 em { color: var(--accent); font-style: normal; }
.page-header p { color: rgba(255,255,255,0.55); font-size: 0.88rem; margin: 0; }
.model-pill { display: inline-flex; align-items: center; gap: 5px; margin-top: 14px; background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4); color: #4ade80; border-radius: 20px; padding: 4px 14px; font-size: 0.72rem; font-weight: 700; }
.model-pill-warn { display: inline-flex; align-items: center; gap: 5px; margin-top: 14px; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #fbbf24; border-radius: 20px; padding: 4px 14px; font-size: 0.72rem; font-weight: 700; }
.form-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 28px 32px 20px; box-shadow: var(--shadow-md); margin-bottom: 24px; }
.form-section-title { font-size: 0.62rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: var(--primary); margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid var(--primary-light); display: flex; align-items: center; gap: 6px; }
button[kind="primary"] { background: linear-gradient(90deg, var(--primary), var(--primary-dark)) !important; color: white !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important; border-radius: var(--radius-md) !important; border: none !important; padding: 15px 28px !important; }
button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(0,82,204,0.35) !important; }
.result-ticket { background: var(--bg-card); border-radius: var(--radius-xl); border: 1px solid var(--border); overflow: hidden; box-shadow: var(--shadow-lg); margin-bottom: 24px; }
.ticket-header { background: linear-gradient(135deg, #09122c 0%, #0d1f5c 50%, #0052cc 100%); padding: 20px 28px; display: flex; justify-content: space-between; align-items: center; position: relative; overflow: hidden; }
.ticket-header::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent), #ff6b35, var(--accent)); }
.ticket-airline { font-family: 'Syne', sans-serif; color: white; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
.ticket-tag { background: rgba(245,166,35,0.18); border: 1px solid rgba(245,166,35,0.45); color: var(--accent); border-radius: 20px; padding: 4px 12px; font-size: 0.7rem; font-weight: 700; margin-left: 6px; }
.ticket-body { padding: 28px 32px; display: flex; align-items: center; background: var(--bg-card); }
.ticket-city { text-align: center; min-width: 100px; flex-shrink: 0; }
.ticket-city .code { font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 900; color: var(--text-primary); line-height: 1; }
.ticket-city .name { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 5px; font-weight: 600; }
.ticket-city .time { font-size: 0.88rem; color: var(--primary); font-weight: 700; margin-top: 8px; }
.ticket-mid { flex: 1; text-align: center; padding: 0 24px; position: relative; }
.ticket-mid .stops-label { font-size: 0.65rem; color: var(--accent); font-weight: 800; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px; display: block; }
.ticket-mid .dash-line { border-top: 2px dashed #cbd5e1; position: relative; height: 24px; display: flex; align-items: center; justify-content: center; }
.ticket-mid .plane { position: relative; top: auto; left: auto; transform: none; background: white; padding: 0 10px; font-size: 1.15rem; z-index: 1; display: inline-block; margin-top: -12px; }
.ticket-mid .dur { font-size: 0.73rem; color: var(--text-muted); margin-top: 12px; display: block; }
.ticket-footer { border-top: 2px dashed #e2e8f0; padding: 22px 32px; display: flex; justify-content: space-between; align-items: center; background: var(--bg-card); }
.price-label { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; font-weight: 700; }
.price-amount { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 900; color: var(--primary); line-height: 1; }
.vs-avg-up { color: var(--danger); font-weight: 700; font-size: 0.8rem; margin-top: 5px; display: block; }
.vs-avg-down { color: var(--success); font-weight: 700; font-size: 0.8rem; margin-top: 5px; display: block; }
.route-info-block { text-align: center; }
.route-info-label { font-size: 0.58rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 700; }
.route-info-items { display: flex; gap: 24px; justify-content: center; }
.route-info-item { text-align: center; }
.route-info-item .icon { font-size: 1.1rem; display: block; margin-bottom: 3px; }
.route-info-item .val { font-family: 'Syne', sans-serif; font-size: 0.88rem; font-weight: 800; color: var(--text-primary); display: block; }
.route-info-item .sub { font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.per-pax-block { text-align: right; }
.per-pax-label { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 4px; }
.per-pax-val { font-family: 'Syne', sans-serif; font-size: 1.45rem; font-weight: 800; color: var(--text-primary); }
.dataset-avg { font-size: 0.68rem; color: var(--text-muted); margin-top: 3px; }
.output-section-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 800; color: var(--text-primary); margin: 28px 0 14px; display: flex; align-items: center; gap: 8px; }
.output-section-title::after { content: ''; flex: 1; height: 1.5px; background: linear-gradient(90deg, var(--primary-light), transparent); margin-left: 8px; }
[data-testid="stMetric"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; padding: 16px 20px !important; }
[data-testid="stMetricLabel"] > div { font-size: 0.7rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; color: var(--text-muted) !important; }
[data-testid="stMetricValue"] > div { font-family: 'Syne', sans-serif !important; font-size: 1.5rem !important; font-weight: 800 !important; color: var(--text-primary) !important; }
.valid-badge { display: inline-flex; align-items: center; gap: 4px; background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); color: #16a34a; border-radius: 20px; padding: 2px 10px; font-size: 0.68rem; font-weight: 700; margin-top: 4px; }
.duration-chip { display: inline-flex; align-items: center; gap: 8px; background: var(--primary-light); border: 1.5px solid rgba(0,82,204,0.2); border-radius: var(--radius-sm); padding: 10px 16px; font-size: 0.84rem; font-weight: 600; color: var(--primary); margin: 8px 0 4px; width: 100%; }
.duration-chip .dur-val { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 900; color: var(--primary-dark); }
.sb-brand { font-family: 'Syne', sans-serif; font-size: 1.45rem; font-weight: 900; color: #fff; }
.sb-brand span { color: var(--accent); }
.sb-tagline { font-size: 0.65rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }
.sb-section-label { font-size: 0.58rem !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--accent) !important; margin: 20px 0 10px !important; display: flex; align-items: center; gap: 6px; }
.sb-section-label::after { content: ''; flex: 1; height: 1px; background: rgba(245,166,35,0.25); }
.sb-info-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.76rem; }
.sb-info-row .k { color: #64748b; }
.sb-info-row .v { color: #e2e8f0; font-weight: 600; }
.sb-model-ok { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #4ade80; border-radius: var(--radius-sm); padding: 9px 12px; font-size: 0.76rem; font-weight: 600; margin-top: 10px; line-height: 1.5; }
.sb-model-err { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); color: #f87171; border-radius: var(--radius-sm); padding: 9px 12px; font-size: 0.76rem; font-weight: 600; margin-top: 10px; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* ── PLOTLY CHARTS — theme-aware card background ──────────── */
[data-testid="stPlotlyChart"] > div {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
    padding: 6px !important;
    border: 1px solid var(--border) !important;
}
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  JAVASCRIPT — Runtime theme detector & color patcher
#  This is the ONLY place that sets selectbox colors.
#  Detects real theme by reading actual background color from DOM.
#  Runs every 300ms so it catches theme toggles instantly.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<script>
(function() {
    'use strict';

    // Detect current theme by reading actual background luminance
    function isDarkMode() {
        const targets = [
            '[data-testid="stAppViewContainer"]',
            '.stApp',
            'body'
        ];
        for (const sel of targets) {
            const el = document.querySelector(sel);
            if (!el) continue;
            const bg = window.getComputedStyle(el).backgroundColor;
            const nums = bg.match(/\d+/g);
            if (!nums || nums.length < 3) continue;
            const lum = (parseInt(nums[0]) * 299 +
                         parseInt(nums[1]) * 587 +
                         parseInt(nums[2]) * 114) / 1000;
            if (lum < 180) return true;  // dark
            if (lum >= 180) return false; // light
        }
        return false; // default light
    }

    var lastTheme = null;

    function applyColors() {
        var dark = isDarkMode();
        var themeKey = dark ? 'dark' : 'light';

        // Only re-apply if theme changed (or elements are new)
        var forceUpdate = (themeKey !== lastTheme);
        lastTheme = themeKey;

        // ── Color palette ──────────────────────────────────────────
        var C = dark ? {
            selectBg:    '#1e2130',
            selectBorder:'1.5px solid rgba(255,255,255,0.20)',
            text:        'rgba(250,250,250,0.87)',
            textFill:    'rgba(250,250,250,0.87)',
            menuBg:      '#1e2130',
            menuBorder:  '1px solid rgba(255,255,255,0.15)',
            menuItem:    'rgba(250,250,250,0.87)',
            menuItemHov: '#2d3a5e',
            menuTextHov: '#93c5fd',
            arrow:       '#94a3b8',
            labelColor:  '#94a3b8',
            inputBg:     '#1e2130',
            inputBorder: '1.5px solid rgba(255,255,255,0.20)'
        } : {
            selectBg:    '#ffffff',
            selectBorder:'1.5px solid #c8d6f0',
            text:        '#111827',
            textFill:    '#111827',
            menuBg:      '#ffffff',
            menuBorder:  '1.5px solid #c8d6f0',
            menuItem:    '#111827',
            menuItemHov: '#e8f0fe',
            menuTextHov: '#0052cc',
            arrow:       '#64748b',
            labelColor:  '#374151',
            inputBg:     '#ffffff',
            inputBorder: '1.5px solid #c8d6f0'
        };

        function s(el, prop, val) {
            if (el.style.getPropertyValue(prop) !== val || forceUpdate) {
                el.style.setProperty(prop, val, 'important');
            }
        }

        // ── Force sidebar always dark (regardless of light/dark theme) ──
        var sidebarEls = document.querySelectorAll(
            '[data-testid="stSidebar"], [data-testid="stSidebar"] > div, ' +
            '[data-testid="stSidebar"] > div > div, [data-testid="stSidebarContent"]'
        );
        sidebarEls.forEach(function(el) {
            s(el, 'background-color', '#09122c');
            s(el, 'background', '#09122c');
        });

        // ── Apply to all selectboxes (skip sidebar) ────────────────
        document.querySelectorAll(
            'div[data-baseweb="select"]'
        ).forEach(function(sel) {
            if (sel.closest('[data-testid="stSidebar"]')) return;

            // Container box
            var box = sel.querySelector(':scope > div:first-child');
            if (box) {
                s(box, 'background-color', C.selectBg);
                s(box, 'border', C.selectBorder);
            }

            // Selected value text
            sel.querySelectorAll(
                '[class*="singleValue"],[class*="placeholder"]'
            ).forEach(function(el) {
                s(el, 'color', C.text);
                s(el, '-webkit-text-fill-color', C.textFill);
            });

            // Input inside select
            sel.querySelectorAll('input').forEach(function(el) {
                s(el, 'color', C.text);
                s(el, '-webkit-text-fill-color', C.textFill);
                s(el, 'background-color', C.selectBg);
            });

            // Arrow SVG
            sel.querySelectorAll('svg').forEach(function(el) {
                s(el, 'fill', C.arrow);
            });
        });

        // ── Dropdown open menus ────────────────────────────────────
        document.querySelectorAll(
            'div[data-baseweb="menu"]'
        ).forEach(function(menu) {
            if (menu.closest('[data-testid="stSidebar"]')) return;
            s(menu, 'background-color', C.menuBg);
            s(menu, 'border', C.menuBorder);

            menu.querySelectorAll(
                'li[role="option"]'
            ).forEach(function(li) {
                s(li, 'color', C.menuItem);
                s(li, '-webkit-text-fill-color', C.menuItem);
                s(li, 'background-color', C.menuBg);
                li.addEventListener('mouseenter', function() {
                    this.style.setProperty('background-color', C.menuItemHov, 'important');
                    this.style.setProperty('color', C.menuTextHov, 'important');
                    this.style.setProperty('-webkit-text-fill-color', C.menuTextHov, 'important');
                });
                li.addEventListener('mouseleave', function() {
                    this.style.setProperty('background-color', C.menuBg, 'important');
                    this.style.setProperty('color', C.menuItem, 'important');
                    this.style.setProperty('-webkit-text-fill-color', C.menuItem, 'important');
                });
            });
        });

        // ── Labels (all widget types) ──────────────────────────────
        document.querySelectorAll(
            '.stSelectbox label, .stSlider label, ' +
            '[data-testid="stDateInput"] label, .stNumberInput label, ' +
            '.stRadio label, .stRadio > label, ' +
            '[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] > div, ' +
            '[data-testid="stWidgetLabel"] p, ' +
            '.stRadio [data-testid="stMarkdownContainer"] p, ' +
            '[data-testid="stExpander"] label, ' +
            '.stCheckbox label, .stMultiSelect label'
        ).forEach(function(el) {
            if (el.closest('[data-testid="stSidebar"]')) return;
            s(el, 'color', C.labelColor);
            s(el, '-webkit-text-fill-color', C.labelColor);
        });

        // ── Slider tick/range labels ───────────────────────────────
        document.querySelectorAll(
            '[data-testid="stSlider"] [class*="tickBarMin"], ' +
            '[data-testid="stSlider"] [class*="tickBarMax"], ' +
            '[data-testid="stSlider"] [data-testid="stTickBarMin"], ' +
            '[data-testid="stSlider"] [data-testid="stTickBarMax"], ' +
            '[data-testid="stSlider"] span'
        ).forEach(function(el) {
            if (el.closest('[data-testid="stSidebar"]')) return;
            s(el, 'color', C.labelColor);
            s(el, '-webkit-text-fill-color', C.labelColor);
        });

        // ── Text inputs (date, number) ─────────────────────────────
        document.querySelectorAll(
            '[data-testid="stDateInput"] input, .stNumberInput input'
        ).forEach(function(el) {
            if (el.closest('[data-testid="stSidebar"]')) return;
            s(el, 'background-color', C.inputBg);
            s(el, 'border', C.inputBorder);
            s(el, 'color', C.text);
            s(el, '-webkit-text-fill-color', C.textFill);
            s(el, 'border-radius', '8px');
        });
    }

    // Run immediately
    applyColors();

    // Run on every DOM change (Streamlit widget re-renders)
    var observer = new MutationObserver(function() {
        applyColors();
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Poll every 300ms — catches theme toggle + Streamlit reruns
    setInterval(applyColors, 300);

    // Also catch Streamlit postMessage events
    window.addEventListener('message', function(e) {
        setTimeout(applyColors, 50);
    });
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ① INTERACTIVE SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style="padding:8px 0 20px">
        <div class="sb-brand">✈ AirFair Vista</div>
        <div class="sb-tagline">Flight Price Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sb-section-label">Display Settings</div>',
                unsafe_allow_html=True)
    currency       = st.selectbox('Currency',
                                  ['INR (₹)', 'USD ($)', 'EUR (€)', 'GBP (£)'])
    show_range     = st.toggle('Show Price Range',        value=True)
    show_compare   = st.toggle('Show Airline Comparison', value=True)

    st.divider()

    st.markdown('<div class="sb-section-label">Model Status</div>',
                unsafe_allow_html=True)
    if MODEL_LOADED:
        st.markdown(
            '<div class="sb-model-ok">✅ Pipeline loaded<br>'
            '<span style="font-weight:400;font-size:0.72rem;">'
            'flight_price_prediction_pipeline.pkl</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="sb-model-err">⚠️ Model not found<br>'
            '<span style="font-weight:400;font-size:0.72rem;">'
            'Using fallback estimator</span></div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown('<div class="sb-section-label">Dataset Info</div>',
                unsafe_allow_html=True)
    for k, v in [
        ('Records',   '10,682 flights'),
        ('Period',    'Mar – Jun 2019'),
        ('Routes',    '6 domestic'),
        ('Airlines',  '12 carriers'),
        ('Min price', '₹1,759'),
        ('Max price', '₹79,512'),
        ('Avg price', '₹9,087'),
    ]:
        st.markdown(
            f'<div class="sb-info-row">'
            f'<span class="k">{k}</span>'
            f'<span class="v">{v}</span></div>',
            unsafe_allow_html=True
        )


# Currency helpers
SYM = {'INR (₹)':'₹', 'USD ($)':'$', 'EUR (€)':'€', 'GBP (£)':'£'}
FAC = {'INR (₹)':1.0, 'USD ($)':0.012, 'EUR (€)':0.011, 'GBP (£)':0.0095}
sym = SYM[currency]
fac = FAC[currency]


# ═══════════════════════════════════════════════════════════════
#  PRICE PREDICTOR — ② Input Form  +  ③ Output Display
# ═══════════════════════════════════════════════════════════════
model_pill = (
    '<span class="model-pill">🤖 Real ML Pipeline Active</span>'
    if MODEL_LOADED
    else '<span class="model-pill-warn">⚠️ Fallback Estimator Active</span>'
)
st.markdown(f"""
<div class="page-header">
    <h1>Flight Price <em>Predictor</em></h1>
    <p>Indian domestic routes · 10,682 real flights · Mar–Jun 2019</p>
    {model_pill}
</div>
""", unsafe_allow_html=True)

import datetime as _dt

if 'source' not in st.session_state:
    st.session_state.source      = SOURCES[0]
if 'destination' not in st.session_state:
    st.session_state.destination = VALID_DESTINATIONS[SOURCES[0]][0]
if 'airline' not in st.session_state:
    st.session_state.airline     = AIRLINES[0]
if 'stops' not in st.session_state:
    st.session_state.stops       = VALID_AIRLINE_STOPS[AIRLINES[0]][0]
if 'submitted' not in st.session_state:
    st.session_state.submitted   = False

def on_source_change():
    new_src  = st.session_state['_source_sel']
    st.session_state.source      = new_src
    valid    = VALID_DESTINATIONS.get(new_src, DESTINATIONS)
    st.session_state.destination = valid[0]
    st.session_state['_dst_sel']  = valid[0]
    st.session_state.submitted    = False

def on_destination_change():
    st.session_state.destination = st.session_state['_dst_sel']
    st.session_state.submitted   = False

def on_airline_change():
    new_al   = st.session_state['_airline_sel']
    st.session_state.airline     = new_al
    valid    = VALID_AIRLINE_STOPS.get(new_al, STOPS)
    st.session_state.stops       = valid[0]
    st.session_state['_stops_sel'] = valid[0]
    st.session_state.submitted    = False

def on_stops_change():
    st.session_state.stops     = st.session_state['_stops_sel']
    st.session_state.submitted = False

st.markdown('<div class="form-card">', unsafe_allow_html=True)
with st.container():

    st.markdown('<div class="form-section-title">✈ Section A · Route</div>',
                unsafe_allow_html=True)
    col_src, col_arrow, col_dst = st.columns([5, 1, 5])

    source = col_src.selectbox(
        'Source City', SOURCES,
        index=SOURCES.index(st.session_state.source),
        key='_source_sel',
        on_change=on_source_change
    )
    col_arrow.markdown(
        '<div style="text-align:center;padding-top:28px;'
        'font-size:1.3rem;color:#0f4c9a">→</div>',
        unsafe_allow_html=True
    )
    valid_dsts  = VALID_DESTINATIONS.get(st.session_state.source, DESTINATIONS)
    if st.session_state.destination not in valid_dsts:
        st.session_state.destination = valid_dsts[0]
    destination = col_dst.selectbox(
        'Destination City', valid_dsts,
        index=valid_dsts.index(st.session_state.destination),
        key='_dst_sel',
        on_change=on_destination_change
    )
    auto_dst = len(valid_dsts) == 1
    col_dst.markdown(
        f'<span class="valid-badge">'
        f'{"✈ Auto-set" if auto_dst else "✅"} '
        f'{len(valid_dsts)} route(s) from {source}</span>',
        unsafe_allow_html=True
    )

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('<div class="form-section-title">🛫 Section B · Flight Details</div>',
                unsafe_allow_html=True)
    col_al, col_st = st.columns(2)

    airline = col_al.selectbox(
        'Airline', AIRLINES,
        index=AIRLINES.index(st.session_state.airline),
        key='_airline_sel',
        on_change=on_airline_change
    )
    valid_stops = VALID_AIRLINE_STOPS.get(st.session_state.airline, STOPS)
    if st.session_state.stops not in valid_stops:
        st.session_state.stops = valid_stops[0]
    stops = col_st.selectbox(
        'Stops', valid_stops,
        index=valid_stops.index(st.session_state.stops),
        key='_stops_sel',
        on_change=on_stops_change
    )
    col_st.markdown(
        f'<span class="valid-badge">'
        f'{"✈ Auto-set" if len(valid_stops)==1 else "✅"} '
        f'{len(valid_stops)} stop option(s) for {airline}</span>',
        unsafe_allow_html=True
    )

    duration_hrs = predict_duration(destination, destination, stops) \
        if source == destination \
        else predict_duration(source, destination, stops)
    st.markdown(
        f'<div class="duration-chip">'
        f'⏱ Auto-predicted Duration: '
        f'<span class="dur-val">{duration_hrs:.1f}h</span>'
        f' &nbsp;·&nbsp; Based on real {source} → {destination} / {stops} records'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<br>', unsafe_allow_html=True)

st.markdown('<div class="form-section-title">📅 Section C · Journey Details</div>',
            unsafe_allow_html=True)

from datetime import date as _date, timedelta as _timedelta

today = _date.today()

cal1, cal2, cal3 = st.columns([3, 2, 1])

# ✅ Future dates only
travel_date = cal1.date_input(
    '📆 Travel Date',
    value=today + _timedelta(days=1),
    min_value=today + _timedelta(days=1),
    max_value=_date(2030, 12, 31),
    help='Only future dates are selectable.',
    key='_travel_date'
)

journey_month   = travel_date.month
journey_day     = travel_date.day
journey_weekday = travel_date.weekday()

cal1.caption(
    f'📅 {travel_date.strftime("%A, %d %B %Y")} · '
    f'Month={journey_month}, Day={journey_day}, Weekday={journey_weekday}'
)

st.info("ℹ️ Model trained on 2019 data. Future date predictions are approximations based on learned patterns.")

dep_hour   = cal2.slider('🕐 Departure Hour', 0, 23, 8, key='_dep_hour')
passengers = cal3.number_input('👥 Passengers', 1, 9, 1, key='_passengers')

st.markdown('<br>', unsafe_allow_html=True)

live_errors, live_warnings = get_validation_errors(
    source, destination, airline, stops, int(passengers), dep_hour
)

if live_errors:
    for e in live_errors:
        st.error(f'❌ {e}')
elif live_warnings:
    for w in live_warnings:
        st.warning(f'⚠️ {w}')
else:
    st.success('✅ All inputs valid — ready to predict!')

    if not live_errors:
        _live_dur = predict_duration(source, destination, stops)
        _live_price = predict_price(
            airline, source, destination, stops,
            dep_hour, journey_month, journey_weekday,
            int(journey_day), _live_dur, int(passengers)
        ) * fac
        _live_avg = PRICE_AVG * fac
        _live_diff = _live_price - _live_avg
        _live_icon = '▲' if _live_diff > 0 else '▼'
        _live_col  = '#ef4444' if _live_diff > 0 else '#22c55e'
        st.markdown(
            f'<div style="background:linear-gradient(90deg,#eff6ff,#dbeafe);'
            f'border:1.5px solid #bfdbfe;border-radius:10px;'
            f'padding:12px 20px;display:flex;align-items:center;gap:16px;margin-bottom:4px">'
            f'<div style="font-size:0.65rem;font-weight:800;color:#3b82f6;'
            f'text-transform:uppercase;letter-spacing:1.5px">⚡ Live Estimate</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:1.6rem;'
            f'font-weight:900;color:#0052cc;letter-spacing:-1px">{sym}{_live_price:,.0f}</div>'
            f'<div style="font-size:0.78rem;font-weight:600;color:{_live_col}">'
            f'{_live_icon} {sym}{abs(_live_diff):,.0f} vs dataset avg</div>'
            f'<div style="font-size:0.72rem;color:#64748b;margin-left:auto">'
            f'Updates live · {airline} · {source}→{destination} · {stops}'
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<br>', unsafe_allow_html=True)

    btn_label = '🔍  Predict Price' if not live_errors else '⚠️  Fix Errors Above to Predict'
    if st.button(
        btn_label,
        use_container_width=True,
        type='primary',
        disabled=bool(live_errors),
        key='_predict_btn'
    ):
        st.session_state.submitted   = True
        st.session_state._src_snap   = source
        st.session_state._dst_snap   = destination
        st.session_state._al_snap    = airline
        st.session_state._st_snap    = stops
        st.session_state._dur_snap   = duration_hrs
        st.session_state._hr_snap    = dep_hour
        st.session_state._mo_snap    = journey_month
        st.session_state._wd_snap    = journey_weekday
        st.session_state._dy_snap    = journey_day
        st.session_state._px_snap    = passengers
        st.session_state._td_snap    = travel_date
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

submitted = st.session_state.submitted
if submitted:
    source          = st.session_state._src_snap
    destination     = st.session_state._dst_snap
    airline         = st.session_state._al_snap
    stops           = st.session_state._st_snap
    duration_hrs    = st.session_state._dur_snap
    dep_hour        = st.session_state._hr_snap
    journey_month   = st.session_state._mo_snap
    journey_weekday = st.session_state._wd_snap
    journey_day     = st.session_state._dy_snap
    passengers      = st.session_state._px_snap
    travel_date     = st.session_state._td_snap


if submitted:
    errors, warnings = get_validation_errors(
        source, destination, airline, stops,
        int(passengers), dep_hour
    )

    for w in warnings:
        st.warning(f'⚠️ {w}')

    if errors:
        for e in errors:
            st.error(f'❌ {e}')
        st.info(
            '💡 **Valid routes:** '
            'Banglore→Delhi/New Delhi · Chennai→Kolkata · '
            'Delhi→Cochin · Kolkata→Banglore · Mumbai→Hyderabad'
        )
        st.stop()

    with st.spinner('🤖  Running ML model...'):
        price_inr = predict_price(
            airline, source, destination, stops,
            dep_hour, journey_month, journey_weekday,
            int(journey_day), duration_hrs, int(passengers)
        )
    price_d = price_inr * fac
    low_d   = price_d * 0.90
    high_d  = price_d * 1.15
    avg_d   = PRICE_AVG * fac
    diff    = price_d - avg_d

    src_code = source[:3].upper()
    dst_code = destination[:3].upper()
    dep_str  = f"{dep_hour:02d}:00"
    day_str  = f"{WEEKDAYS[journey_weekday][:3]}, {int(journey_day)} {MONTHS[journey_month]}"
    vs_class = 'vs-avg-up' if diff > 0 else 'vs-avg-down'
    vs_text  = (f'▲ {sym}{abs(diff):,.0f} above avg'
                if diff > 0 else f'▼ {sym}{abs(diff):,.0f} below avg')
    dist_km      = haversine_km(source, destination)
    dist_str     = f'{dist_km:,} km' if dist_km > 0 else 'N/A'
    speed_str    = f'{round(dist_km / duration_hrs):,} km/h' if dist_km > 0 and duration_hrs > 0 else 'N/A'
    price_per_km = f'{sym}{round(price_d / dist_km, 1)}/km' if dist_km > 0 else 'N/A'
    holiday_name = INDIAN_HOLIDAYS.get((journey_month, int(journey_day)), None)
    holiday_tag  = f'🎉 {holiday_name}' if holiday_name else ''

    rc1, rc2 = st.columns([6, 1])
    rc1.success('✅  Prediction ready!')
    if rc2.button('🔄 New', key='_reset_btn', help='Reset and make a new prediction'):
        st.session_state.submitted = False
        st.rerun()

    st.markdown(
        '<div class="output-section-title">📋 Output A · Your Flight Estimate</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="result-ticket">
        <div class="ticket-header">
            <div>
                <div class="ticket-airline">{airline.upper()}</div>
                <div style="color:#aac4ef;font-size:0.72rem;margin-top:2px">{day_str} {holiday_tag}</div>
            </div>
            <div>
                <span class="ticket-tag">{stops}</span>
                <span class="ticket-tag">{duration_hrs}h</span>
                <span class="ticket-tag">{int(passengers)} pax</span>
                <span class="ticket-tag">{'🤖 ML Model' if MODEL_LOADED else '📊 Estimator'}</span>
            </div>
        </div>
        <div class="ticket-body">
            <div class="ticket-city">
                <div class="code">{src_code}</div>
                <div class="name">{source}</div>
                <div class="time">{dep_str}</div>
            </div>
            <div class="ticket-mid">
                <div class="stops-label">{stops.upper()}</div>
                <div class="dash-line"><span class="plane" style="background:var(--bg-card,#fff)">✈️</span></div>
                <div class="dur">
                    <span style="color:#1e2b4a;font-weight:600">{duration_hrs}h</span>
                    &nbsp;·&nbsp;
                    <span style="color:#0f4c9a;font-weight:600">{dist_str}</span>
                </div>
            </div>
            <div class="ticket-city">
                <div class="code">{dst_code}</div>
                <div class="name">{destination}</div>
            </div>
        </div>
        <div class="ticket-footer">
            <div>
                <div class="price-label">Total Estimated Fare · {int(passengers)} Pax</div>
                <div class="price-amount">{sym}{price_d:,.0f}</div>
                <span class="{vs_class}">{vs_text}</span>
            </div>
            <div class="route-info-block">
                <div class="route-info-label">Route Info</div>
                <div class="route-info-items">
                    <div class="route-info-item">
                        <span class="icon">📍</span>
                        <span class="val">{dist_str}</span>
                        <span class="sub">Distance</span>
                    </div>
                    <div class="route-info-item">
                        <span class="icon">⚡</span>
                        <span class="val">{speed_str}</span>
                        <span class="sub">Avg Speed</span>
                    </div>
                    <div class="route-info-item">
                        <span class="icon">💰</span>
                        <span class="val" style="color:var(--primary)">{price_per_km}</span>
                        <span class="sub">Cost/km</span>
                    </div>
                </div>
            </div>
            <div class="per-pax-block">
                <div class="per-pax-label">Per Passenger</div>
                <div class="per-pax-val">{sym}{price_d / passengers:,.0f}</div>
                <div class="dataset-avg">Dataset avg: {sym}{avg_d:,.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_range:
        st.markdown(
            '<div class="output-section-title">📉 Output B · Price Range</div>',
            unsafe_allow_html=True
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric('🟢 Low Estimate',  f'{sym}{low_d:,.0f}',  delta=f'-{sym}{price_d - low_d:,.0f}')
        b2.metric('🎯 Predicted',     f'{sym}{price_d:,.0f}')
        b3.metric('🔴 High Estimate', f'{sym}{high_d:,.0f}', delta=f'+{sym}{high_d - price_d:,.0f}')
        b4.metric('📊 Dataset Avg',   f'{sym}{avg_d:,.0f}')

    st.markdown(
        '<div class="output-section-title">📊 Output C · Price Comparison Visualizations</div>',
        unsafe_allow_html=True
    )

    PLOT_BG    = 'rgba(0,0,0,0)'          # transparent — CSS card bg shows through
    GRID_COLOR = 'rgba(148,163,184,0.15)' # subtle grid, visible on both themes
    FONT_FMLY  = 'Plus Jakarta Sans, sans-serif'
    FONT_CLR   = '#94a3b8'                # neutral mid-gray, readable on both
    AXIS_LINE  = 'rgba(148,163,184,0.30)'
    PRIMARY    = '#0052cc'
    ACCENT     = '#f5a623'
    SUCCESS    = '#22c55e'
    DANGER     = '#ef4444'

    def base_layout(title, xaxis_title='', yaxis_title=''):
        return dict(
            title=dict(text=title, font=dict(family=FONT_FMLY, size=14, color=FONT_CLR),
                       x=0, xanchor='left', pad=dict(l=4, b=12)),
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(family=FONT_FMLY, color=FONT_CLR, size=11),
            xaxis=dict(title=xaxis_title, gridcolor=GRID_COLOR,
                       linecolor=AXIS_LINE, tickfont=dict(size=10, color=FONT_CLR)),
            yaxis=dict(title=yaxis_title, gridcolor=GRID_COLOR,
                       linecolor=AXIS_LINE, tickfont=dict(size=10, color=FONT_CLR), tickprefix=sym),
            margin=dict(l=10, r=10, t=46, b=10),
            hoverlabel=dict(bgcolor='#0f172a', font_size=12,
                           font_family=FONT_FMLY, font_color='white',
                           bordercolor='#1e293b'),
            showlegend=False
        )

    _base = dict(source=source, destination=destination,
                 dep_hour=dep_hour, journey_month=journey_month,
                 journey_weekday=journey_weekday, journey_day=int(journey_day),
                 duration_hours=duration_hrs)
    _al_combos = [{**_base, 'airline': al} for al in AIRLINES]
    _al_raw    = batch_predict_app(_al_combos, int(passengers))
    al_prices  = {al: round(p * fac, 2) for al, p in zip(AIRLINES, _al_raw)}

    al_df = (
        pd.DataFrame({'Airline': list(al_prices.keys()), 'Price': list(al_prices.values())})
        .sort_values('Price').reset_index(drop=True)
    )
    al_df['Selected'] = al_df['Airline'] == airline
    al_df['Label']    = al_df['Price'].apply(lambda v: f'{sym}{v:,.0f}')

    viz_c1, viz_c2 = st.columns(2)

    with viz_c1:
        st.markdown('<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">'
                    '🏷️ All Airlines · Ranked by Price</div>', unsafe_allow_html=True)
        bar_colors = [PRIMARY if a == airline else
                      ('rgba(0,82,204,0.25)' if al_prices[a] < price_d else 'rgba(239,68,68,0.25)')
                      for a in al_df['Airline']]
        fig1 = go.Figure(go.Bar(
            x=al_df['Price'], y=al_df['Airline'], orientation='h',
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=al_df['Label'], textposition='outside',
            textfont=dict(size=10, family=FONT_FMLY, color=FONT_CLR),
            hovertemplate='<b>%{y}</b><br>Price: ' + sym + '%{x:,.0f}<extra></extra>',
            width=0.65
        ))
        fig1.add_vline(x=price_d, line_width=2, line_dash='dash', line_color=ACCENT,
                       annotation_text='Your pick',
                       annotation_font=dict(size=10, color=ACCENT, family=FONT_FMLY),
                       annotation_position='top right')
        layout1 = base_layout('', xaxis_title=f'Price ({sym})')
        layout1['yaxis']['title'] = ''
        layout1['margin'] = dict(l=10, r=120, t=10, b=10)
        layout1['height'] = 340
        # Pad x-axis so outside text labels have room
        _max_price = al_df['Price'].max()
        layout1['xaxis']['range'] = [0, _max_price * 1.28]
        fig1.update_layout(**layout1)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    with viz_c2:
        st.markdown('<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">'
                    '🔄 Price by Number of Stops</div>', unsafe_allow_html=True)
        cheapest_al = al_df.iloc[0]['Airline']
        stops_data  = {}
        for st_opt in STOPS:
            stops_data[st_opt] = {
                'selected': predict_price(airline, source, destination, st_opt,
                                         dep_hour, journey_month, journey_weekday,
                                         int(journey_day), duration_hrs, int(passengers)) * fac,
                'cheapest': predict_price(cheapest_al, source, destination, st_opt,
                                         dep_hour, journey_month, journey_weekday,
                                         int(journey_day), duration_hrs, int(passengers)) * fac,
            }
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name=airline, x=STOPS,
            y=[stops_data[s]['selected'] for s in STOPS],
            marker_color=PRIMARY,
            text=[f'{sym}{stops_data[s]["selected"]:,.0f}' for s in STOPS],
            textposition='outside', textfont=dict(size=9, family=FONT_FMLY),
            hovertemplate='<b>' + airline + '</b><br>%{x}<br>' + sym + '%{y:,.0f}<extra></extra>',
        ))
        if cheapest_al != airline:
            fig2.add_trace(go.Bar(
                name=cheapest_al, x=STOPS,
                y=[stops_data[s]['cheapest'] for s in STOPS],
                marker_color='rgba(0,82,204,0.25)',
                text=[f'{sym}{stops_data[s]["cheapest"]:,.0f}' for s in STOPS],
                textposition='outside',
                textfont=dict(size=9, family=FONT_FMLY, color=FONT_CLR),
                hovertemplate='<b>' + cheapest_al + '</b><br>%{x}<br>' + sym + '%{y:,.0f}<extra></extra>',
            ))
        layout2 = base_layout('', xaxis_title='Stops', yaxis_title=f'Price ({sym})')
        layout2['barmode'] = 'group'
        layout2['height']  = 340
        layout2['showlegend'] = True
        layout2['legend'] = dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='left', x=0, font=dict(size=10, family=FONT_FMLY))
        layout2['margin'] = dict(l=10, r=10, t=40, b=10)
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    viz_c3, viz_c4 = st.columns(2)

    with viz_c3:
        st.markdown('<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">'
                    '⏰ Price vs Departure Hour (Top 3 Airlines)</div>', unsafe_allow_html=True)
        mid_idx = len(al_df) // 2
        mid_al  = al_df.iloc[mid_idx]['Airline']
        top3    = list(dict.fromkeys([cheapest_al, airline, mid_al]))[:3]
        hours   = list(range(0, 24))
        fig3    = go.Figure()
        line_colors = [SUCCESS, PRIMARY, ACCENT]
        _c3_combos = [{**_base, 'airline': al, 'dep_hour': h} for al in top3 for h in hours]
        _c3_raw    = batch_predict_app(_c3_combos, int(passengers))
        _c3_prices = {al: [round(_c3_raw[ai*24+hi]*fac, 2) for hi in range(24)]
                      for ai, al in enumerate(top3)}
        for idx, al in enumerate(top3):
            is_selected = (al == airline)
            fig3.add_trace(go.Scatter(
                x=hours, y=_c3_prices[al], name=al,
                mode='lines+markers',
                line=dict(color=line_colors[idx],
                          width=2.5 if is_selected else 1.5,
                          dash='solid' if is_selected else 'dot'),
                marker=dict(size=5 if is_selected else 3),
                hovertemplate='<b>' + al + '</b><br>Hour: %{x}<br>Price: ' +
                              sym + '%{y:,.0f}<extra></extra>',
            ))
        fig3.add_vline(x=dep_hour, line_width=1.5, line_dash='dash', line_color='#94a3b8',
                       annotation_text=f'{dep_hour:02d}:00',
                       annotation_font=dict(size=9, family=FONT_FMLY, color='#94a3b8'),
                       annotation_position='top')
        layout3 = base_layout('', xaxis_title='Hour', yaxis_title=f'Price ({sym})')
        layout3['height'] = 300
        layout3['showlegend'] = True
        layout3['legend'] = dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='left', x=0, font=dict(size=9, family=FONT_FMLY))
        layout3['margin'] = dict(l=10, r=10, t=40, b=40)
        fig3.update_layout(**layout3)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with viz_c4:
        st.markdown('<div style="font-size:0.72rem;font-weight:700;color:#64748b;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">'
                    '⏱️ Price vs Duration (All Airlines)</div>', unsafe_allow_html=True)
        _c4_meta   = [(al, st_opt, predict_duration(source, destination, st_opt))
                      for al in AIRLINES for st_opt in STOPS]
        _c4_combos = [{**_base, 'airline': al, 'duration_hours': dur}
                      for al, st_opt, dur in _c4_meta]
        _c4_raw    = batch_predict_app(_c4_combos, int(passengers))
        scatter_rows = []
        for (al, st_opt, dur), p_raw in zip(_c4_meta, _c4_raw):
            scatter_rows.append({
                'Airline':  al, 'Duration': dur,
                'Price':    round(p_raw * fac, 2),
                'Stops':    st_opt,
                'Selected': (al == airline and st_opt == stops),
            })
        sc_df = pd.DataFrame(scatter_rows)
        fig4  = go.Figure()
        mask_other = ~sc_df['Selected']
        fig4.add_trace(go.Scatter(
            x=sc_df[mask_other]['Duration'], y=sc_df[mask_other]['Price'],
            mode='markers', name='Other options',
            marker=dict(color='rgba(100,116,139,0.35)', size=8, line=dict(color='#94a3b8', width=1)),
            customdata=sc_df[mask_other][['Airline','Stops']].values,
            hovertemplate='<b>%{customdata[0]}</b><br>Stops: %{customdata[1]}<br>'
                          'Duration: %{x:.1f}h<br>Price: ' + sym + '%{y:,.0f}<extra></extra>',
        ))
        mask_sel = sc_df['Selected']
        if mask_sel.any():
            fig4.add_trace(go.Scatter(
                x=sc_df[mask_sel]['Duration'], y=sc_df[mask_sel]['Price'],
                mode='markers+text', name='Your pick',
                marker=dict(color=PRIMARY, size=14, line=dict(color='white', width=2), symbol='star'),
                text=[f'  {airline}<br>  {sym}{price_d:,.0f}'],
                textposition='middle right',
                textfont=dict(size=10, color=PRIMARY, family=FONT_FMLY),
                hovertemplate='<b>⭐ YOUR PICK</b><br>' + airline + '<br>'
                              'Duration: %{x:.1f}h<br>Price: ' + sym + '%{y:,.0f}<extra></extra>',
            ))
        fig4.add_hline(y=avg_d, line_width=1.5, line_dash='dash', line_color=ACCENT,
                       annotation_text=f'Dataset avg {sym}{avg_d:,.0f}',
                       annotation_font=dict(size=9, color=ACCENT, family=FONT_FMLY),
                       annotation_position='bottom right')
        layout4 = base_layout('', xaxis_title='Duration (hours)', yaxis_title=f'Price ({sym})')
        layout4['height'] = 300
        layout4['showlegend'] = True
        layout4['legend'] = dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='left', x=0, font=dict(size=9, family=FONT_FMLY))
        layout4['margin'] = dict(l=10, r=10, t=40, b=10)
        fig4.update_layout(**layout4)
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

    st.markdown(
        '<div class="output-section-title">📋 Output D · Full Price Comparison Table</div>',
        unsafe_allow_html=True
    )
    st.caption(f'All {len(AIRLINES)} airlines · {source} → {destination} · '
               f'{stops} · {travel_date.strftime("%d %b %Y")} · '
               f'{dep_hour:02d}:00 · {int(passengers)} pax · Sorted cheapest first.')
    table_rows = []
    for al in AIRLINES:
        p      = al_prices[al]
        dur    = predict_duration(source, destination, stops)
        diff_v = p - price_d
        saving = price_d - p
        table_rows.append({
            'Airline':             al,
            f'Predicted ({sym})':  f'{sym}{p:,.0f}',
            'vs Your Pick':        (f'🟢 Save {sym}{saving:,.0f}' if saving > 50
                                    else (f'🔴 +{sym}{abs(diff_v):,.0f}' if diff_v > 50 else '🟡 Same')),
            'vs Dataset Avg':      (f'✅ {sym}{avg_d-p:,.0f} below avg' if p < avg_d
                                    else f'⚠️ {sym}{p-avg_d:,.0f} above avg'),
            'Duration':            f'{dur:.1f}h',
            'Cost/km':             (f'{sym}{p/dist_km:.1f}/km' if dist_km > 0 else 'N/A'),
            'Rank':                '',
        })
    tdf = (pd.DataFrame(table_rows)
           .assign(Rank=lambda df: [f'#{i+1}' for i in range(len(df))]))
    tdf = tdf[['Rank','Airline', f'Predicted ({sym})', 'vs Your Pick',
               'vs Dataset Avg','Duration','Cost/km']]
    st.dataframe(tdf, use_container_width=True, hide_index=True)

    cheapest_price = al_df.iloc[0]['Price']
    priciest_price = al_df.iloc[-1]['Price']
    cheapest_name  = al_df.iloc[0]['Airline']
    priciest_name  = al_df.iloc[-1]['Airline']
    saving_vs_pick = price_d - cheapest_price

    st.markdown(f"""
    <div style="display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;">
        <div style="flex:1;min-width:160px;background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:0.65rem;font-weight:800;color:#16a34a;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
                🏆 Cheapest Option</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        font-weight:900;color:#0f172a">{cheapest_name}</div>
            <div style="font-size:0.85rem;color:#16a34a;font-weight:700">
                {sym}{cheapest_price:,.0f}</div>
        </div>
        <div style="flex:1;min-width:160px;background:#eff6ff;border:1px solid #bfdbfe;
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:0.65rem;font-weight:800;color:#0052cc;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
                ⭐ Your Selection</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        font-weight:900;color:#0f172a">{airline}</div>
            <div style="font-size:0.85rem;color:#0052cc;font-weight:700">
                {sym}{price_d:,.0f}</div>
        </div>
        <div style="flex:1;min-width:160px;background:#fff7ed;border:1px solid #fed7aa;
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:0.65rem;font-weight:800;color:#ea580c;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
                💡 Potential Saving</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        font-weight:900;color:#0f172a">
                {f"Switch to {cheapest_name}" if saving_vs_pick > 50 else "Best choice!"}</div>
            <div style="font-size:0.85rem;color:#ea580c;font-weight:700">
                {f"{sym}{saving_vs_pick:,.0f} cheaper" if saving_vs_pick > 50 else "Already cheapest"}</div>
        </div>
        <div style="flex:1;min-width:160px;background:#fdf4ff;border:1px solid #e9d5ff;
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:0.65rem;font-weight:800;color:#9333ea;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
                📊 Price Spread</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        font-weight:900;color:#0f172a">
                {sym}{priciest_price - cheapest_price:,.0f} range</div>
            <div style="font-size:0.85rem;color:#9333ea;font-weight:700">
                {cheapest_name} → {priciest_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Task 3 — SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander('🔀 Scenario Comparison — Compare up to 3 flight configurations', expanded=False):
    st.caption('Define up to 3 scenarios to compare. All prices from the real ML model.')

    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = [
            {'airline':'IndiGo',    'stops':'non-stop','dep_hour':6,  'month':4,'weekday':0,'label':'Scenario A'},
            {'airline':'SpiceJet',  'stops':'non-stop','dep_hour':10, 'month':4,'weekday':4,'label':'Scenario B'},
            {'airline':'Air India', 'stops':'1 stop',  'dep_hour':18, 'month':5,'weekday':5,'label':'Scenario C'},
        ]

    n_sc = st.radio('Number of scenarios', [2, 3], horizontal=True, index=0, key='_n_sc')
    sc_cols = st.columns(int(n_sc))

    for sci in range(int(n_sc)):
        sc = st.session_state.scenarios[sci]
        with sc_cols[sci]:
            st.markdown(f'<div style="font-size:0.65rem;font-weight:800;color:#0052cc;'
                        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">'
                        f'{sc["label"]}</div>', unsafe_allow_html=True)
            sc['airline']  = st.selectbox('Airline', AIRLINES,
                                           index=AIRLINES.index(sc['airline']),
                                           key=f'_sc_{sci}_al')
            _vst = VALID_AIRLINE_STOPS.get(sc['airline'], STOPS)
            if sc['stops'] not in _vst: sc['stops'] = _vst[0]
            sc['stops']    = st.selectbox('Stops', _vst,
                                           index=_vst.index(sc['stops']),
                                           key=f'_sc_{sci}_st')
            sc['dep_hour'] = st.slider('Dep Hour', 0, 23, sc['dep_hour'], key=f'_sc_{sci}_hr')
            sc['month']    = st.selectbox('Month', [3,4,5,6],
                                           format_func=lambda m: MONTHS[m],
                                           index=[3,4,5,6].index(sc['month']),
                                           key=f'_sc_{sci}_mo')
            sc['weekday']  = st.selectbox('Weekday', list(range(7)),
                                           format_func=lambda d: WEEKDAYS[d],
                                           index=sc['weekday'], key=f'_sc_{sci}_wd')

    if st.button('⚡ Compare Now', key='_sc_run', type='primary', use_container_width=True):
        _sc_src  = st.session_state.get('source', SOURCES[0])
        _sc_dsts = VALID_DESTINATIONS.get(_sc_src, DESTINATIONS)
        _sc_dst  = st.session_state.get('destination', _sc_dsts[0])
        _active  = st.session_state.scenarios[:int(n_sc)]
        _sc_durs = [predict_duration(_sc_src, _sc_dst, sc['stops']) for sc in _active]
        _sc_combos = [
            dict(airline=sc['airline'], source=_sc_src, destination=_sc_dst,
                 dep_hour=sc['dep_hour'], journey_month=sc['month'],
                 journey_weekday=sc['weekday'], journey_day=15,
                 duration_hours=dur)
            for sc, dur in zip(_active, _sc_durs)
        ]
        _sc_prices = batch_predict_app(_sc_combos, 1)
        _sc_min    = min(_sc_prices)

        res_cols = st.columns(int(n_sc))
        for sci, (sc, p_inr, dur) in enumerate(zip(_active, _sc_prices, _sc_durs)):
            p_d  = round(p_inr * fac, 0)
            best = (p_inr == _sc_min)
            bg   = '#f0fdf4' if best else '#eff6ff'
            bdr  = '#22c55e' if best else '#bfdbfe'
            with res_cols[sci]:
                st.markdown(
                    f'<div style="background:{bg};border:2px solid {bdr};'
                    f'border-radius:12px;padding:16px;text-align:center;">'
                    f'<div style="font-size:0.62rem;font-weight:800;color:#64748b;'
                    f'text-transform:uppercase;letter-spacing:1px">'
                    f'{sc["label"]}{"  🏆" if best else ""}</div>'
                    f'<div style="font-size:0.82rem;color:#0f172a;font-weight:700;margin:4px 0">'
                    f'{sc["airline"]}</div>'
                    f'<div style="font-size:0.7rem;color:#64748b">'
                    f'{sc["stops"]} · {MONTHS[sc["month"]]} · '
                    f'{WEEKDAYS[sc["weekday"]][:3]} · {sc["dep_hour"]:02d}:00 · {dur:.1f}h</div>'
                    f'<div style="font-family:Syne,sans-serif;font-size:2rem;'
                    f'font-weight:900;color:#0052cc;margin:8px 0">{sym}{p_d:,.0f}</div>'
                    f'<div style="font-size:0.68rem;color:#94a3b8">'
                    f'{_sc_src} → {_sc_dst} · 1 pax</div>'
                    f'</div>', unsafe_allow_html=True
                )

        _fig_sc = go.Figure(go.Bar(
            x=[sc['label'] for sc in _active],
            y=[round(p*fac) for p in _sc_prices],
            marker_color=['#22c55e' if p==_sc_min else '#0052cc' for p in _sc_prices],
            text=[f'{sym}{round(p*fac):,}' for p in _sc_prices],
            textposition='outside',
            textfont=dict(size=12, family='Syne, sans-serif'),
            width=0.5
        ))
        _fig_sc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Plus Jakarta Sans, sans-serif', color='#94a3b8'),
            yaxis=dict(tickprefix=sym, gridcolor='rgba(148,163,184,0.15)',
                       linecolor='rgba(148,163,184,0.3)', tickfont=dict(color='#94a3b8')),
            xaxis=dict(tickfont=dict(size=13, family='Syne, sans-serif', color='#94a3b8')),
            margin=dict(l=10,r=10,t=30,b=10), height=260, showlegend=False
        )
        st.plotly_chart(_fig_sc, use_container_width=True, config={'displayModeBar': False})

        if len(_sc_prices) > 1:
            _sorted = sorted(zip(_sc_prices, _active, _sc_durs))
            _cheap_p, _cheap_sc, _ = _sorted[0]
            _max_p   = _sorted[-1][0]
            _saving  = round((_max_p - _cheap_p)*fac)
            if _saving > 0:
                st.info(
                    f'💡 Switch from **{_sorted[-1][1]["label"]}** to '
                    f'**{_cheap_sc["label"]}** ({_cheap_sc["airline"]}, '
                    f'{_cheap_sc["stops"]}) to save **{sym}{_saving:,}** per passenger.'
                )
