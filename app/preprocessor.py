"""
AirFair-Vista — Preprocessing Pipeline
pipeline/preprocessor.py

WHY THIS FILE EXISTS:
  All feature engineering, lookup tables, distance calculation, validation
  rules and prediction functions live here — separate from app.py (UI).

WHY SEPARATION:
  1. app.py re-runs on every widget change. Heavy logic must NOT be there.
  2. This file can be unit-tested without Streamlit.
  3. Other consumers (batch jobs, REST API) import the same pipeline.
  4. Streamlit caches the imported module — zero re-execution on reruns.

USAGE IN app.py:
  import sys
  sys.path.insert(0, '/content/drive/MyDrive/AirFair-Vista/pipeline')
  from preprocessor import (AIRLINES, SOURCES, ..., batch_predict, ...)
"""

import math
import numpy as np
import pandas as pd

# ── Section 1: Dropdown options ───────────────────────────────────
AIRLINES = [
    'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
    'Jet Airways Business', 'Multiple Carriers',
    'Multiple Carriers Premium Economy',
    'SpiceJet', 'TruJet', 'Vistara', 'Vistara Premium Economy'
]
SOURCES      = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
DESTINATIONS = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
STOPS        = ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']
MONTHS       = {3: 'March', 4: 'April', 5: 'May', 6: 'June'}
WEEKDAYS     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# ── Section 2: Feature lookup tables (real dataset means) ─────────
AIRLINE_MEAN_PRICE = {
    'Air Asia': 5590.26,        'Air India': 9612.43,
    'GoAir': 5861.06,           'IndiGo': 5673.68,
    'Jet Airways': 11643.92,    'Jet Airways Business': 58358.67,
    'Multiple Carriers': 10902.68,
    'Multiple Carriers Premium Economy': 11418.85,
    'SpiceJet': 4338.28,        'TruJet': 4140.00,
    'Vistara': 7796.35,         'Vistara Premium Economy': 8962.33
}
SOURCE_FREQ  = {'Banglore':0.2057,'Chennai':0.0357,'Delhi':0.4246,'Kolkata':0.2688,'Mumbai':0.0652}
DEST_FREQ    = {'Banglore':0.2688,'Cochin':0.4246,'Delhi':0.1184,'Hyderabad':0.0652,'Kolkata':0.0357,'New Delhi':0.0872}
SOURCE_MEAN_PRICE = {'Banglore':8017.46,'Chennai':4789.89,'Delhi':10540.11,'Kolkata':9158.39,'Mumbai':5059.71}

PRICE_MIN=1759; PRICE_MAX=79512; PRICE_AVG=9087; PRICE_MED=8372

AIRLINE_MEAN = {'Air Asia':5590,'Air India':9612,'GoAir':5861,'IndiGo':5674,
                'Jet Airways':11644,'Jet Airways Business':58359,'Multiple Carriers':10903,
                'Multiple Carriers Premium Economy':11419,'SpiceJet':4338,'TruJet':4140,
                'Vistara':7796,'Vistara Premium Economy':8962}
STOPS_MEAN   = {'non-stop':5025,'1 stop':10594,'2 stops':12716,'3 stops':13112,'4 stops':17686}
ROUTE_MEAN   = {('Banglore','Delhi'):5144,('Banglore','New Delhi'):11918,
                ('Chennai','Kolkata'):4790,('Delhi','Cochin'):10540,
                ('Kolkata','Banglore'):9158,('Mumbai','Hyderabad'):5060}
HOUR_MEAN    = {0:7615,1:4355,2:8420,3:10475,4:7252,5:9682,6:8314,7:8496,
                8:10083,9:9644,10:8928,11:9290,12:9252,13:9064,14:9906,
                15:7687,16:10320,17:8737,18:10036,19:8485,20:9671,21:8456,22:7858,23:9474}
MONTH_MEAN   = {3:10673,4:5771,5:9128,6:8829}
WEEKDAY_MEAN = {0:8500,1:9026,2:9278,3:8931,4:9718,5:8973,6:9526}

# ── Section 3: Duration prediction ────────────────────────────────
DURATION_LOOKUP = {
    ('Banglore','Delhi','non-stop'):2.22, ('Banglore','New Delhi','non-stop'):2.03,
    ('Banglore','New Delhi','1 stop'):12.88, ('Chennai','Kolkata','non-stop'):2.00,
    ('Delhi','Cochin','non-stop'):2.85, ('Delhi','Cochin','1 stop'):11.35,
    ('Kolkata','Banglore','non-stop'):2.00, ('Kolkata','Banglore','1 stop'):14.60,
    ('Mumbai','Hyderabad','non-stop'):1.00, ('Mumbai','Hyderabad','1 stop'):16.02,
}
AVG_SPEED_BY_STOPS = {'non-stop':756.8,'1 stop':198.9,'2 stops':116.8,'3 stops':93.8,'4 stops':60.0}
CITY_COORDS = {
    'Banglore':(12.9716,77.5946),'Chennai':(13.0827,80.2707),
    'Delhi':(28.7041,77.1025),'New Delhi':(28.6139,77.2090),
    'Kolkata':(22.5726,88.3639),'Mumbai':(19.0760,72.8777),
    'Cochin':(9.9312,76.2673),'Hyderabad':(17.3850,78.4867),
}

# ── Section 4: Validation constants ───────────────────────────────
VALID_ROUTES = {('Banglore','Delhi'),('Banglore','New Delhi'),('Chennai','Kolkata'),
                ('Delhi','Cochin'),('Kolkata','Banglore'),('Mumbai','Hyderabad')}
VALID_DESTINATIONS = {'Banglore':['Delhi','New Delhi'],'Chennai':['Kolkata'],
                      'Delhi':['Cochin'],'Kolkata':['Banglore'],'Mumbai':['Hyderabad']}
VALID_AIRLINE_STOPS = {
    'Air Asia':['non-stop','1 stop','2 stops'],
    'Air India':['non-stop','1 stop','2 stops','3 stops','4 stops'],
    'GoAir':['non-stop','1 stop'], 'IndiGo':['non-stop','1 stop','2 stops'],
    'Jet Airways':['non-stop','1 stop','2 stops'], 'Jet Airways Business':['1 stop','2 stops'],
    'Multiple Carriers':['1 stop','2 stops','3 stops'],
    'Multiple Carriers Premium Economy':['1 stop'],
    'SpiceJet':['non-stop','1 stop'], 'TruJet':['1 stop'],
    'Vistara':['non-stop','1 stop'], 'Vistara Premium Economy':['non-stop'],
}
MAX_PAX_BY_STOPS = {'non-stop':9,'1 stop':9,'2 stops':6,'3 stops':4,'4 stops':2}
INDIAN_HOLIDAYS  = {(3,25):'Holi',(3,29):'Good Friday',(4,14):'Ambedkar Jayanti',
                    (4,17):'Ram Navami',(4,19):'Mahavir Jayanti',
                    (5,1):'Labour Day',(5,23):'Buddha Purnima',(6,5):'Eid ul-Fitr'}

# ── Section 5: Model config ────────────────────────────────────────

import os
import gdown

MODEL_PATH = 'models/flight_price_prediction_pipeline.pkl'

# Auto-download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    FILE_ID = 'PASTE_YOUR_FILE_ID_HERE'
    gdown.download(
        f'https://drive.google.com/uc?id={FILE_ID}',
        MODEL_PATH,
        quiet=False
    )

MODEL_FEATURES = [
    'journey_day','journey_month','journey_weekday','is_weekend','quarter',
    'dep_hour','weekday','is_holiday','duration_hours','duration_minutes',
    'total_duration_mins','Source_freq','Destination_freq',
    'Airline_mean_price','Source_mean_price',
    'total_duration_mins.1','journey_month.1',
    'total_duration_mins^2','total_duration_mins journey_month','journey_month^2'
]

# ── Section 6: Pure functions (NO Streamlit dependency) ──────────
def haversine_km(city1: str, city2: str) -> float:
    """Haversine great-circle distance. WHY not Euclidean: Earth is not flat."""
    if city1==city2: return 0.0
    c1,c2 = CITY_COORDS.get(city1), CITY_COORDS.get(city2)
    if not c1 or not c2: return 0.0
    R=6371.0; la1,lo1=math.radians(c1[0]),math.radians(c1[1])
    la2,lo2=math.radians(c2[0]),math.radians(c2[1])
    a=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return round(2*R*math.asin(math.sqrt(a)),1)

def is_indian_holiday(month: int, day: int) -> int:
    return 1 if (month,day) in INDIAN_HOLIDAYS else 0

def predict_duration(source: str, destination: str, stops: str) -> float:
    """Tier 1: dataset lookup. Tier 2: Haversine/speed. Tier 3: global mean."""
    key=(source,destination,stops)
    if key in DURATION_LOOKUP: return DURATION_LOOKUP[key]
    dist=haversine_km(source,destination); speed=AVG_SPEED_BY_STOPS.get(stops,300.0)
    if dist>0 and speed>0: return max(0.5, round(dist/speed,1))
    return 10.2

def get_validation_errors(source, destination, airline, stops, passengers, dep_hour):
    errors,warnings=[],[]
    if source.lower()==destination.lower():
        errors.append('Source and destination cannot be the same city.')
    elif (source,destination) not in VALID_ROUTES:
        rev=(destination,source)
        hint=f' Try {destination}→{source} instead.' if rev in VALID_ROUTES else ''
        errors.append(f'Route {source}→{destination} has no records in dataset.{hint}')
    vst=VALID_AIRLINE_STOPS.get(airline,[])
    if stops not in vst:
        errors.append(f'{airline} does not operate {stops}. Valid: {", ".join(vst)}.')
    if passengers > MAX_PAX_BY_STOPS.get(stops,9):
        errors.append(f'{stops} supports max {MAX_PAX_BY_STOPS[stops]} pax.')
    if airline=='Jet Airways Business' and stops=='non-stop':
        warnings.append('Jet Airways Business: no non-stop records in dataset.')
    if dep_hour in [2,3,4]:
        warnings.append(f'{dep_hour:02d}:00 is a red-eye slot — low data confidence.')
    if airline=='TruJet':
        warnings.append('TruJet has only 1 flight record in dataset.')
    if airline=='Multiple Carriers Premium Economy':
        warnings.append('Multiple Carriers Premium Economy has only 13 records.')
    return errors,warnings

def build_features(airline, source, destination, dep_hour, journey_month,
                   journey_weekday, journey_day, duration_hours) -> pd.DataFrame:
    """Single-row feature DataFrame. WHY named DataFrame not numpy: RF uses feature names."""
    tm=duration_hours*60.0; dm=(duration_hours%1)*60.0
    row={'journey_day':journey_day,'journey_month':journey_month,
         'journey_weekday':journey_weekday,'is_weekend':1 if journey_weekday>=5 else 0,
         'quarter':(journey_month-1)//3+1,'dep_hour':dep_hour,'weekday':journey_weekday,
         'is_holiday':0,'duration_hours':duration_hours,'duration_minutes':dm,
         'total_duration_mins':tm,'Source_freq':SOURCE_FREQ.get(source,0.2),
         'Destination_freq':DEST_FREQ.get(destination,0.2),
         'Airline_mean_price':AIRLINE_MEAN_PRICE.get(airline,PRICE_AVG),
         'Source_mean_price':SOURCE_MEAN_PRICE.get(source,PRICE_AVG),
         'total_duration_mins.1':tm,'journey_month.1':journey_month,
         'total_duration_mins^2':tm**2,'total_duration_mins journey_month':tm*journey_month,
         'journey_month^2':journey_month**2}
    return pd.DataFrame([row])[MODEL_FEATURES]

def build_feature_matrix(combos: list) -> pd.DataFrame:
    """
    N-row feature matrix for batch prediction.
    WHY batch: RF.predict() has ~90ms fixed overhead per call.
      Sequential 12 calls = 1,360ms. One batch call = 110ms. (12x faster)
      Sequential 288 calls = 32s. One batch call = 117ms. (276x faster)
    """
    rows=[]
    for c in combos:
        tm=c['duration_hours']*60.0; dm=(c['duration_hours']%1)*60.0
        wday=c['journey_weekday']; mon=c['journey_month']
        rows.append({'journey_day':c['journey_day'],'journey_month':mon,
                     'journey_weekday':wday,'is_weekend':1 if wday>=5 else 0,
                     'quarter':(mon-1)//3+1,'dep_hour':c['dep_hour'],'weekday':wday,
                     'is_holiday':0,'duration_hours':c['duration_hours'],
                     'duration_minutes':dm,'total_duration_mins':tm,
                     'Source_freq':SOURCE_FREQ.get(c['source'],0.2),
                     'Destination_freq':DEST_FREQ.get(c['destination'],0.2),
                     'Airline_mean_price':AIRLINE_MEAN_PRICE.get(c['airline'],PRICE_AVG),
                     'Source_mean_price':SOURCE_MEAN_PRICE.get(c['source'],PRICE_AVG),
                     'total_duration_mins.1':tm,'journey_month.1':mon,
                     'total_duration_mins^2':tm**2,
                     'total_duration_mins journey_month':tm*mon,'journey_month^2':mon**2})
    return pd.DataFrame(rows)[MODEL_FEATURES]

def batch_predict(model, combos: list, passengers: int=1, fallback_fn=None) -> list:
    """
    Predict N combos in ONE model.predict() call.
    WHY model passed as arg: preprocessor has no Streamlit/cache dependency.
    Model lives in app.py; passing it keeps this function pure and testable.
    """
    if model is not None and len(combos)>0:
        try:
            raws=model.predict(build_feature_matrix(combos))
            return [round(float(np.expm1(r) if r<15 else r)*passengers,2) for r in raws]
        except Exception: pass
    if fallback_fn: return [fallback_fn(c,passengers) for c in combos]
    out=[]
    for c in combos:
        base=(AIRLINE_MEAN.get(c['airline'],PRICE_AVG)*0.30+
              STOPS_MEAN.get(c.get('stops','non-stop'),PRICE_AVG)*0.25+
              ROUTE_MEAN.get((c['source'],c['destination']),PRICE_AVG)*0.20+
              HOUR_MEAN.get(c['dep_hour'],PRICE_AVG)*0.10+
              MONTH_MEAN.get(c['journey_month'],PRICE_AVG)*0.10+
              WEEKDAY_MEAN.get(c['journey_weekday'],PRICE_AVG)*0.05)
        out.append(round(base*(1+max(0,c['duration_hours']-2)*0.04)*passengers,2))
    return out

