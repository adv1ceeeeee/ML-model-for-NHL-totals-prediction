from sklearn.base import clone
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from geopy.distance import geodesic
from joblib import Memory
from lightgbm import LGBMRegressor
from pykalman import KalmanFilter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (StackingRegressor, RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from colorama import init, Fore, Style
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from nhlpy import NHLClient
from datetime import datetime
from meteostat import Point, Daily
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import tsfel
import hashlib
import logging
import warnings
import time
import requests
import cairosvg

# Инициализация colorama
init()

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('nhlpy').setLevel(logging.WARNING)

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return f"{Fore.LIGHTWHITE_EX}{message}{Style.RESET_ALL}"

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
    force=True,
    encoding='utf-8',
)

# ─────────────────── Константы ────────────────────
# Данные для подключения к PostgreSQL
PG_HOST = "localhost"
PG_DATABASE = "NHL_project"
PG_USER = "postgres"
PG_PASSWORD = "89137961052"
PG_PORT = "5432"
DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# Прочие глобалки
BATCH_SIZE = 1000
RANDOM_SEED = 42

# Для OddsAPI
API_KEY = "57f43d922c809e12c4d5e0aea4b4e10f"
BASE_URL = "https://api.the-odds-api.com/v4/"

TEAM_STRUCTURE = {
    "ATLANTIC": {
        "conference": "EAST",
        "teams": ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"]
    },
    "METROPOLITAN": {
        "conference": "EAST",
        "teams": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"]
    },
    "CENTRAL": {
        "conference": "WEST",
        "teams": ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "WPG", "UTA"]
    },
    "PACIFIC": {
        "conference": "WEST",
        "teams": ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"]
    }
} # Словарь соответствия команд и их позиций в НХЛ

TEAM_VENUE_MAPPING = {
    "ANA": "Honda Center",
    "ARI": "Mullett Arena",
    "BOS": "TD Garden",
    "BUF": "KeyBank Center",
    "CGY": "Scotiabank Saddledome",
    "CAR": "PNC Arena",
    "CHI": "United Center",
    "COL": "Ball Arena",
    "CBJ": "Nationwide Arena",
    "DAL": "American Airlines Center",
    "DET": "Little Caesars Arena",
    "EDM": "Rogers Place",
    "FLA": "Amerant Bank Arena",
    "LAK": "Crypto.com Arena",
    "MIN": "Xcel Energy Center",
    "MTL": "Bell Centre",
    "NSH": "Bridgestone Arena",
    "NJD": "Prudential Center",
    "NYI": "UBS Arena",
    "NYR": "Madison Square Garden",
    "OTT": "Canadian Tire Centre",
    "PHI": "Wells Fargo Center",
    "PIT": "PPG Paints Arena",
    "SJS": "SAP Center",
    "SEA": "Climate Pledge Arena",
    "STL": "Enterprise Center",
    "TBL": "Amalie Arena",
    "TOR": "Scotiabank Arena",
    "VAN": "Rogers Arena",
    "VGK": "T-Mobile Arena",
    "WSH": "Capital One Arena",
    "WPG": "Canada Life Centre",
    "UTA": "Delta Center"
} # Словарь соответствия команд и их домашних арен

TEAM_FULL_NAMES = {
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'DET': 'Detroit Red Wings',
    'FLA': 'Florida Panthers',
    'MTL': 'Montreal Canadiens',
    'OTT': 'Ottawa Senators',
    'TBL': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs',
    'CAR': 'Carolina Hurricanes',
    'CBJ': 'Columbus Blue Jackets',
    'NJD': 'New Jersey Devils',
    'NYI': 'New York Islanders',
    'NYR': 'New York Rangers',
    'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins',
    'WSH': 'Washington Capitals',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'DAL': 'Dallas Stars',
    'MIN': 'Minnesota Wild',
    'NSH': 'Nashville Predators',
    'STL': 'St. Louis Blues',
    'WPG': 'Winnipeg Jets',
    'ANA': 'Anaheim Ducks',
    'CGY': 'Calgary Flames',
    'EDM': 'Edmonton Oilers',
    'LAK': 'Los Angeles Kings',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights'
} # Словарь полного названия команд

TEAM_CITY_COORDS = {
    "ANA": (33.8353, -117.9145), "ARI": (33.4450, -112.0095), "BOS": (42.3601, -71.0589),
    "BUF": (42.8864, -78.8784), "CGY": (51.0447, -114.0719), "CAR": (35.7796, -78.6382),
    "CHI": (41.8781, -87.6298), "COL": (39.7392, -104.9903), "CBJ": (39.9612, -82.9988),
    "DAL": (32.7767, -96.7970), "DET": (42.3314, -83.0458), "EDM": (53.5461, -113.4938),
    "FLA": (26.1224, -80.1373), "LAK": (34.0522, -118.2437), "MIN": (44.9778, -93.2650),
    "MTL": (45.5017, -73.5673), "NSH": (36.1627, -86.7816), "NJD": (40.7357, -74.1724),
    "NYI": (40.7891, -73.1350), "NYR": (40.7128, -74.0060), "OTT": (45.4215, -75.6990),
    "PHI": (39.9526, -75.1652), "PIT": (40.4406, -79.9959), "SJS": (37.3382, -121.8863),
    "SEA": (47.6062, -122.3321), "STL": (38.6270, -90.1994), "TBL": (27.9506, -82.4572),
    "TOR": (43.651070, -79.347015), "VAN": (49.2827, -123.1207), "VGK": (36.1699, -115.1398),
    "WSH": (38.9072, -77.0369), "WPG": (49.8951, -97.1384), "UTA": (40.7608, -111.8910)
} # Словарь соответствия команд и их координат домашних арен

warnings.filterwarnings("ignore")
memory = Memory(location='cache', verbose=0)

# ─────────────────── Вспомогательные функции ────────────────────
def get_pg_engine():
    try:
        engine = create_engine(DATABASE_URL)
        logging.info("Успешное подключение к PostgreSQL через SQLAlchemy")
        return engine
    except Exception as e:
        logging.error(f"Ошибка подключения к PostgreSQL: {str(e)}")
        raise

@memory.cache
def get_avg_temperature(lat: float, lon: float, date: datetime) -> float:
    """Получает среднюю температуру для заданной даты и координат с помощью Meteostat."""
    try:
        location = Point(lat, lon)
        data = Daily(location, date, date)
        data = data.fetch()
        if not data.empty:
            return data['tavg'].iloc[0]  # Средняя температура за день
        return None
    except Exception as e:
        logging.error(f"Ошибка Meteostat: {e}")
        return None

@memory.cache
def get_historical_avg_temperature(lat: float, lon: float, month: int, day: int) -> float:
    """Получает среднюю историческую температуру за указанный месяц и день за последние 20 лет."""
    try:
        location = Point(lat, lon)
        end_year = datetime.now().year
        start_year = end_year - 20
        dates = [datetime(year, month, day) for year in range(start_year, end_year + 1)]
        data = Daily(location, dates[0], dates[-1])
        data = data.fetch()
        if not data.empty:
            return data['tavg'].mean()  # Средняя температура за указанный день и месяц
        return None
    except Exception as e:
        logging.error(f"Ошибка при получении исторической температуры: {e}")
        return None

def add_temperature_feature(df: pd.DataFrame, team_city_coords: dict) -> pd.DataFrame:
    df = df.copy()
    df['home_arena_avg_temp'] = None
    for index, row in df.iterrows():
        home_team = row['home_team']
        game_date = row['game_date']
        if home_team in team_city_coords and pd.notnull(game_date):
            lat, lon = team_city_coords[home_team]
            month, day = game_date.month, game_date.day
            temp = get_historical_avg_temperature(lat, lon, month, day)
            df.at[index, 'home_arena_avg_temp'] = temp
    df['home_arena_avg_temp'] = df['home_arena_avg_temp'].fillna(df['home_arena_avg_temp'].mean())
    return df

def hash_dataframe(df: pd.DataFrame) -> str:
    """Создает уникальный хэш для DataFrame на основе его содержимого."""
    relevant_cols = ['game_date', 'home_team', 'away_team']
    df_str = df[relevant_cols].to_string()
    return hashlib.md5(df_str.encode()).hexdigest()

@memory.cache
def add_city_distance_feature_cached(df: pd.DataFrame, df_hash: str) -> pd.DataFrame:
    """Внутренняя функция для кэширования расчетов расстояний с векторизацией."""
    df = df.copy()
    TEAM_CITY_COORDS = {
        "ANA": (33.8353, -117.9145), "ARI": (33.4450, -112.0095), "BOS": (42.3601, -71.0589),
        "BUF": (42.8864, -78.8784), "CGY": (51.0447, -114.0719), "CAR": (35.7796, -78.6382),
        "CHI": (41.8781, -87.6298), "COL": (39.7392, -104.9903), "CBJ": (39.9612, -82.9988),
        "DAL": (32.7767, -96.7970), "DET": (42.3314, -83.0458), "EDM": (53.5461, -113.4938),
        "FLA": (26.1224, -80.1373), "LAK": (34.0522, -118.2437), "MIN": (44.9778, -93.2650),
        "MTL": (45.5017, -73.5673), "NSH": (36.1627, -86.7816), "NJD": (40.7357, -74.1724),
        "NYI": (40.7891, -73.1350), "NYR": (40.7128, -74.0060), "OTT": (45.4215, -75.6990),
        "PHI": (39.9526, -75.1652), "PIT": (40.4406, -79.9959), "SJS": (37.3382, -121.8863),
        "SEA": (47.6062, -122.3321), "STL": (38.6270, -90.1994), "TBL": (27.9506, -82.4572),
        "TOR": (43.651070, -79.347015), "VAN": (49.2827, -123.1207), "VGK": (36.1699, -115.1398),
        "WSH": (38.9072, -77.0369), "WPG": (49.8951, -97.1384)
    }

    df["home_travel_distance"] = 0.0
    df["away_travel_distance"] = 0.0
    df["home_travel_fatigue"] = 0.0
    df["away_travel_fatigue"] = 0.0
    df["home_days_since_last_trip"] = 0
    df["away_days_since_last_trip"] = 0

    team_last_location = {}
    team_last_date = {}
    team_fatigue = {}
    all_teams = set(df["home_team"].unique()).union(set(df["away_team"].unique()))
    for team in all_teams:
        team_last_location[team] = None
        team_last_date[team] = None
        team_fatigue[team] = 0.0

    df = df.sort_values("game_date").reset_index(drop=True)

    # Векторизованные списки для накопления результатов
    home_travel_distances = []
    away_travel_distances = []
    home_travel_fatigues = []
    away_travel_fatigues = []
    home_days_since_last_trips = []
    away_days_since_last_trips = []

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        match_date = row["game_date"]
        current_location = home_team

        # Для домашней команды
        if team_last_location[home_team] is not None:
            prev_location = team_last_location[home_team]
            distance = geodesic(
                TEAM_CITY_COORDS.get(prev_location, (0, 0)),
                TEAM_CITY_COORDS.get(home_team, (0, 0))
            ).km
            days_rest = (match_date - team_last_date[home_team]).days
            fatigue = distance / 1000 * (1 - min(days_rest / 7, 1))
            team_fatigue[home_team] = team_fatigue[home_team] * 0.9 + fatigue
            home_travel_distances.append(distance)
            home_travel_fatigues.append(team_fatigue[home_team])
            home_days_since_last_trips.append(days_rest)
        else:
            home_travel_distances.append(0.0)
            home_travel_fatigues.append(0.0)
            home_days_since_last_trips.append(0)

        # Для гостевой команды
        if team_last_location[away_team] is not None:
            prev_location = team_last_location[away_team]
            distance = geodesic(
                TEAM_CITY_COORDS.get(prev_location, (0, 0)),
                TEAM_CITY_COORDS.get(current_location, (0, 0))
            ).km
            days_rest = (match_date - team_last_date[away_team]).days
            fatigue = distance / 1000 * (1 - min(days_rest / 7, 1))
            team_fatigue[away_team] = team_fatigue[away_team] * 0.9 + fatigue
            away_travel_distances.append(distance)
            away_travel_fatigues.append(team_fatigue[away_team])
            away_days_since_last_trips.append(days_rest)
        else:
            # Для первого матча сезона: расстояние от домашней арены away_team до home_team
            distance = geodesic(
                TEAM_CITY_COORDS.get(away_team, (0, 0)),
                TEAM_CITY_COORDS.get(home_team, (0, 0))
            ).km
            away_travel_distances.append(distance)
            away_travel_fatigues.append(0.0)
            away_days_since_last_trips.append(0)

        team_last_location[home_team] = home_team
        team_last_date[home_team] = match_date
        team_last_location[away_team] = current_location
        team_last_date[away_team] = match_date

    df["home_travel_distance"] = home_travel_distances
    df["away_travel_distance"] = away_travel_distances
    df["home_travel_fatigue"] = home_travel_fatigues
    df["away_travel_fatigue"] = away_travel_fatigues
    df["home_days_since_last_trip"] = home_days_since_last_trips
    df["away_days_since_last_trip"] = away_days_since_last_trips

    df["home_long_trip"] = (df["home_travel_distance"] > 2000).astype(int)
    df["away_long_trip"] = (df["away_travel_distance"] > 2000).astype(int)

    return df

def add_city_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Обертка для вызова кэшированной функции."""
    df_hash = hash_dataframe(df)
    return add_city_distance_feature_cached(df, df_hash)

def add_home_advantage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки домашнего преимущества."""
    df = df.copy()
    df = df.sort_values("game_date")
    try:
        df["home_goalies_savepctg"] = pd.to_numeric(df["home_goalies_savepctg"], errors='coerce').fillna(0)
        df["away_goalies_savepctg"] = pd.to_numeric(df["away_goalies_savepctg"], errors='coerce').fillna(0)
        df["home_faceoffwinningpctg"] = pd.to_numeric(df["home_faceoffwinningpctg"], errors='coerce').fillna(0)
        df["away_faceoffwinningpctg"] = pd.to_numeric(df["away_faceoffwinningpctg"], errors='coerce').fillna(0)
        df["home_sog"] = pd.to_numeric(df["home_sog"], errors='coerce').fillna(0)
        df["away_sog"] = pd.to_numeric(df["away_sog"], errors='coerce').fillna(0)
        df["home_blockedshots"] = pd.to_numeric(df["home_blockedshots"], errors='coerce').fillna(0)
        df["away_blockedshots"] = pd.to_numeric(df["away_blockedshots"], errors='coerce').fillna(0)
        df["home_pim"] = pd.to_numeric(df["home_pim"], errors='coerce').fillna(0)
        df["away_pim"] = pd.to_numeric(df["away_pim"], errors='coerce').fillna(0)
        df["home_goalies_saves"] = pd.to_numeric(df["home_goalies_saves"], errors='coerce').fillna(0)
        df["away_goalies_saves"] = pd.to_numeric(df["away_goalies_saves"], errors='coerce').fillna(0)

        logging.info("Вычисляем savepctg_diff")
        df["savepctg_diff"] = df["home_goalies_savepctg"] - df["away_goalies_savepctg"]
        logging.info("Вычисляем faceoff_diff")
        df["faceoff_diff"] = df["home_faceoffwinningpctg"] - df["away_faceoffwinningpctg"]
        logging.info("Вычисляем sog_diff")
        df["sog_diff"] = df["home_sog"] - df["away_sog"]
        logging.info("Вычисляем blockedshots_diff")
        df["blockedshots_diff"] = df["home_blockedshots"] - df["away_blockedshots"]
        logging.info("Вычисляем home_advantage_score")
        df["home_advantage_score"] = (
            df["savepctg_diff"] +
            df["faceoff_diff"] +
            df["sog_diff"] / 10 +
            df["blockedshots_diff"] / 10
        )
        logging.info("Вычисляем is_home_favored")
        df["is_home_favored"] = ((df["savepctg_diff"] > 0) & (df["faceoff_diff"] > 0)).astype(int)
        logging.info("Вычисляем goal_difference")
        df["goal_difference"] = df["home_score"] - df["away_score"]
        logging.info("Вычисляем lagged_goal_difference")
        df["lagged_goal_difference"] = df["goal_difference"].shift(1).fillna(0)
        logging.info("Вычисляем pim_diff")
        df["pim_diff"] = df["home_pim"] - df["away_pim"]
        logging.info("Вычисляем pim_total")
        df["pim_total"] = df["home_pim"] + df["away_pim"]
        logging.info("Вычисляем faceoff_save_interaction")
        df["faceoff_save_interaction"] = df["faceoff_diff"] * df["savepctg_diff"]
        logging.info("Вычисляем home_shooting_efficiency")
        df["home_shooting_efficiency"] = df["home_sog"] / (df["away_goalies_saves"] + 1e-6)
        logging.info("Вычисляем away_shooting_efficiency")
        df["away_shooting_efficiency"] = df["away_sog"] / (df["home_goalies_saves"] + 1e-6)
    except Exception as e:
        logging.info(f"Ошибка в add_home_advantage_features: {e}")
        raise
    return df

def coach_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки тренеров с векторизацией."""
    df = df.copy()

    # Вычисляем статистику для каждого тренера
    coach_stats = {}
    for coach in set(df['home_coach'].dropna()).union(set(df['away_coach'].dropna())):
        home_games = df[df['home_coach'] == coach]
        away_games = df[df['away_coach'] == coach]
        total_games = len(home_games) + len(away_games)
        if total_games > 0:
            home_wins = (home_games['home_score'] > home_games['away_score']).sum()
            away_wins = (away_games['away_score'] > away_games['home_score']).sum()
            coach_stats[coach] = {
                'avg_scored': (home_games['home_score'].sum() + away_games['away_score'].sum()) / total_games,
                'avg_conceded': (home_games['away_score'].sum() + away_games['home_score'].sum()) / total_games,
                'win_rate': (home_wins + away_wins) / total_games,
                'avg_sog': (home_games['home_sog'].sum() + away_games['away_sog'].sum()) / total_games,
                'avg_pim': (home_games['home_pim'].sum() + away_games['away_pim'].sum()) / total_games
            }

    # Вычисляем статистику для тренера по командам
    coach_team_stats = {}
    for coach in set(df['home_coach'].dropna()).union(set(df['away_coach'].dropna())):
        home_games = df[df['home_coach'] == coach]
        away_games = df[df['away_coach'] == coach]
        for team in set(home_games['home_team']).union(set(away_games['away_team'])):
            team_games = pd.concat([
                home_games[home_games['home_team'] == team],
                away_games[away_games['away_team'] == team]
            ])
            if len(team_games) > 0:
                coach_team_stats[(coach, team)] = {
                    'avg_scored': team_games.apply(
                        lambda x: x['home_score'] if x['home_coach'] == coach else x['away_score'], axis=1).mean()
                }

    # Векторизованное добавление признаков
    for prefix in ['home', 'away']:
        coach_col = f'{prefix}_coach'
        team_col = f'{prefix}_team'
        df[f'{prefix}_coach_avg_scored'] = df[coach_col].map(lambda x: coach_stats.get(x, {}).get('avg_scored', 0))
        df[f'{prefix}_coach_avg_conceded'] = df[coach_col].map(lambda x: coach_stats.get(x, {}).get('avg_conceded', 0))
        df[f'{prefix}_coach_win_rate'] = df[coach_col].map(lambda x: coach_stats.get(x, {}).get('win_rate', 0))
        df[f'{prefix}_coach_avg_sog'] = df[coach_col].map(lambda x: coach_stats.get(x, {}).get('avg_sog', 0))
        df[f'{prefix}_coach_avg_pim'] = df[coach_col].map(lambda x: coach_stats.get(x, {}).get('avg_pim', 0))
        if prefix == 'home':
            df['home_coach_team_avg_scored'] = df.apply(
                lambda row: coach_team_stats.get((row['home_coach'], row['home_team']), {}).get('avg_scored', 0),
                axis=1
            )

    # Векторизованное вычисление разниц
    df['coach_scored_diff'] = df['home_coach_avg_scored'] - df['away_coach_avg_scored']
    df['coach_conceded_diff'] = df['home_coach_avg_conceded'] - df['away_coach_avg_conceded']
    df['coach_win_rate_diff'] = df['home_coach_win_rate'] - df['away_coach_win_rate']
    df['coach_sog_diff'] = df['home_coach_avg_sog'] - df['away_coach_avg_sog']
    df['coach_pim_diff'] = df['home_coach_avg_pim'] - df['away_coach_avg_pim']
    df['coach_offense_power'] = df['home_coach_avg_scored'] * df['home_sog']
    df['coach_defense_strength'] = df['away_coach_avg_conceded'] * df['away_goalies_savepctg']

    # Заполнение пропусков
    coach_features = [col for col in df.columns if 'coach_' in col]
    df[coach_features] = df[coach_features].fillna(0)

    return df

def load_full_table(table_name: str = "nhl_games_extended") -> pd.DataFrame:
    """Загружает данные из Supabase, нормализует названия команд и добавляет базовые признаки."""
    TEAM_NAME_MAPPING = {
        'PHX': 'ARI', 'ATL': 'WPG', 'HFD': 'CAR', 'QUE': 'COL', 'WIN': 'WPG',
        'USA': None, 'RUS': None, 'SUI': None, 'CAN': None, 'CZE': None,
        'SWE': None, 'FIN': None, 'GER': None, 'SVK': None, 'LAT': None,
        'KLS': None, 'LIB': None, 'N.A': None, 'BLK': None, 'HBF': None,
        'MAN': None, 'AIK': None, 'ASE': None, 'ASW': None, 'AWA': None,
        'AUT': None, 'BLR': None, 'BRY': None, 'CMP': None, 'CSK': None,
        'TCH': None, 'DIN': None, 'DJU': None, 'MOS': None, 'RIG': None,
        'DDF': None, 'MUN': None, 'EIS': None, 'EUR': None, 'EVZ': None,
        'FAR': None, 'FEL': None, 'AFM': 'CGY', 'FRA': None, 'GOT': None,
        'DEU': None, 'BFT': None, 'GRA': None, 'KLH': None, 'DAV': None,
        'SLV': None, 'IFK': None, 'ITA': None, 'JPN': None, 'HEL': None,
        'KAR': None, 'KAZ': None, 'KHI': None, 'KLA': None, 'LAU': None,
        'LIN': None, 'MET': None, 'MOD': None, 'NAT': None, 'MNS': 'DAL',
        'NOR': None, 'PAC': None, 'PHT': None, 'MAL': None, 'CLR': 'NJD',
        'SCB': None, 'SKA': None, 'PET': None, 'SLO': None, 'SOK': None,
        'URS': None, 'SWI': None, 'SPR': None, 'SPA': None, 'ILV': None,
        'TAP': None, 'RED': None, 'MAT': None, 'MCD': None, 'ASR': None,
        'LNY': None, 'UKR': None, 'VFF': None, 'WLS': None, 'FRG': None,
        'YSW': None, 'ZUR': None, 'CGS': 'DAL', 'KCS': 'NJD', 'OAK': 'DAL',
        'TXS': None
    }

    engine = get_pg_engine()
    query = f"SELECT * FROM {table_name};"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            logging.warning(f"Таблица {table_name} пуста!")
            return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке таблицы {table_name}: {str(e)}")
        raise

        # Преобразование типов и создание 'total_goals'
        df["home_score"] = pd.to_numeric(df["home_score"], errors='coerce').fillna(0)
        df["away_score"] = pd.to_numeric(df["away_score"], errors='coerce').fillna(0)
        df["total_goals"] = df["home_score"] + df["away_score"]

        # Проверка создания столбца
        if "total_goals" not in df.columns or df["total_goals"].isnull().all():
            logging.error(
                "Столбец 'total_goals' не был создан корректно. Проверьте данные в 'home_score' и 'away_score'.")
            raise ValueError("Ошибка при создании столбца 'total_goals'")

    for col in ["referee_1", "referee_2", "linesman_1", "linesman_2"]:
        if col not in df.columns:
            df[col] = "Unknown"

    for col in ["home_team", "away_team", "home_abbrev", "away_abbrev"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAPPING)

    valid_teams = set()
    for division in TEAM_STRUCTURE.values():
        valid_teams.update(division["teams"])

    mask = (
        df["home_team"].isin(valid_teams) &
        df["away_team"].isin(valid_teams) &
        df["home_team"].notna() &
        df["away_team"].notna()
    )
    df = df[mask].copy()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors='coerce')
        df["day_name"] = df["game_date"].dt.day_name()
        df["match_month"] = df["game_date"].dt.month_name()

    df["total_goals"] = df["home_score"] + df["away_score"]
    df["home_shifts_sum"] = df["home_forwards_shifts"] + df["home_defense_shifts"]
    df["away_shifts_sum"] = df["away_forwards_shifts"] + df["away_defense_shifts"]
    df["home_giveaways_sum"] = df["home_forwards_giveaways"] + df["home_defense_giveaways"]
    df["away_giveaways_sum"] = df["away_forwards_giveaways"] + df["away_defense_giveaways"]
    df["home_takeaways_sum"] = df["home_forwards_takeaways"] + df["home_defense_takeaways"]
    df["away_takeaways_sum"] = df["away_forwards_takeaways"] + df["away_defense_takeaways"]
    df["game_duration"] = pd.to_numeric(df["game_duration"], errors='coerce').fillna(60.0)
    df["hits_per_minute"] = (df["home_hits"] + df["away_hits"]) / df["game_duration"]
    df["sog_per_minute"] = (df["home_sog"] + df["away_sog"]) / df["game_duration"]
    df["pim_per_minute"] = (df["home_pim"] + df["away_pim"]) / df["game_duration"]
    df["shifts_per_minute"] = (df["home_shifts_sum"] + df["away_shifts_sum"]) / df["game_duration"]

    return df

def split_features(df: pd.DataFrame):
    """Разделяет признаки на числовые и категориальные."""
    exclude = {"total_goals", "game_date", "home_score", "away_score"}
    feat_cols = [c for c in df.columns if c not in exclude]
    cat_cols = [c for c in feat_cols if df[c].dtype == "object" or c in ["referee_1", "referee_2", "linesman_1", "linesman_2"]]
    return feat_cols, cat_cols

def select_features(df, target_col, numerical_cols, k=20, vif_threshold=10):
    """
    Отбирает числовые признаки с высокой корреляцией с целевой переменной и низкой мультиколлинеарностью.

    Args:
        df (pd. DataFrame): Датасет с признаками и целевой переменной.
        target_col (str): Название столбца с целевой переменной ('total_goals').
        numerical_cols (list): Список числовых признаков для анализа.
        k (int): Количество признаков с максимальной корреляцией для начального отбора.
        vif_threshold (float): Порог VIF для исключения признаков с высокой мультиколлинеарностью.

    Returns:
        list: Список отобранных числовых признаков.
    """
    # Удаляем строки с пропусками в целевой переменной
    df_clean = df.dropna(subset=[target_col]).copy()

    # 1. Рассчитываем корреляцию с целевой переменной и выбираем топ-k признаков
    selector = SelectKBest(score_func=f_regression, k=min(k, len(numerical_cols)))
    selector.fit(df_clean[numerical_cols], df_clean[target_col])
    scores = pd.Series(selector.scores_, index=numerical_cols)
    top_k_features = scores.nlargest(k).index.tolist()

    # 2. Оцениваем мультиколлинеарность с помощью VIF
    selected_features = []
    vif_data = df_clean[top_k_features].dropna()  # Удаляем строки с пропусками для VIF
    if vif_data.empty or len(vif_data) < 2:
        return top_k_features  # Если данных недостаточно, возвращаем топ-k без VIF

    for feature in top_k_features:
        vif_features = selected_features + [feature]
        vif_df = vif_data[vif_features]
        if vif_df.shape[1] > 1:  # VIF требует минимум 2 признака
            vif = variance_inflation_factor(vif_df.values, vif_df.columns.get_loc(feature))
            if vif < vif_threshold:
                selected_features.append(feature)
        else:
            selected_features.append(feature)

    return selected_features

def select_models_based_on_data_size(n_matches, h2h_df=None, home_team=None, away_team=None):
    """Выбирает модели и их параметры в зависимости от размера данных."""
    if n_matches < 10 and h2h_df is not None and home_team is not None and away_team is not None:
        home_last_home = h2h_df[h2h_df['home_team'] == home_team].sort_values('game_date', ascending=False).head(10)
        home_last_away = h2h_df[h2h_df['away_team'] == home_team].sort_values('game_date', ascending=False).head(10)
        away_last_home = h2h_df[h2h_df['home_team'] == away_team].sort_values('game_date', ascending=False).head(10)
        away_last_away = h2h_df[h2h_df['away_team'] == away_team].sort_values('game_date', ascending=False).head(10)

        home_home_avg = home_last_home['total_goals'].mean() if not home_last_home.empty else 0
        home_away_avg = home_last_away['total_goals'].mean() if not home_last_away.empty else 0
        away_home_avg = away_last_home['total_goals'].mean() if not away_last_home.empty else 0
        away_away_avg = away_last_away['total_goals'].mean() if not away_last_away.empty else 0

        home_avg = (home_home_avg * 0.6 + home_away_avg * 0.4) if (home_home_avg and home_away_avg) else (
                    home_home_avg or home_away_avg or 0)
        away_avg = (away_away_avg * 0.6 + away_home_avg * 0.4) if (away_away_avg and away_home_avg) else (
                    away_away_avg or away_home_avg or 0)

        simplified_prediction = (home_avg * 0.5 + away_avg * 0.5)
        return None, None, None, None, simplified_prediction

    if n_matches < 30:
        base_models = [
            ('rf', RandomForestRegressor(max_depth=5, n_estimators=100, min_samples_split=5, random_state=RANDOM_SEED)),
            ('et', ExtraTreesRegressor(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=RANDOM_SEED))
        ]
        meta_model = Ridge(alpha=0.5, random_state=RANDOM_SEED)
        param_grid = {
            'model__rf__max_depth': [3, 5, 7],
            'model__rf__min_samples_split': [2, 5, 10],
            'model__et__max_depth': [3, 5, 7],
            'model__et__min_samples_leaf': [1, 3, 5]
        }
        arima_params = {'p_values': [0, 1, 2, 3], 'q_values': [0, 1, 2, 3], 'd': 0}
    elif 30 <= n_matches < 100:
        base_models = [
            ('ada', AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=RANDOM_SEED)),
            ('cat', CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, l2_leaf_reg=3, random_seed=RANDOM_SEED, verbose=0)),
            ('lgbm', LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, num_leaves=31, random_state=RANDOM_SEED, verbose=-1)),
            ('hgb', HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, random_state=RANDOM_SEED)),
            ('xgb', XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, gamma=0.1, random_state=RANDOM_SEED)),
            ('gbr', GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, min_samples_split=5, random_state=RANDOM_SEED))
        ]
        meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED)
        param_grid = {
            'model__ada__n_estimators': [50, 100, 150],
            'model__ada__learning_rate': [0.01, 0.05, 0.1],
            'model__cat__iterations': [100, 200, 300],
            'model__cat__learning_rate': [0.03, 0.05, 0.07],
            'model__lgbm__n_estimators': [100, 150, 200],
            'model__lgbm__num_leaves': [15, 31, 63],
            'model__hgb__max_iter': [100, 150, 200],
            'model__hgb__learning_rate': [0.03, 0.05, 0.07],
            'model__xgb__n_estimators': [100, 150, 200],
            'model__xgb__gamma': [0, 0.1, 0.2],
            'model__gbr__max_depth': [3, 5, 7],
            'model__final_estimator__alpha': [0.01, 0.1, 1.0],
            'model__final_estimator__l1_ratio': [0.3, 0.5, 0.7]
        }
        arima_params = None
    else:
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=200, max_features='sqrt', min_samples_leaf=3, random_state=RANDOM_SEED)),
            ('ada', AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_SEED)),
            ('cat', CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05, l2_leaf_reg=5, random_seed=RANDOM_SEED, verbose=0)),
            ('lgbm', LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.03, feature_fraction=0.8, random_state=RANDOM_SEED, verbose=-1)),
            ('hgb', HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=63, random_state=RANDOM_SEED)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_features=0.7, min_samples_leaf=3, random_state=RANDOM_SEED)),
            ('svr', SVR(kernel='rbf', C=1.5, epsilon=0.05, gamma='scale')),
            ('gbr', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_split=10, random_state=RANDOM_SEED)),
            ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance', p=1)),
            ('xgb', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, gamma=0.2, subsample=0.8, random_state=RANDOM_SEED)),
            ('rls', Ridge(alpha=1.0, random_state=RANDOM_SEED))
        ]
        meta_model = MLPRegressor(hidden_layer_sizes=(200, 150, 100), activation='relu', solver='adam', alpha=0.001,
                                  learning_rate='adaptive', max_iter=1000, random_state=RANDOM_SEED, early_stopping=False)
        param_grid = {
            'model__rf__n_estimators': [100, 200, 300],
            'model__rf__max_features': ['sqrt', 0.7, 0.8],
            'model__cat__iterations': [200, 300, 400],
            'model__cat__depth': [6, 8, 10],
            'model__lgbm__n_estimators': [200, 300, 400],
            'model__lgbm__num_leaves': [31, 63, 127],
            'model__gbr__max_depth': [3, 5, 7],
            'model__knn__n_neighbors': [5, 7, 9],
            'model__xgb__n_estimators': [200, 300, 400],
            'model__xgb__gamma': [0.1, 0.2, 0.3],
            'model__rls__alpha': [0.1, 1.0, 10.0],
            'model__final_estimator__hidden_layer_sizes': [(100,), (150, 100), (200, 150, 100)],
            'model__final_estimator__learning_rate': ['constant', 'adaptive'],
            'model__final_estimator__alpha': [0.0001, 0.001, 0.01]
        }
        arima_params = None

    return base_models, meta_model, param_grid, arima_params, None

def train_evaluate(h2h_df: pd.DataFrame, home_team: str, away_team: str, choice: str):
    feat_cols, cat_cols = split_features(h2h_df)

    # Добавляем фильтрацию признаков
    use_feature_filter = 0
    if use_feature_filter == 1:
        numerical_cols = [col for col in feat_cols if col not in cat_cols]
        selected_numerical_cols = select_features(h2h_df, 'total_goals', numerical_cols, k=10, vif_threshold=10)
        feat_cols = selected_numerical_cols + cat_cols
        logging.info(f"Отобрано числовых признаков: {len(selected_numerical_cols)}")
        logging.info(f"Отобранные числовые признаки: {selected_numerical_cols}")
        logging.info(f"Общее количество используемых признаков: {len(feat_cols)}")
    else:
        logging.info("Отбор признаков отключен")

    # Добавим важные категориальные признаки
    mandatory_categoricals = ["venue", "home_team", "away_team", "referee_1", "referee_2", "linesman_1", "linesman_2"]
    for col in mandatory_categoricals:
        if col in feat_cols and col not in cat_cols:
            cat_cols.append(col)

    # Удалим лишние столбцы для обучения
    columns_to_exclude = ['game_id', 'period_type', 'home_abbrev', 'home_name', 'away_abbrev', 'away_name',
                          'away_city', 'home_city', 'game_type', 'neutral_site', 'home_team', 'away_team',
                          'game_duration', 'home_id', 'away_id', 'total_goals_trend', 'home_forwards_goals',
                          'away_forwards_goals', 'home_defense_goals', 'away_defense_goals',
                          'away_defense_assists', 'away_forwards_assists', 'home_defense_assists',
                          'home_forwards_assists']
    feat_cols = [col for col in feat_cols if col not in columns_to_exclude]
    cat_cols = [col for col in cat_cols if col not in columns_to_exclude]

    data = h2h_df.dropna(subset=feat_cols + ["total_goals"]).copy()
    data = data.sort_values("game_date").reset_index(drop=True)

    n_matches = len(data)
    logging.info(f"Количество очных матчей для обучения: {n_matches}")

    base_models, _, _, _, simplified_pred = select_models_based_on_data_size(n_matches, h2h_df, home_team, away_team)

    if simplified_pred is not None:
        logging.info("⚠️ Мало очных матчей (<10), используем упрощенный метод прогнозирования")
        logging.info(f"Прогноз тотала: {simplified_pred:.2f}")
        return None, feat_cols, cat_cols, None, None, None, simplified_pred, None, None, None, None

    # Разделение на train/test
    train_ratio = 0.9 if n_matches < 30 else 0.85 if n_matches < 100 else 0.8
    split_idx = int(len(data) * train_ratio) if len(data) > 5 else len(data) - 1
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    logging.info(f"Разделение данных: {train_ratio * 100:.0f}% train / {(1 - train_ratio) * 100:.0f}% test")
    logging.info(
        f"Диапазон дат обучающей выборки: {train['game_date'].min().date()} — {train['game_date'].max().date()}")
    logging.info(f"Диапазон дат тестовой выборки: {test['game_date'].min().date()} — {test['game_date'].max().date()}")

    X_train, y_train = train[feat_cols], train["total_goals"]
    X_test, y_test = test[feat_cols], test["total_goals"]
    X_all, y_all = data[feat_cols], data["total_goals"]

    num_cols = [f for f in feat_cols if f not in cat_cols]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ], n_jobs=-1)

    # Выбор модели вручную
    if choice == "1":
        logging.info("Доступные модели:")
        for i, (name, _) in enumerate(base_models, 1):
            logging.info(f"{i}. {name}")
        model_choice = int(input("Выберите номер модели: ").strip()) - 1
        if model_choice < 0 or model_choice >= len(base_models):
            raise ValueError("Некорректный выбор модели")
        model_name, model_instance = base_models[model_choice]
        logging.info(f"Выбрана модель: {model_name}")

        if model_name == "cat":
            cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
            model_instance.fit(X_train, y_train, cat_features=cat_feature_indices)
            preds_train = np.round(model_instance.predict(X_train))
            preds_test = np.round(model_instance.predict(X_test))
            mae_test = mean_absolute_error(y_test, preds_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
            logging.info(f"MAE на тестовой выборке: {mae_test:.2f}, RMSE: {rmse_test:.2f}")
            cat_feature_indices_all = [X_all.columns.get_loc(c) for c in cat_cols if c in X_all.columns]
            model_instance.fit(X_all, y_all, cat_features=cat_feature_indices_all)
            return model_instance, feat_cols, cat_cols, mae_test, rmse_test, split_idx, None, preds_train, preds_test, None, (X_test, y_test)

        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_instance)
            ])
            pipeline.fit(X_train, y_train)
            preds_train = np.round(pipeline.predict(X_train))
            preds_test = np.round(pipeline.predict(X_test))
            mae_test = mean_absolute_error(y_test, preds_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
            logging.info(f"MAE на тестовой выборке: {mae_test:.2f}, RMSE: {rmse_test:.2f}")
            pipeline.fit(X_all, y_all)
            return pipeline, feat_cols, cat_cols, mae_test, rmse_test, split_idx, None, preds_train, preds_test, None, (X_test, y_test)

    # Автоподбор лучшей модели
    elif choice == "2":
        best_rmse = float('inf')
        best_model = None
        best_model_name = None
        best_preds_train = None
        best_preds_test = None

        for name, model_instance in base_models:
            logging.info(f"Пробуем модель: {name}")

            if name == "cat":
                cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
                model_instance.fit(X_train, y_train, cat_features=cat_feature_indices)
                preds_test = np.round(model_instance.predict(X_test))
                preds_train = np.round(model_instance.predict(X_train))
                rmse = np.sqrt(mean_squared_error(y_test, preds_test))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_instance
                    best_model_name = name
                    best_preds_test = preds_test
                    best_preds_train = preds_train
            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model_instance)
                ])
                pipeline.fit(X_train, y_train)
                preds_test = np.round(pipeline.predict(X_test))
                preds_train = np.round(pipeline.predict(X_train))
                rmse = np.sqrt(mean_squared_error(y_test, preds_test))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = pipeline
                    best_model_name = name
                    best_preds_test = preds_test
                    best_preds_train = preds_train

        logging.info(f"Оптимальная модель: {best_model_name} с RMSE: {best_rmse:.2f}")

        # Финальное переобучение
        if best_model_name == "cat":
            cat_feature_indices_all = [X_all.columns.get_loc(c) for c in cat_cols if c in X_all.columns]
            best_model.fit(X_all, y_all, cat_features=cat_feature_indices_all)
        else:
            best_model.fit(X_all, y_all)

        mae_test = mean_absolute_error(y_test, best_preds_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, best_preds_test))

        return best_model, feat_cols, cat_cols, mae_test, rmse_test, split_idx, None, best_preds_train, best_preds_test, None, (X_test, y_test)

    else:
        raise ValueError("Некорректный выбор режима обучения")

def train_stacking_evaluate(h2h_df: pd.DataFrame, home_team: str, away_team: str):
    """Обучает Stacking ансамбль с выбором моделей по размеру данных."""
    feat_cols, cat_cols = split_features(h2h_df)

    # Убедимся, что 'venue' и другие ключевые категориальные признаки есть в cat_cols
    mandatory_categoricals = ["venue", "home_team", "away_team", "referee_1", "referee_2", "linesman_1", "linesman_2"]
    for col in mandatory_categoricals:
        if col in feat_cols and col not in cat_cols:
            cat_cols.append(col)

    # Список столбцов для исключения перед обучением
    columns_to_exclude = ['game_id', 'period_type', 'home_abbrev', 'home_name', 'away_abbrev', 'away_name',
                          'away_city', 'home_city', 'game_type', 'neutral_site', 'home_team', 'away_team',
                          'game_duration', 'home_id', 'away_id', 'total_goals_trend', 'home_forwards_goals',
                          'away_forwards_goals', 'home_defense_goals', 'away_defense_goals',
                          'away_defense_assists', 'away_forwards_assists', 'home_defense_assists',
                          'home_forwards_assists']
    feat_cols = [col for col in feat_cols if col not in columns_to_exclude]
    cat_cols = [col for col in cat_cols if col not in columns_to_exclude]

    # Добавляем фильтрацию признаков
    use_feature_filter = 0
    if use_feature_filter == 1:
        numerical_cols = [col for col in feat_cols if col not in cat_cols]
        selected_numerical_cols = select_features(h2h_df, 'total_goals', numerical_cols, k=10, vif_threshold=10)
        feat_cols = selected_numerical_cols + cat_cols
        logging.info(f"Отобрано числовых признаков: {len(selected_numerical_cols)}")
        logging.info(f"Отобранные числовые признаки: {selected_numerical_cols}")
        logging.info(f"Общее количество используемых признаков: {len(feat_cols)}")
    else:
        logging.info("Отбор признаков отключен")

    ts_features = [col for col in h2h_df.columns if col.startswith('ts_total_goals_')]
    valid_mask = h2h_df[ts_features].notna().all(axis=1) if ts_features else pd.Series(True, index=h2h_df.index)

    data = h2h_df[valid_mask].dropna(subset=feat_cols + ["total_goals"]).copy()
    data = data.sort_values("game_date").reset_index(drop=True)

    logging.info("Таблица признаков для прогноза:")
    logging.info("───────────────────────────────")
    recent_matches = data.sort_values("game_date", ascending=False)
    table_data = recent_matches[feat_cols + ["game_date"]].copy()
    table_data["game_date"] = table_data["game_date"].dt.strftime("%d.%m.%y")
    table_transposed = table_data.set_index("game_date").T
    table_transposed = table_transposed.sort_index()
    max_width = 20
    logging.info(table_transposed.to_string(justify="left", col_space=max_width))
    logging.info("───────────────────────────────")
    logging.info(f"Общее количество используемых признаков: {len(feat_cols)}")

    n_matches = len(data)
    logging.info(f"Количество очных матчей для обучения: {n_matches}")

    base_models, meta_model, param_grid, arima_params, simplified_pred = select_models_based_on_data_size(
        n_matches, h2h_df, home_team, away_team)

    if simplified_pred is not None:
        logging.info("⚠️ Мало очных матчей (<10), используем упрощенный метод прогнозирования")
        logging.info(f"Прогноз тотала: {simplified_pred:.2f}")
        return None, feat_cols, cat_cols, None, None, None, simplified_pred, None, None, None

    if n_matches < 30:
        train_ratio = 0.9
    elif 30 <= n_matches < 100:
        train_ratio = 0.85
    else:
        train_ratio = 0.8

    split_idx = int(len(data) * train_ratio) if len(data) > 5 else len(data) - 1
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    logging.info(f"Разделение данных: {train_ratio * 100:.0f}% train / {(1 - train_ratio) * 100:.0f}% test")
    logging.info(
        f"Диапазон дат обучающей выборки: {train['game_date'].min().date()} — {train['game_date'].max().date()}")
    logging.info(f"Диапазон дат тестовой выборки: {test['game_date'].min().date()} — {test['game_date'].max().date()}")

    logging.info(f"Запускаем обучение ансамбля...")
    X_train, y_train = train[feat_cols], train["total_goals"]
    X_test, y_test = test[feat_cols], test["total_goals"]

    num_cols = [f for f in feat_cols if f not in cat_cols]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ], n_jobs=-1)

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=True,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stacking)
    ])

    # Кросс-валидация и обучение для оценки
    logging.info(f"Валидация и оптимизация моделей...")
    n_splits = min(5, len(X_train) - 1)
    if n_splits >= 2:
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            scoring='neg_mean_absolute_error',
            cv=TimeSeriesSplit(n_splits=n_splits),
            n_jobs=-1,
            error_score='raise',
            random_state=RANDOM_SEED
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    else:
        logging.warning("Слишком мало данных для кросс-валидации. Обучение на всех данных.")
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    arima_pred = None
    if n_matches < 30 and arima_params:
        try:
            best_aic = float('inf')
            best_order = None
            best_fit = None
            for p in arima_params['p_values']:
                for q in arima_params['q_values']:
                    try:
                        model = ARIMA(y_train, order=(p, arima_params['d'], q))
                        fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, arima_params['d'], q)
                            best_fit = fit
                    except:
                        continue
            if best_order:
                arima_pred = best_fit.forecast(steps=len(y_test))
        except Exception as e:
            logging.info(f"Ошибка ARIMA: {e}")

    logging.info(f"Обучение успешно выполнено!")
    logging.info(f"Оценка точности ансамбля...")

    mae_train = None
    mae_test = None
    rmse_test = None

    # Сохраняем прогнозы до переобучения
    if not test.empty:
        preds_test = np.round(best_model.predict(X_test))
        if arima_pred is not None:
            preds_test = np.round(0.5 * preds_test + 0.5 * arima_pred)

        # Метрики для тестовой выборки
        mae_test = mean_absolute_error(y_test, preds_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))

        # Метрики для тренировочной выборки
        preds_train = np.round(best_model.predict(X_train))
        mae_train = mean_absolute_error(y_train, preds_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))

        # Вывод метрик
        logging.info(f"MAE на тренировочной выборке: {mae_train:.2f}, RMSE: {rmse_train:.2f}")
        logging.info(f"MAE на тестовой выборке: {mae_test:.2f}, RMSE: {rmse_test:.2f}")
    else:
        logging.warning("Тестовая выборка пуста, метрики не рассчитаны")
        preds_test = None
        preds_train = None

    # Новый шаг: обучение на всех данных для прогноза
    logging.info("Переобучаем модели на всех данных для построения вневыборочного прогноза...")
    X_all, y_all = data[feat_cols], data["total_goals"]
    best_model_forecast = clone(pipeline)
    best_model_forecast.fit(X_all, y_all)
    logging.info("Модели переобучены на всех данных для построения вневыборочного прогноза!")

    # Возвращаем test_data как 11-й параметр
    test_data = (X_test, y_test) if not test.empty else None

    return best_model_forecast, feat_cols, cat_cols, mae_test, rmse_test, split_idx, None, preds_train, preds_test, arima_pred, test_data

def get_user_input_for_categorical(cat_cols: list, h2h: pd.DataFrame, home: str, away: str) -> dict:
    """Запрашивает у пользователя дату матча и определяет категориальные признаки (day_name, match_month, tournament_stage),
    автоматически заполняя conference, division и venue."""
    user_inputs = {}
    team_to_conf_div = {}
    for division, info in TEAM_STRUCTURE.items():
        for team in info["teams"]:
            team_to_conf_div[team] = {"conference": info["conference"], "division": division}

    # Запрашиваем дату матча
    logging.info("Введите предварительную дату матча (формат: ДД.ММ.ГГ, например, 17.03.22):")
    date_input = input().strip()
    try:
        match_date = pd.to_datetime(date_input, format="%d.%m.%y")
        user_inputs["game_date"] = match_date
        user_inputs["day_name"] = match_date.day_name()
        user_inputs["match_month"] = match_date.month_name()
        # Определяем tournament_stage на основе функции add_tournament_stage
        sub = h2h[((h2h["home_team"] == home) & (h2h["away_team"] == away)) |
                  ((h2h["home_team"] == away) & (h2h["away_team"] == home))].sort_values("game_date")
        pair_key = (home, away)
        regular_season = sub[sub['game_type'] == 2]
        team_pairs = regular_season.groupby(['home_team', 'away_team'])['game_date'].agg(['min', 'max'])

        if pair_key in team_pairs.index:
            first_date = team_pairs.loc[pair_key, 'min']
            last_date = team_pairs.loc[pair_key, 'max']
            if sub['game_type'].iloc[-1] == 1:
                user_inputs["tournament_stage"] = 'preparation'
            elif sub['game_type'].iloc[-1] == 3:
                user_inputs["tournament_stage"] = 'champion_stage'
            elif match_date.month >= 9 and match_date.month <= 11:
                user_inputs["tournament_stage"] = 'early_stage'
            elif match_date.month == 12 or match_date.month <= 2:
                user_inputs["tournament_stage"] = 'mid_stage'
            else:
                user_inputs["tournament_stage"] = 'late_stage'
        else:
            user_inputs["tournament_stage"] = 'Unknown'
    except ValueError:
        logging.error("Некорректный формат даты, используются значения по умолчанию.")
        user_inputs["day_name"] = h2h["day_name"].mode()[0] if not h2h["day_name"].mode().empty else "Saturday"
        user_inputs["match_month"] = h2h["match_month"].mode()[0] if not h2h["match_month"].mode().empty else "November"
        user_inputs["tournament_stage"] = h2h["tournament_stage"].mode()[0] if not h2h[
            "tournament_stage"].mode().empty else "early_stage"

    for col in cat_cols:
        if col in ["home_team", "away_team"]:
            user_inputs[col] = home if col == "home_team" else away
        elif col in ["home_conference", "away_conference", "home_division", "away_division"]:
            team = home if col.startswith("home") else away
            if col.endswith("conference"):
                user_inputs[col] = team_to_conf_div.get(team, {}).get("conference", "Unknown")
            else:
                user_inputs[col] = team_to_conf_div.get(team, {}).get("division", "Unknown")
        elif col == "venue":
            user_inputs[col] = TEAM_VENUE_MAPPING.get(home, "Unknown")
        elif col not in ["day_name", "match_month", "tournament_stage"]:
            unique_values = h2h[col].dropna().unique().tolist()
            default_value = h2h[col].mode()[0] if not h2h[col].mode().empty else "Unknown"
            logging.info(
                f"Введите значение для {col} (возможные варианты: {', '.join(map(str, unique_values))}, по умолчанию: {default_value}):")
            value = input().strip()
            user_inputs[col] = value or default_value

    return user_inputs

def build_next_row(h2h: pd.DataFrame, home: str, away: str, feat_cols: list, cat_cols: list, df: pd.DataFrame, client: NHLClient, manual_numerical_cols: list = None, match_date: str = None, game_order=None) -> pd.DataFrame:
    """Создаёт следующую строку признаков с учётом данных из расписания NHL или пользовательского ввода."""
    # Список признаков, которые берутся из последнего актуального матча
    LAST_MATCH_FEATURES = [
        'away_winrate', 'home_winrate', 'home_team_goals_range_last',
        'away_team_goals_range_last', 'combined_goals_range',
        'away_div_position', 'away_conf_position', 'away_league_position',
        'home_div_position', 'home_conf_position', 'home_league_position',
        'lagged_goal_difference'
    ]

    # Список признаков, которые не нужно округлять
    NO_ROUND_FEATURES = [
        'home_forwards_avg_toi', 'home_defense_avg_toi', 'home_forwards_total_toi',
        'home_defense_total_toi', 'home_skaters_total_toi', 'home_goalies_savepctg',
        'away_forwards_avg_toi', 'away_defense_avg_toi', 'away_forwards_total_toi',
        'away_defense_total_toi', 'away_skaters_total_toi', 'away_goalies_savepctg',
        'home_puck_control_total', 'away_puck_control_total', 'home_faceoffwinningpctg',
        'away_faceoffwinningpctg', 'home_powerplaypctg', 'away_powerplaypctg',
        'goals_per_minute', 'hits_per_minute', 'sog_per_minute', 'pim_per_minute',
        'shifts_per_minute', 'savepctg_diff', 'faceoff_diff', 'home_advantage_score',
        'faceoff_save_interaction', 'home_shooting_efficiency', 'away_shooting_efficiency',
        'home_travel_distance', 'away_travel_distance', 'home_travel_fatigue',
        'away_travel_fatigue', 'home_coach_avg_scored', 'home_coach_avg_conceded',
        'away_coach_avg_scored', 'home_coach_avg_conceded', 'away_coach_avg_conceded',
        'home_coach_win_rate', 'away_coach_win_rate', 'home_coach_avg_pim',
        'away_coach_avg_pim', 'away_coach_avg_sog', 'away_coach_avg_sog', 'coach_scored_diff',
        'coach_conceded_diff', 'coach_win_rate_diff', 'coach_sog_diff', 'coach_pim_diff',
        'coach_offense_power', 'coach_defense_strength', 'away_readiness', 'home_readiness',
        'readiness_diff', 'home_cf_pct', 'away_cf_pct', 'home_ff_pct', 'away_ff_pct',
        'total_goals_trend', 'home_coach_avg_sog', 'home_coach_team_avg_scored', 'home_arena_avg_temp'
    ]

    if manual_numerical_cols is None:
        logging.info(f"Список числовых фичей не задан, используем автоподбор...")
        manual_numerical_cols = ['away_scratch_count', 'home_scratch_count']

    sub = h2h[((h2h["home_team"] == home) & (h2h["away_team"] == away)) |
              ((h2h["home_team"] == away) & (h2h["away_team"] == home))].sort_values("game_date")
    if sub.empty:
        raise ValueError("Нет данных для команд.")

    last_n = sub.tail(10)
    last_match = sub.tail(1)
    next_row = {}

    # Попытка получить данные из расписания NHL
    try:
        logging.info(f"Попытка получить данные из расписания NHL для {home} vs {away}")
        # Используем переданную дату, если она есть, или запрашиваем у пользователя
        if match_date is None:
            logging.info("Введите дату матча (формат: ДД.ММ.ГГ, например, 17.03.22):")
            date_input = input().strip()
            match_date = pd.to_datetime(date_input, format="%d.%m.%y")
        else:
            match_date = pd.to_datetime(match_date, format="%Y-%m-%d")
        date_str = match_date.strftime("%Y-%m-%d")

        # Добавляем температуру для следующего матча
        if home in TEAM_CITY_COORDS:
            lat, lon = TEAM_CITY_COORDS[home]
            if match_date is not None:
                match_date = pd.to_datetime(match_date, format="%Y-%m-%d")
                avg_temp = get_historical_avg_temperature(lat, lon, match_date.month, match_date.day)
                next_row['home_arena_avg_temp'] = avg_temp if avg_temp is not None else h2h[
                    'home_arena_avg_temp'].mean()
            else:
                next_row['home_arena_avg_temp'] = h2h['home_arena_avg_temp'].mean()
        else:
            next_row['home_arena_avg_temp'] = h2h['home_arena_avg_temp'].mean()

        # Получаем расписание на указанную дату
        try:
            schedule = client.schedule.get_schedule(date=date_str)
            logging.info(f"Получено расписание: {len(schedule.get('games', []))} матчей")
        except Exception as e:
            logging.error(f"Ошибка API NHL: {str(e)}")
            schedule = {'games': []}

        game_id = None
        selected_game = None
        for i, game in enumerate(schedule.get('games', []), 1):
            home_team_abbrev = game.get('homeTeam', {}).get('abbrev', 'N/A')
            away_team_abbrev = game.get('awayTeam', {}).get('abbrev', 'N/A')
            start_time_utc = game.get('startTimeUTC', '')
            game_type = game.get('gameType', None)
            logging.info(
                f"Матч {i}: {home_team_abbrev} vs {away_team_abbrev}, game_type: {game_type}, start_time: {start_time_utc}"
            )

            if start_time_utc:
                game_date = pd.to_datetime(start_time_utc).date()
                if (home_team_abbrev == home and away_team_abbrev == away and
                        abs((game_date - match_date.date()).days) <= 1):
                    selected_game = game
                    game_id = str(game.get('id', 'N/A'))
                    break

        if game_id and selected_game:
            logging.info(f"Найден матч с ID {game_id}")
            game_info = collect_game_info(game_id, client)
            match_date = pd.to_datetime(game_info.get('game_date', None))
            if not match_date:
                raise ValueError("Дата матча не найдена в данных NHL")
            logging.info(f"Дата матча автоматически получена: {match_date.strftime('%d.%m.%y')}")
            user_inputs = {
                'game_date': match_date,
                'day_name': match_date.day_name(),
                'match_month': match_date.month_name(),
                'home_team': home,
                'away_team': away,
                'venue': game_info.get('venue', 'Unknown'),
                'home_coach': game_info.get('home_coach', 'Unknown'),
                'away_coach': game_info.get('away_coach', 'Unknown'),
                'referee_1': game_info.get('referee_1', 'Unknown'),
                'referee_2': game_info.get('referee_2', 'Unknown'),
                'linesman_1': game_info.get('linesman_1', 'Unknown'),
                'linesman_2': game_info.get('linesman_2', 'Unknown'),
                'home_scratch_count': game_info.get('home_scratch_count', 0),
                'away_scratch_count': game_info.get('away_scratch_count', 0),
                'home_total_players': game_info.get('home_total_players', 0),
                'away_total_players': game_info.get('away_total_players', 0),
            }
        else:
            logging.warning(f"Матч {home} vs {away} на {date_str} не найден, используем пользовательский ввод.")
            user_inputs = get_user_input_for_categorical(cat_cols, h2h, home, away)
            user_inputs['game_date'] = match_date
            user_inputs['home_team'] = home
            user_inputs['away_team'] = away
            user_inputs['venue'] = TEAM_VENUE_MAPPING.get(home, 'Unknown')
            user_inputs['home_coach'] = last_match['home_coach'].iloc[0] if not last_match[
                'home_coach'].empty else 'Unknown'
            user_inputs['away_coach'] = last_match['away_coach'].iloc[0] if not last_match[
                'away_coach'].empty else 'Unknown'

    except Exception as e:
        logging.error(f"Ошибка при получении данных из расписания: {str(e)}")
        user_inputs = get_user_input_for_categorical(cat_cols, h2h, home, away)
        user_inputs['game_date'] = match_date
        user_inputs['home_team'] = home
        user_inputs['away_team'] = away
        user_inputs['venue'] = TEAM_VENUE_MAPPING.get(home, 'Unknown')
        user_inputs['home_coach'] = last_match['home_coach'].iloc[0] if not last_match[
            'home_coach'].empty else 'Unknown'
        user_inputs['away_coach'] = last_match['away_coach'].iloc[0] if not last_match[
            'away_coach'].empty else 'Unknown'

    # Определяем, первый ли это матч в сезоне
    season = match_date.year if match_date.month >= 9 else match_date.year - 1
    season_start = pd.to_datetime(f"{season}-09-01")
    team_games = h2h[(h2h['home_team'].isin([home, away]) | h2h['away_team'].isin([home, away])) &
                     (h2h['game_date'] >= season_start) & (h2h['game_date'] < match_date)]
    is_first_match = team_games.empty

    # Определяем числовые признаки
    numerical_cols = [col for col in feat_cols if col not in cat_cols]

    # Заполняем признаки
    #logging.info(f"Столбцы в h2h: {list(h2h.columns)}")
    #if 'referee_1' in h2h.columns:
        #logging.info(f"Тип данных referee_1 в h2h: {h2h['referee_1'].dtype}")
        #logging.info(f"Уникальные значения referee_1: {h2h['referee_1'].unique()}")
    for col in feat_cols:
        if col in user_inputs:
            next_row[col] = user_inputs[col]
        elif col in cat_cols:
            default_value = last_n[col].mode().iloc[0] if (
                        col in last_n.columns and not last_n[col].mode().empty) else "Unknown"
            next_row[col] = str(default_value)
        elif col in LAST_MATCH_FEATURES:
            series = last_n[col].dropna()
            if len(series) >= 2:
                kalman_series = calculate_kalman_filter(series)
                default_value = kalman_series.iloc[-1]
                if col not in NO_ROUND_FEATURES:
                    default_value = round(default_value)
            else:
                default_value = last_n[col].iloc[-1] if not last_n[col].empty else 0
                if col not in NO_ROUND_FEATURES:
                    default_value = round(default_value)
            next_row[col] = default_value
        elif col in ['home_travel_distance', 'away_travel_distance']:
            if is_first_match:
                if col == 'home_travel_distance':
                    next_row[col] = 0.0
                else:
                    temp_df = pd.DataFrame([{'home_team': home, 'away_team': away, 'game_date': match_date}])
                    temp_df = add_city_distance_feature(temp_df)
                    next_row[col] = temp_df['away_travel_distance'].iloc[0]
            else:
                series = last_n[col].dropna()
                if len(series) >= 2:
                    kalman_series = calculate_kalman_filter(series)
                    default_value = kalman_series.iloc[-1]
                else:
                    default_value = last_n[col].mean() if not last_n[col].empty else 0
                next_row[col] = default_value
        elif col in manual_numerical_cols:
            series = last_n[col].dropna()
            if len(series) >= 2:
                kalman_series = calculate_kalman_filter(series)
                default_value = kalman_series.iloc[-1]
                if col not in NO_ROUND_FEATURES:
                    default_value = round(default_value)
            else:
                default_value = last_n[col].mean() if not last_n[col].empty else 0
                if col not in NO_ROUND_FEATURES:
                    default_value = round(default_value)
            logging.info(f"Введите значение для {col} (по умолчанию: {default_value:.2f}):")
            value = input().strip()
            if value:
                try:
                    next_row[col] = float(value)
                except ValueError:
                    logging.info(f"Некорректное значение для {col}, используется прогноз Калмана: {default_value:.2f}")
                    next_row[col] = default_value
            else:
                next_row[col] = default_value
        else:
            series = last_n[col].dropna()
            if len(series) >= 2:
                kalman_series = calculate_kalman_filter(series)
                default_value = kalman_series.iloc[-1]
                if col not in NO_ROUND_FEATURES:
                    default_value = round(default_value)
            else:
                default_value = last_n[col].mean() if not last_n[col].empty else 0
                if col not in NO_ROUND_FEATURES:
                    default_value = round(default_value)
            next_row[col] = default_value

    last_n = sub.tail(10)

    # Добавляем дату и команды
    next_row["game_date"] = match_date
    next_row["home_team"] = home
    next_row["away_team"] = away

    return pd.DataFrame([next_row])[feat_cols], user_inputs

def calculate_kalman_filter(series: pd.Series) -> pd.Series:
    try:
        kf = KalmanFilter(
            initial_state_mean=series.iloc[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.1,
            transition_matrices=[1]
        )
        state_means, state_covariances = kf.filter(series.values)
        # Прогноз на один шаг вперёд
        next_mean, next_cov = kf.filter_update(
            filtered_state_mean=state_means[-1],
            filtered_state_covariance=state_covariances[-1],
            transition_matrix=np.array([1])
        )
        # Создаём новый индекс для прогноза
        if isinstance(series.index, (pd.RangeIndex, pd.Index)) and series.index.dtype == 'int64':
            next_index = series.index[-1] + 1
        else:
            next_index = series.index[-1]
        # Возвращаем сглаженные значения + прогноз
        result = pd.Series(np.append(state_means.flatten(), next_mean), index=list(series.index) + [next_index])
        return result
    except Exception as e:
        logging.info(f"Ошибка при расчёте фильтра Калмана: {e}")
        return series

def add_recent_games_features(df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    """Добавляет фичи на основе последних матчей с частичной векторизацией."""
    df = df.copy()
    df['home_team_goals_range_last'] = np.nan
    df['away_team_goals_range_last'] = np.nan
    df['combined_goals_range'] = np.nan
    df['lagged_total_goals'] = np.nan

    def get_team_range(team, date, n_games=10):
        team_games = df[((df['home_team'] == team) | (df['away_team'] == team)) &
                        (df['game_date'] < date)]
        last_games = team_games.tail(n_games)
        if len(last_games) > 0:
            return last_games['total_goals'].max() - last_games['total_goals'].min()
        return np.nan

    logging.info(f"Вычисляем калмановский тренд")
    kalman_trend = calculate_kalman_filter(df['total_goals'])
    df['total_goals_trend'] = kalman_trend
    df['lagged_total_goals'] = df['total_goals'].shift(1)
    df.loc[df.index[0], 'lagged_total_goals'] = kalman_trend.iloc[0]

    mask = ((df['home_team'] == home_team) | (df['away_team'] == home_team) |
            (df['home_team'] == away_team) | (df['away_team'] == away_team))

    # Векторизация через apply
    logging.info(f"Вычисляем размахи тоталов")
    df.loc[mask, 'home_team_goals_range_last'] = df[mask].apply(
        lambda row: get_team_range(home_team, row['game_date']), axis=1)
    df.loc[mask, 'away_team_goals_range_last'] = df[mask].apply(
        lambda row: get_team_range(away_team, row['game_date']), axis=1)

    logging.info(f"Вычисляем средние значения тоталов")
    avg_range = df['total_goals'].max() - df['total_goals'].min()
    df['home_team_goals_range_last'] = df['home_team_goals_range_last'].fillna(avg_range)
    df['away_team_goals_range_last'] = df['away_team_goals_range_last'].fillna(avg_range)
    df['combined_goals_range'] = (df['home_team_goals_range_last'] + df['away_team_goals_range_last'])

    def compute_winrate(team, is_home, date, n_games=5):
        games = df[((df['home_team'] == team) | (df['away_team'] == team)) & (df['game_date'] < date)]
        games = games.sort_values("game_date").tail(n_games)
        wins = 0
        for _, g in games.iterrows():
            if g["home_team"] == team and g["home_score"] > g["away_score"]:
                wins += 1
            elif g["away_team"] == team and g["away_score"] > g["home_score"]:
                wins += 1
        return wins / n_games if len(games) == n_games else np.nan

    logging.info(f"Вычисляем винрейт")
    df["home_winrate"] = df[mask].apply(
        lambda row: compute_winrate(home_team, True, row["game_date"]), axis=1)
    df["away_winrate"] = df[mask].apply(
        lambda row: compute_winrate(away_team, False, row["game_date"]), axis=1)

    return df

def add_ts_features(df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    h2h = df[((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
             ((df["home_team"] == away_team) & (df["away_team"] == home_team))].sort_values("game_date")

    if len(h2h) < 5:
        logging.info(f"Недостаточно матчей для TSFEL между {home_team} и {away_team}: {len(h2h)} < 5")
        return df

    cfg = tsfel.get_features_by_domain('statistical')
    features_df = pd.DataFrame(index=h2h.index)

    for i in range(len(h2h)):
        if i >= 5:  # Минимальный размер окна
            window = h2h.iloc[i - 5:i]["total_goals"].values
            if len(window) < 5 or np.isnan(window).any():
                continue  # Пропустить некорректные данные

            try:
                features = tsfel.time_series_features_extractor(cfg, window)
                features = features.add_prefix('ts_total_goals_')
                for col in features.columns:
                    features_df.at[h2h.index[i], col] = features[col].values[0]
            except Exception as e:
                logging.error(f"Ошибка TSFEL: {e}")
                continue

    features_df = features_df.fillna(0)
    return pd.concat([df, features_df], axis=1)

def load_scratches_batch(offset, batch_size):
    """Загружает батч данных из таблицы scratches."""
    try:
        resp = supabase.table("scratches").select("*").range(offset, offset + batch_size - 1).execute()
        return resp.data
    except Exception as e:
        logging.error(f"Ошибка при загрузке offset {offset}: {str(e)}")
        return []


def load_full_scratches() -> pd.DataFrame:
    """Загружает все записи из таблицы scratches."""
    engine = get_pg_engine()
    query = "SELECT * FROM scratches;"

    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Загружено {len(df)} записей из таблицы scratches")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке scratches: {str(e)}")
        raise

def add_team_readiness(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет метрики готовности команд с учетом травмированных игроков."""
    df = df.copy()
    scratches_df = load_full_scratches()
    scratches_df = scratches_df.drop_duplicates(subset=['game_id', 'team_type', 'player_name'])
    logging.info(f"Количество записей в scratches: {len(scratches_df)}")

    if scratches_df.empty:
        logging.warning("Таблица scratches пуста!")
        df['home_scratch_count'] = 0
        df['away_scratch_count'] = 0
    else:
        scratches_df['game_id'] = scratches_df['game_id'].astype(str).str.strip()
        df['game_id'] = df['game_id'].astype(str).str.strip()

        scratch_counts = (
            scratches_df
            .dropna(subset=['player_name'])
            .groupby(['game_id', 'team_type'])['player_name']
            .count()
            .unstack(fill_value=0)
            .reset_index()
        )
        scratch_counts.rename(columns={'home': 'home_scratch_count', 'away': 'away_scratch_count'}, inplace=True)

        df = df.merge(scratch_counts, on='game_id', how='left')
        df['home_scratch_count'] = df['home_scratch_count'].fillna(0).astype(int)
        df['away_scratch_count'] = df['away_scratch_count'].fillna(0).astype(int)

    df['home_total_players'] = (
        df['home_forwards_count'] + df['home_defense_count'] + df['home_goalies_count'] + df['home_scratch_count']
    )
    df['away_total_players'] = (
        df['away_forwards_count'] + df['away_defense_count'] + df['away_goalies_count'] + df['away_scratch_count']
    )
    df['home_readiness'] = (
        (df['home_forwards_count'] + df['home_defense_count'] + df['home_goalies_count']) /
        df['home_total_players'].clip(lower=1)
    )
    df['away_readiness'] = (
        (df['away_forwards_count'] + df['away_defense_count'] + df['away_goalies_count']) /
        df['away_total_players'].clip(lower=1)
    )
    df['home_readiness'] = df['home_readiness'].clip(0, 1)
    df['away_readiness'] = df['away_readiness'].clip(0, 1)
    df['readiness_diff'] = df['home_readiness'] - df['away_readiness']

    return df

def add_tournament_stage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    regular_season = df[df['game_type'] == 2].copy()
    team_pairs = regular_season.groupby(['home_team', 'away_team'])['game_date'].agg(['min', 'max'])

    def get_stage(row):
        # Предсезонные игры: сентябрь и начало октября
        if (row['game_date'].month == 9 or
            (row['game_date'].month == 10 and row['game_date'].day <= 10)) or row['game_type'] == 1:
            return 'preparation'
        elif row['game_type'] == 3:
            return 'champion_stage'
        elif row['game_type'] == 2:
            pair_key = (row['home_team'], row['away_team'])
            first_date = team_pairs.loc[pair_key, 'min'] if pair_key in team_pairs.index else row['game_date']
            last_date = team_pairs.loc[pair_key, 'max'] if pair_key in team_pairs.index else row['game_date']
            if row['game_date'].month >= 10 and row['game_date'].month <= 11:
                return 'early_stage'
            elif row['game_date'].month == 12 or row['game_date'].month <= 2:
                return 'mid_stage'
            else:
                return 'late_stage'
        return 'unknown'

    df['tournament_stage'] = df.apply(get_stage, axis=1)
    return df

def add_lagged_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['game_date', 'game_id'])
    all_teams = set(df['home_team']).union(set(df['away_team']))
    for team in all_teams:
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('game_date')
        df.loc[team_games.index, 'lagged_total_goals'] = team_games['total_goals'].shift(1).fillna(method='bfill')
    df['lagged_total_goals'] = df['lagged_total_goals'].fillna(df['total_goals'].mean())
    return df

def add_team_standings(df: pd.DataFrame, team_structure: dict) -> pd.DataFrame:
    """Добавляет позиции команд в таблице с векторизацией."""
    df = df.copy()
    df = df.sort_values('game_date')
    if df['game_id'].duplicated().sum() > 0:
        df.drop_duplicates(subset=['game_id', 'home_team', 'away_team', 'game_date'], inplace=True)

    def get_season(date):
        year = date.year if date.month >= 9 else date.year - 1
        return f"{year}-{year + 1}"

    df['season'] = df['game_date'].apply(get_season)

    team_to_conf_div = {}
    for division, info in team_structure.items():
        for team in info["teams"]:
            team_to_conf_div[team] = {"conference": info["conference"], "division": division}

    df['home_conference'] = df['home_team'].map(lambda x: team_to_conf_div.get(x, {}).get('conference', 'Unknown'))
    df['away_conference'] = df['away_team'].map(lambda x: team_to_conf_div.get(x, {}).get('conference', 'Unknown'))
    df['home_division'] = df['home_team'].map(lambda x: team_to_conf_div.get(x, {}).get('division', 'Unknown'))
    df['away_division'] = df['away_team'].map(lambda x: team_to_conf_div.get(x, {}).get('division', 'Unknown'))

    points_data = []
    home_points = df[['season', 'game_date', 'home_team', 'home_score', 'away_score']].copy()
    home_points['points'] = np.where(home_points['home_score'] > home_points['away_score'], 2,
                                     np.where(home_points['home_score'] == home_points['away_score'], 1, 0)).astype('int16')
    points_data.append(home_points[['season', 'game_date', 'home_team', 'points']].rename(columns={'home_team': 'team'}))

    away_points = df[['season', 'game_date', 'away_team', 'away_score', 'home_score']].copy()
    away_points['points'] = np.where(away_points['away_score'] > away_points['home_score'], 2,
                                     np.where(away_points['away_score'] == away_points['home_score'], 1, 0)).astype('int16')
    points_data.append(away_points[['season', 'game_date', 'away_team', 'points']].rename(columns={'away_team': 'team'}))

    matches_long = pd.concat(points_data, ignore_index=True)
    matches_long = matches_long.drop_duplicates().sort_values(['season', 'team', 'game_date'])
    if matches_long.duplicated(subset=['season', 'team', 'game_date']).sum() > 0:
        matches_long = matches_long.drop_duplicates(subset=['season', 'team', 'game_date'])

    matches_long['cum_points'] = matches_long.groupby(['season', 'team'])['points'].cumsum().astype('int16')
    matches_long['conference'] = matches_long['team'].map(lambda x: team_to_conf_div.get(x, {}).get('conference', 'Unknown'))
    matches_long['division'] = matches_long['team'].map(lambda x: team_to_conf_div.get(x, {}).get('division', 'Unknown'))

    def calculate_positions(group_df):
        group_df = group_df.copy()
        group_df['position'] = group_df['cum_points'].rank(method='min', ascending=False).astype('int16')
        return group_df[['season', 'game_date', 'team', 'position']]

    standings_list = []
    for season in matches_long['season'].unique():
        season_data = matches_long[matches_long['season'] == season].copy()
        league_pos = season_data.groupby('game_date', group_keys=False).apply(calculate_positions).rename(columns={'position': 'league_position'})
        conf_pos = season_data.groupby(['game_date', 'conference'], group_keys=False).apply(
            lambda g: calculate_positions(g).rename(columns={'position': 'conf_position'}))
        div_pos = season_data.groupby(['game_date', 'division'], group_keys=False).apply(
            lambda g: calculate_positions(g).rename(columns={'position': 'div_position'}))
        standings = league_pos.merge(conf_pos, on=['season', 'game_date', 'team'], how='left').merge(
            div_pos, on=['season', 'game_date', 'team'], how='left')
        standings_list.append(standings)

    standings_all = pd.concat(standings_list, ignore_index=True)
    standings_all = standings_all.drop_duplicates(subset=['season', 'game_date', 'team'])

    df = df.merge(
        standings_all.rename(columns={
            'team': 'home_team',
            'league_position': 'home_league_position',
            'conf_position': 'home_conf_position',
            'div_position': 'home_div_position'
        }),
        on=['season', 'game_date', 'home_team'],
        how='left'
    )

    df = df.merge(
        standings_all.rename(columns={
            'team': 'away_team',
            'league_position': 'away_league_position',
            'conf_position': 'away_conf_position',
            'div_position': 'away_div_position'
        }),
        on=['season', 'game_date', 'away_team'],
        how='left'
    )
    if len(df) > df['game_id'].nunique():
        df.drop_duplicates(subset=['game_id'], inplace=True)

    max_pos = max(df['home_league_position'].max(skipna=True) or 0, df['away_league_position'].max(skipna=True) or 0) + 1
    for col in ['home_league_position', 'away_league_position', 'home_conf_position', 'away_conf_position',
                'home_div_position', 'away_div_position']:
        df[col] = df[col].fillna(max_pos).astype('int16')

    df = df.drop(columns=['season'])
    return df

def add_corsi_fenwick(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет метрики Corsi и Fenwick."""
    df = df.copy()
    df['home_cf'] = df['home_sog'] + df['home_blockedshots'] + (df['total_shots_home'] - df['home_sog'])
    df['away_cf'] = df['away_sog'] + df['away_blockedshots'] + (df['total_shots_away'] - df['away_sog'])
    df['home_ca'] = df['away_sog'] + df['away_blockedshots'] + (df['total_shots_away'] - df['away_sog'])
    df['away_ca'] = df['home_sog'] + df['home_blockedshots'] + (df['total_shots_home'] - df['home_sog'])
    df['home_cf_pct'] = df['home_cf'] / (df['home_cf'] + df['home_ca'] + 1e-6) * 100
    df['away_cf_pct'] = 100 - df['home_cf_pct']
    df['home_ff'] = df['home_sog'] + (df['total_shots_home'] - df['home_sog'])
    df['away_ff'] = df['away_sog'] + (df['total_shots_away'] - df['away_sog'])
    df['home_fa'] = df['away_sog'] + (df['total_shots_away'] - df['away_sog'])
    df['away_fa'] = df['home_sog'] + (df['total_shots_home'] - df['home_sog'])
    df['home_ff_pct'] = df['home_ff'] / (df['home_ff'] + df['home_fa'] + 1e-6) * 100
    df['away_ff_pct'] = 100 - df['home_ff_pct']
    return df


def visualize_results(h2h_reset, y_train_pred, y_test_pred, kalman_trend, y_pred, mean_val, mae, rmse, home, away,
                      n_matches, split_idx, total_line=None):
    """Визуализирует результаты прогнозирования с доверительными интервалами.

    Args:
        h2h_reset: DataFrame с историческими данными
        y_train_pred: Прогнозы на тренировочной выборке (numpy array)
        y_test_pred: Прогнозы на тестовой выборке (numpy array)
        kalman_trend: Тренд Калмана (pd.Series или numpy array)
        y_pred: Вневыборочный прогноз (float)
        mean_val: Среднее значение тотала (float)
        mae: Средняя абсолютная ошибка (float or str)
        rmse: Среднеквадратичная ошибка (float or str)
        home: Домашняя команда (str)
        away: Гостевая команда (str)
        n_matches: Количество матчей (int)
        split_idx: Индекс разделения train/test (int)
        total_line: Букмекерская линия тоталов для следующего матча (float or None)
    """
    plt.figure(figsize=(16, 8))
    x_vals = list(range(len(h2h_reset)))
    dates = h2h_reset["game_date"].dt.strftime("%d.%m.%y")
    next_x = len(h2h_reset)

    # Фактические значения
    plt.plot(x_vals, h2h_reset["total_goals"], marker="o", linewidth=2, label="Фактические тоталы")

    # Прогнозы на тренировочной выборке
    if y_train_pred is not None and len(y_train_pred) > 0:
        plt.plot(x_vals[:split_idx], y_train_pred,
                 marker="x", linestyle="--", color="orange",
                 label="Прогнозы на train")

    # Прогнозы на тестовой выборке
    if y_test_pred is not None and len(y_test_pred) > 0:
        plt.plot(x_vals[split_idx:], y_test_pred,
                 marker="x", linestyle="--", color="orange",
                 label="Прогнозы на test")

    # Тренд Калмана
    kalman_trend = kalman_trend[:len(h2h_reset)]
    plt.plot(x_vals, kalman_trend, color="purple", linewidth=2,
             linestyle="-.", label="Актуальная форма команд")

    # Доверительный интервал
    if y_test_pred is not None and len(y_test_pred) > 0 and split_idx < len(h2h_reset):
        test_actual = h2h_reset["total_goals"].iloc[split_idx:].values
        errors = test_actual - y_test_pred
        std_dev = np.std(errors)
        plt.fill_between(
            x_vals[split_idx:],
            y_test_pred - 1.96 * std_dev,
            y_test_pred + 1.96 * std_dev,
            color='orange',
            alpha=0.2,
            label='95% доверительный интервал (test)'
        )
        plt.errorbar(
            next_x, y_pred,
            yerr=1.96 * std_dev,
            fmt='X',
            markersize=10,
            color='red',
            capsize=5,
            capthick=2,
            label=f'Прогноз ({y_pred:.1f}±{1.96 * std_dev:.1f})'
        )
    else:
        plt.scatter(next_x, y_pred, marker="X", s=150, color="red",
                    label=f"Прогноз: {y_pred:.1f}")

    # Букмекерская линия
    if total_line is not None:
        plt.axhline(y=total_line, color='green', linestyle='--', label=f"Букмекерская линия: {total_line}")
        plt.text(next_x, total_line + 0.2, f"{total_line}", ha="center", va="bottom", fontsize=10, color="green")

    last_total = h2h_reset['total_goals'].iloc[-1]
    if total_line is not None:
        bookmaker_lines = {"OddsAPI": total_line}  # Используем полученную линию
    else:
        bookmaker_lines = {}
    for book, line in bookmaker_lines.items():
        plt.plot([x_vals[-1], next_x], [last_total + (line - last_total), line], color='red', linestyle=':', alpha=0.5,
                 label=f"{book}: {line}")

    # Соединение с последним тоталом
    plt.plot([x_vals[-1], next_x], [h2h_reset['total_goals'].iloc[-1], y_pred],
             color="red", linewidth=1.5)

    # Подписи значений
    for x, act in zip(x_vals, h2h_reset["total_goals"]):
        plt.text(x, act + 0.3, str(act), ha="center", va="bottom", fontsize=8)

    # Линия разделения train/test
    if y_test_pred is not None and len(y_test_pred) > 0:
        plt.axvline(x=split_idx, color="black", linestyle=":", linewidth=1.8,
                    label="Разделение train/test")
        y_pos = plt.ylim()[1] * 0.95
        plt.text(split_idx - 1.5, y_pos, 'Train', ha='right', va='top', backgroundcolor='white')
        plt.text(split_idx + 1.5, y_pos, 'Test', ha='left', va='top', backgroundcolor='white')

    # Среднее значение
    plt.axhline(mean_val, linestyle="--", color="gray",
                label=f"Средний тотал пары: {mean_val:.1f}")

    # Оформление заголовка
    mae_str = f"{mae:.2f}" if isinstance(mae, (int, float)) else str(mae)
    rmse_str = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
    plt.title(f"Динамика тоталов: {home} vs {away} ({n_matches} матчей)\nMAE: {mae_str} | RMSE: {rmse_str}")

    # Оформление
    plt.xticks(ticks=x_vals[::3], labels=dates[::3], rotation=45)
    plt.xlabel("Дата матча")
    plt.ylabel("Тотал голов")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def calculate_recommendation_stats(pred_arr, actual_arr, line, direction):
    """
    Рассчитывает статистику рекомендаций для заданной линии и направления.

    Args:
        pred_arr: Массив предсказанных значений
        actual_arr: Массив фактических значений
        line: Значение линии тотала (например, 5.5)
        direction: Направление ('over' или 'under')

    Returns:
        Словарь с success_count, failure_count, return_count и rates
    """
    if direction == 'over':
        predicted = pred_arr > line
        actual_hit = actual_arr > line
        actual_fail = actual_arr < line
        actual_push = actual_arr == line
    else:  # 'under'
        predicted = pred_arr < line
        actual_hit = actual_arr < line
        actual_fail = actual_arr > line
        actual_push = actual_arr == line

    success_count = np.sum(predicted & actual_hit)
    failure_count = np.sum(predicted & actual_fail)
    return_count = np.sum(predicted & actual_push)
    total_cases = success_count + failure_count + return_count

    return {
        'success_count': success_count,
        'failure_count': failure_count,
        'return_count': return_count,
        'success_rate': success_count / total_cases if total_cases > 0 else 0,
        'failure_rate': failure_count / total_cases if total_cases > 0 else 0,
        'return_rate': return_count / total_cases if total_cases > 0 else 0
    }


def plot_feature_importances(model, X_test, y_test, feat_cols, cat_cols, preprocessor, home, away, top_n=50):
    try:
        # Преобразуем X_test через препроцессор
        X_test_preprocessed = model.named_steps['preprocessor'].transform(X_test)

        # Получаем имена признаков после препроцессинга
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            # Fallback: объединяем имена категориальных и числовых признаков
            ohe_cat_names = []
            for col in cat_cols:
                unique_vals = X_test[col].unique()
                ohe_cat_names.extend([f"cat_{col}_{val}" for val in unique_vals])
            feature_names = ohe_cat_names + num_cols

        # Рассчитываем важность признаков
        result = permutation_importance(
            model.named_steps['model'],
            X_test_preprocessed,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        # Создаём DataFrame с важностями
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False).head(top_n)

        # Визуализация
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        plt.xlabel("Permutation Importance")
        plt.title(f"Top {top_n} Features for {home} vs {away}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Ошибка при построении важности признаков: {e}")
        raise

def collect_game_info(game_id, client):
    landing_data = client.game_center.landing(game_id)
    right_rail_data = client.game_center.right_rail(game_id)
    play_by_play_data = client.game_center.play_by_play(game_id)

    # Собираем ростер (rosterSpots из play_by_play)
    roster = play_by_play_data.get('rosterSpots', [])
    home_roster = []
    away_roster = []
    for player in roster:
        first_name = player.get('firstName', {}).get('default', '') if isinstance(player.get('firstName'),
                                                                              dict) else player.get('firstName', '')
        last_name = player.get('lastName', {}).get('default', '') if isinstance(player.get('lastName'),
                                                                            dict) else player.get('lastName', '')
        full_name = f"{first_name} {last_name}".strip()
        if full_name:
            team_id = player.get('teamId')
            if team_id == landing_data.get('homeTeam', {}).get('id'):
                home_roster.append(full_name)
            elif team_id == landing_data.get('awayTeam', {}).get('id'):
                away_roster.append(full_name)

    # Собираем запасных/травмированных (scratches из right_rail)
    home_scratches = []
    away_scratches = []
    for team in ['homeTeam', 'awayTeam']:
        scratches = right_rail_data.get('gameInfo', {}).get(team, {}).get('scratches', [])
        for player in scratches:
            first_name = player.get('firstName', {}).get('default', '') if isinstance(player.get('firstName'),
                                                                                      dict) else player.get('firstName',
                                                                                                            '')
            last_name = player.get('lastName', {}).get('default', '') if isinstance(player.get('lastName'),
                                                                                    dict) else player.get('lastName',
                                                                                                          '')
            full_name = f"{first_name} {last_name}".strip()
            if full_name:
                if team == 'homeTeam':
                    home_scratches.append(full_name)
                else:
                    away_scratches.append(full_name)

    # Подсчет общего количества игроков (ростер + скретчес)
    home_total_players = len(home_roster) + len(home_scratches)
    away_total_players = len(away_roster) + len(away_scratches)

    # Обработка судей и линейных судей
    referees = [ref.get('default', 'N/A') for ref in right_rail_data.get('gameInfo', {}).get('referees', [])]
    linesmen = [line.get('default', 'N/A') for line in right_rail_data.get('gameInfo', {}).get('linesmen', [])]

    game_info = {
        'ID': str(game_id),
        'game_date': landing_data.get('gameDate', 'N/A'),
        'home_team': landing_data.get('homeTeam', {}).get('abbrev', 'N/A'),
        'away_team': landing_data.get('awayTeam', {}).get('abbrev', 'N/A'),
        'venue': landing_data.get('venue', {}).get('default', 'N/A'),
        'game_time': pd.to_datetime(landing_data.get('startTimeUTC')).tz_convert('US/Eastern').strftime(
            '%H:%M') if landing_data.get('startTimeUTC') else 'N/A',
        'home_coach': right_rail_data.get('gameInfo', {}).get('homeTeam', {}).get('headCoach', {}).get('default',
                                                                                                       'N/A'),
        'away_coach': right_rail_data.get('gameInfo', {}).get('awayTeam', {}).get('headCoach', {}).get('default',
                                                                                                       'N/A'),
        'referee_1': referees[0] if len(referees) > 0 else 'N/A',
        'referee_2': referees[1] if len(referees) > 1 else 'N/A',
        'linesman_1': linesmen[0] if len(linesmen) > 0 else 'N/A',
        'linesman_2': linesmen[1] if len(linesmen) > 1 else 'N/A',
        'home_scratch_count': len(home_scratches),
        'away_scratch_count': len(away_scratches),
        'home_total_players': home_total_players,
        'away_total_players': away_total_players
    }
    return game_info

def get_bookmaker_total_line(date, home_team, away_team, api_key):
    """
    Получает медианную линию тоталов от букмекеров для заданного матча с использованием Odds API.

    Args:
        date (datetime): Дата матча.
        home_team (str): Аббревиатура домашней команды (например, "BOS").
        away_team (str): Аббревиатура гостевой команды (например, "NYR").
        api_key (str): Ключ API для Odds API.

    Returns:
        float or None: Медианная линия тоталов или None, если данные недоступны.
    """
    url = f"{BASE_URL}sports/icehockey_nhl/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',  # Можно изменить на 'eu' или другие регионы при необходимости
        'markets': 'totals',  # Запрашиваем только рынок тоталов
        'date': date.strftime("%Y-%m-%d"),
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for event in data:
                # Проверяем соответствие команд (учитываем возможную инверсию home/away)
                if (event['home_team'] == home_team and event['away_team'] == away_team) or \
                   (event['home_team'] == away_team and event['away_team'] == home_team):
                    total_lines = []
                    for bookmaker in event['bookmakers']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'totals':
                                outcomes = market['outcomes']
                                if outcomes:  # Берем значение линии тотала (point)
                                    total_lines.append(outcomes[0]['point'])
                    if total_lines:
                        return np.median(total_lines)  # Возвращаем медиану для устойчивости
            logging.warning(f"Матч {home_team} vs {away_team} на {date.strftime('%Y-%m-%d')} не найден в Odds API.")
            return None
        else:
            logging.error(f"Ошибка Odds API: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Исключение при запросе к Odds API: {str(e)}")
        return None

def download_logo_png(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        if url.endswith('.svg'):
            img_buffer = BytesIO(response.content)
            png_buffer = BytesIO()
            cairosvg.svg2png(bytestring=img_buffer.getvalue(), write_to=png_buffer)
            png_buffer.seek(0)
            img = Image.open(png_buffer).convert("RGBA")
        else:
            img = Image.open(BytesIO(response.content)).convert("RGBA")
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background.convert("RGB")
    except Exception as e:
        logging.error(f"Ошибка при загрузке логотипа: {e}")
        return None

def resize_logo_keep_ratio(img, target_size):
    target_w, target_h = target_size
    ratio = min(target_w / img.width, target_h / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def render_text_group(texts, font, padding=5, line_spacing=5):
    draw_dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = [draw_dummy.textlength(t, font=font) for t in texts]
    heights = [font.getbbox(t)[3] - font.getbbox(t)[1] for t in texts]
    canvas_w = int(max(widths) + 2 * padding)
    canvas_h = int(sum(heights) + line_spacing * (len(texts) - 1) + 2 * padding)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    cd = ImageDraw.Draw(canvas)
    y = padding
    for i, t in enumerate(texts):
        tw = cd.textlength(t, font=font)
        x = (canvas_w - tw) // 2
        cd.text((x, y), t, fill="black", font=font)
        y += heights[i] + line_spacing
    return canvas

def add_defense_stability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Предполагаем, что стабильность защиты = (goalies_saves / goalies_total_shots_against) * 100
    df['home_defense_stability'] = (df['home_goalies_saves'] / df['home_goalies_total_shots_against'].replace(0, 1e-6)) * 100
    df['away_defense_stability'] = (df['away_goalies_saves'] / df['away_goalies_total_shots_against'].replace(0, 1e-6)) * 100
    df['home_defense_stability'] = df['home_defense_stability'].fillna(0)
    df['away_defense_stability'] = df['away_defense_stability'].fillna(0)
    return df

def generate_prematch_image(home, away, h2h, df):
    """Генерирует предматчевое изображение с логотипами, текстовой информацией и радарной диаграммой H2H статистики."""
    # Фильтрация очных встреч
    last_10 = h2h.tail(10) if len(h2h) >= 10 else h2h
    num_matches = len(h2h)
    min_year = h2h['game_date'].min().year if not h2h.empty else "N/A"

    # Расчёт побед
    home_wins = len(last_10[((last_10['home_team'] == home) & (last_10['home_score'] > last_10['away_score'])) |
                            ((last_10['away_team'] == home) & (last_10['away_score'] > last_10['home_score']))])
    away_wins = len(last_10[((last_10['home_team'] == away) & (last_10['home_score'] > last_10['away_score'])) |
                            ((last_10['away_team'] == away) & (last_10['away_score'] > last_10['home_score']))])
    home_win_percent_all = round((h2h.apply(lambda row: 1 if ((row['home_team'] == home and row['home_score'] > row['away_score']) or
                                                              (row['away_team'] == home and row['away_score'] > row['home_score'])) else 0, axis=1).sum() / num_matches * 100), 1) if num_matches > 0 else 0
    home_as_home_percent = round((h2h[h2h['home_team'] == home].apply(lambda row: 1 if row['home_score'] > row['away_score'] else 0, axis=1).sum() / len(h2h[h2h['home_team'] == home]) * 100), 1) if len(h2h[h2h['home_team'] == home]) > 0 else 0
    home_as_away_percent = round((h2h[h2h['away_team'] == home].apply(lambda row: 1 if row['away_score'] > row['home_score'] else 0, axis=1).sum() / len(h2h[h2h['away_team'] == home]) * 100), 1) if len(h2h[h2h['away_team'] == home]) > 0 else 0
    away_win_percent_all = round((h2h.apply(lambda row: 1 if ((row['home_team'] == away and row['home_score'] > row['away_score']) or
                                                              (row['away_team'] == away and row['away_score'] > row['home_score'])) else 0, axis=1).sum() / num_matches * 100), 1) if num_matches > 0 else 0
    away_as_away_percent = round((h2h[h2h['away_team'] == away].apply(lambda row: 1 if row['away_score'] > row['home_score'] else 0, axis=1).sum() / len(h2h[h2h['away_team'] == away]) * 100), 1) if len(h2h[h2h['away_team'] == away]) > 0 else 0
    away_as_home_percent = round((h2h[h2h['home_team'] == away].apply(lambda row: 1 if row['home_score'] > row['away_score'] else 0, axis=1).sum() / len(h2h[h2h['home_team'] == away]) * 100), 1) if len(h2h[h2h['home_team'] == away]) > 0 else 0

    # Метрики для радарной диаграммы
    metrics = [
        "Винрейт", "Среднее число голов", "Удары в створ", "Силовые приёмы",
        "Штрафные минуты", "Плюс-минус полевых игроков", "Реализация большинства",
        "Реализация вбрасываний", "Бросковая эффективность", "Контроль шайбы",
        "Частота замен", "Сейвы вратаря", "Эффективность защиты", "Вероятность овертаймов"
    ]

    # Расчёт H2H метрик для домашней команды
    home_win_rate_h2h = round((last_10.apply(
        lambda row: 1 if ((row['home_team'] == home and row['home_score'] > row['away_score']) or
                          (row['away_team'] == home and row['away_score'] > row['home_score'])) else 0,
        axis=1).sum() / len(last_10) * 100) if len(last_10) > 0 else 0)
    home_avg_goals_h2h = round(last_10.apply(
        lambda row: row['home_score'] if row['home_team'] == home else row['away_score'], axis=1).mean(), 1) if not last_10.empty else 0
    home_avg_sog_h2h = round(last_10.apply(
        lambda row: row['home_sog'] if row['home_team'] == home else row['away_sog'], axis=1).mean(), 1) if not last_10.empty else 0
    home_avg_hits_h2h = round(last_10.apply(
        lambda row: row['home_hits'] if row['home_team'] == home else row['away_hits'], axis=1).mean(), 1) if not last_10.empty else 0
    home_avg_pim_h2h = round(last_10.apply(
        lambda row: row['home_pim'] if row['home_team'] == home else row['away_pim'], axis=1).mean(), 1) if not last_10.empty else 0
    home_avg_plusminus_h2h = round(last_10.apply(
        lambda row: row['home_total_plusminus'] if row['home_team'] == home else row['away_total_plusminus'], axis=1).mean(), 1) if not last_10.empty else 0
    home_avg_powerplay_h2h = round(last_10.apply(
        lambda row: row['home_powerplaypctg'] if row['home_team'] == home else row['away_powerplaypctg'], axis=1).mean() * 100, 1) if not last_10.empty else 0
    home_avg_faceoff_h2h = round(last_10.apply(
        lambda row: row['home_faceoffwinningpctg'] if row['home_team'] == home else row['away_faceoffwinningpctg'], axis=1).mean() * 100, 1) if not last_10.empty else 0
    home_avg_shooting_eff_h2h = round(last_10.apply(
        lambda row: row['home_shooting_efficiency'] if row['home_team'] == home else row['away_shooting_efficiency'], axis=1).mean(), 2) if not last_10.empty else 0
    home_avg_puck_control_h2h = round(last_10.apply(
        lambda row: (row['home_cf_pct'] + row['home_ff_pct']) / 2 if row['home_team'] == home else (row['away_cf_pct'] + row['away_ff_pct']) / 2, axis=1).mean(), 1) if not last_10.empty else 0
    home_shifts_per_min_h2h = round(last_10.apply(
        lambda row: (row['home_shifts_sum'] / row['game_duration']) if row['home_team'] == home else (row['away_shifts_sum'] / row['game_duration']), axis=1).mean(), 2) if not last_10.empty else 0
    home_savepctg_h2h = round(last_10.apply(
        lambda row: row['home_goalies_saves'] / row['home_goalies_total_shots_against'] * 100 if row['home_team'] == home
        else row['away_goalies_saves'] / row['away_goalies_total_shots_against'] * 100, axis=1).mean(), 1) if not last_10.empty else 0
    home_defense_stability_h2h = round(last_10.apply(
        lambda row: (row['home_defense_blockedshots'] / row['home_sog'] * 100) if row['home_team'] == home else (row['away_defense_blockedshots'] / row['away_sog'] * 100), axis=1).mean(), 1) if not last_10.empty else 0
    home_overtime_prob_h2h = round(len(last_10[last_10['period_number'] > 3]) / 10 * 100, 1) if not last_10.empty else 0

    # Расчёт H2H метрик для гостевой команды
    away_win_rate_h2h = round((last_10.apply(
        lambda row: 1 if ((row['home_team'] == away and row['home_score'] > row['away_score']) or
                          (row['away_team'] == away and row['away_score'] > row['home_score'])) else 0,
        axis=1).sum() / len(last_10) * 100) if len(last_10) > 0 else 0)
    away_avg_goals_h2h = round(last_10.apply(
        lambda row: row['away_score'] if row['away_team'] == away else row['home_score'], axis=1).mean(), 1) if not last_10.empty else 0
    away_avg_sog_h2h = round(last_10.apply(
        lambda row: row['away_sog'] if row['away_team'] == away else row['home_sog'], axis=1).mean(), 1) if not last_10.empty else 0
    away_avg_hits_h2h = round(last_10.apply(
        lambda row: row['away_hits'] if row['away_team'] == away else row['home_hits'], axis=1).mean(), 1) if not last_10.empty else 0
    away_avg_pim_h2h = round(last_10.apply(
        lambda row: row['away_pim'] if row['away_team'] == away else row['home_pim'], axis=1).mean(), 1) if not last_10.empty else 0
    away_avg_plusminus_h2h = round(last_10.apply(
        lambda row: row['away_total_plusminus'] if row['away_team'] == away else row['home_total_plusminus'], axis=1).mean(), 1) if not last_10.empty else 0
    away_avg_powerplay_h2h = round(last_10.apply(
        lambda row: row['away_powerplaypctg'] if row['away_team'] == away else row['home_powerplaypctg'], axis=1).mean() * 100, 1) if not last_10.empty else 0
    away_avg_faceoff_h2h = round(last_10.apply(
        lambda row: row['away_faceoffwinningpctg'] if row['away_team'] == away else row['home_faceoffwinningpctg'], axis=1).mean() * 100, 1) if not last_10.empty else 0
    away_avg_shooting_eff_h2h = round(last_10.apply(
        lambda row: row['away_shooting_efficiency'] if row['away_team'] == away else row['home_shooting_efficiency'], axis=1).mean(), 2) if not last_10.empty else 0
    away_avg_puck_control_h2h = round(last_10.apply(
        lambda row: (row['away_cf_pct'] + row['away_ff_pct']) / 2 if row['away_team'] == away else (row['home_cf_pct'] + row['home_ff_pct']) / 2, axis=1).mean(), 1) if not last_10.empty else 0
    away_shifts_per_min_h2h = round(last_10.apply(
        lambda row: (row['home_shifts_sum'] / row['game_duration']) if row['home_team'] == away else (row['away_shifts_sum'] / row['game_duration']), axis=1).mean(), 2) if not last_10.empty else 0
    away_savepctg_h2h = round(last_10.apply(
        lambda row: row['home_goalies_saves'] / row['home_goalies_total_shots_against'] * 100 if row['home_team'] == away
        else row['away_goalies_saves'] / row['away_goalies_total_shots_against'] * 100, axis=1).mean(), 1) if not last_10.empty else 0
    away_defense_stability_h2h = round(last_10.apply(
        lambda row: (row['home_defense_blockedshots'] / row['home_sog'] * 100) if row['home_team'] == away else (row['away_defense_blockedshots'] / row['away_sog'] * 100), axis=1).mean(), 1) if not last_10.empty else 0
    away_overtime_prob_h2h = round(len(last_10[last_10['period_number'] > 3]) / 10 * 100, 1) if not last_10.empty else 0

    # Данные для радарной диаграммы (H2H)
    home_data = [
        home_win_rate_h2h, home_avg_goals_h2h, home_avg_sog_h2h, home_avg_hits_h2h,
        home_avg_pim_h2h, home_avg_plusminus_h2h, home_avg_powerplay_h2h,
        home_avg_faceoff_h2h, home_avg_shooting_eff_h2h, home_avg_puck_control_h2h,
        home_shifts_per_min_h2h, home_savepctg_h2h, home_defense_stability_h2h, home_overtime_prob_h2h
    ]
    away_data = [
        away_win_rate_h2h, away_avg_goals_h2h, away_avg_sog_h2h, away_avg_hits_h2h,
        away_avg_pim_h2h, away_avg_plusminus_h2h, away_avg_powerplay_h2h,
        away_avg_faceoff_h2h, away_avg_shooting_eff_h2h, away_avg_puck_control_h2h,
        away_shifts_per_min_h2h, away_savepctg_h2h, away_defense_stability_h2h, away_overtime_prob_h2h
    ]

    # Словарь для соответствия метрик и имен столбцов в df
    metric_to_column = {
        "Винрейт": "win_rate",
        "Среднее число голов": "score",
        "Удары в створ": "sog",
        "Силовые приёмы": "hits",
        "Штрафные минуты": "pim",
        "Плюс-минус полевых игроков": "total_plusminus",
        "Реализация большинства": "powerplaypctg",
        "Реализация вбрасываний": "faceoffwinningpctg",
        "Бросковая эффективность": "shooting_efficiency",
        "Контроль шайбы": "puck_control",
        "Частота замен": "shifts_per_min",
        "Сейвы вратаря": "savepctg",
        "Эффективность защиты": "defense_stability",
        "Вероятность овертаймов": "overtime_prob"
    }

    # Расчёт максимальных значений по последним 10 играм
    last_10_df = df[
        ((df['home_team'].isin([home, away])) | (df['away_team'].isin([home, away])))
    ].sort_values('game_date').tail(10)

    league_max = []
    for metric in metrics:
        col_name = metric_to_column[metric]
        if metric == "Контроль шайбы":
            home_vals = last_10_df.apply(
                lambda row: (row['home_cf_pct'] + row['home_ff_pct']) / 2 if row['home_team'] == home
                else (row['away_cf_pct'] + row['away_ff_pct']) / 2, axis=1)
            away_vals = last_10_df.apply(
                lambda row: (row['home_cf_pct'] + row['home_ff_pct']) / 2 if row['home_team'] == away
                else (row['away_cf_pct'] + row['away_ff_pct']) / 2, axis=1)
            all_vals = np.concatenate([home_vals, away_vals])
            max_val = np.nanmax([0 if np.isnan(x) or np.isinf(x) else x for x in all_vals]) + 1e-6
        elif metric == "Частота замен":
            home_vals = last_10_df.apply(
                lambda row: row['home_shifts_sum'] / max(row['game_duration'], 1e-6) if row['home_team'] == home
                else row['away_shifts_sum'] / max(row['game_duration'], 1e-6), axis=1)
            away_vals = last_10_df.apply(
                lambda row: row['home_shifts_sum'] / max(row['game_duration'], 1e-6) if row['home_team'] == away
                else row['away_shifts_sum'] / max(row['game_duration'], 1e-6), axis=1)
            all_vals = np.concatenate([home_vals, away_vals])
            max_val = np.nanmax([0 if np.isnan(x) or np.isinf(x) else x for x in all_vals]) + 1e-6
        elif metric == "Сейвы вратаря":
            home_vals = last_10_df.apply(
                lambda row: (row['home_goalies_saves'] / max(row['home_goalies_total_shots_against'], 1e-6) * 100) if
                row['home_team'] == home
                else (row['away_goalies_saves'] / max(row['away_goalies_total_shots_against'], 1e-6) * 100), axis=1)
            away_vals = last_10_df.apply(
                lambda row: (row['home_goalies_saves'] / max(row['home_goalies_total_shots_against'], 1e-6) * 100) if
                row['home_team'] == away
                else (row['away_goalies_saves'] / max(row['away_goalies_total_shots_against'], 1e-6) * 100), axis=1)
            all_vals = np.concatenate([home_vals, away_vals])
            max_val = np.nanmax([0 if np.isnan(x) or np.isinf(x) else x for x in all_vals]) + 1e-6
        elif metric == "Винрейт":
            home_vals = last_10_df.apply(
                lambda row: 100 if ((row['home_team'] == home and row['home_score'] > row['away_score']) or
                                    (row['away_team'] == home and row['away_score'] > row['home_score'])) else 0,
                axis=1)
            away_vals = last_10_df.apply(
                lambda row: 100 if ((row['home_team'] == away and row['home_score'] > row['away_score']) or
                                    (row['away_team'] == away and row['away_score'] > row['home_score'])) else 0,
                axis=1)
            all_vals = np.concatenate([home_vals, away_vals])
            max_val = np.nanmax([0 if np.isnan(x) or np.isinf(x) else x for x in all_vals]) + 1e-6
        elif metric == "Вероятность овертаймов":
            all_vals = last_10_df.apply(
                lambda row: 100 if row['period_number'] > 3 else 0, axis=1)
            max_val = np.nanmax([0 if np.isnan(x) or np.isinf(x) else x for x in all_vals]) + 1e-6
        else:
            home_vals = last_10_df.apply(
                lambda row: row[f'home_{col_name}'] if row['home_team'] == home else row[f'away_{col_name}'], axis=1)
            away_vals = last_10_df.apply(
                lambda row: row[f'home_{col_name}'] if row['home_team'] == away else row[f'away_{col_name}'], axis=1)
            all_vals = np.concatenate([home_vals, away_vals])
            max_val = np.nanmax([0 if np.isnan(x) or np.isinf(x) else x for x in all_vals]) + 1e-6
        league_max.append(max_val)

    # Обработка пропусков и некорректных значений
    home_data = [0 if np.isnan(x) or np.isinf(x) else x for x in home_data]
    away_data = [0 if np.isnan(x) or np.isinf(x) else x for x in away_data]

    # Проверка соответствия количества метрик
    assert len(metrics) == len(home_data) == len(away_data) == len(league_max), "Несоответствие количества метрик и данных"

    percent_metrics = {"Винрейт", "Реализация большинства", "Реализация вбрасываний", "Сейвы вратаря",
                       "Эффективность защиты", "Вероятность овертаймов"}
    home_normalized = [h / max_v * 100 if max_v != 0 and metrics[i] not in percent_metrics else h if max_v != 0 else 0
                       for i, (h, max_v) in enumerate(zip(home_data, league_max))]
    away_normalized = [a / max_v * 100 if max_v != 0 and metrics[i] not in percent_metrics else a if max_v != 0 else 0
                       for i, (a, max_v) in enumerate(zip(away_data, league_max))]

    # Замыкание данных для радарной диаграммы
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    home_normalized += home_normalized[:1]
    away_normalized += away_normalized[:1]

    # Проверка длин
    assert len(angles) == len(home_normalized) == len(away_normalized), "Несоответствие длин углов и данных"

    # Загружаем логотипы
    home_logo_url = f"https://assets.nhle.com/logos/nhl/svg/{home}_light.svg"
    away_logo_url = f"https://assets.nhle.com/logos/nhl/svg/{away}_light.svg"
    home_logo = download_logo_png(home_logo_url)
    away_logo = download_logo_png(away_logo_url)
    if not home_logo or not away_logo:
        logging.error("Не удалось загрузить логотипы.")
        return None

    # Масштабируем логотипы
    target_size = (300, 300)
    home_logo = resize_logo_keep_ratio(home_logo, target_size)
    away_logo = resize_logo_keep_ratio(away_logo, target_size)

    # Создаем изображение
    img_width, img_height = 900, 1200
    img = Image.new('RGB', (img_width, img_height), 'white')

    # Вставляем логотипы
    spacing = 125
    total_width = home_logo.width + spacing + away_logo.width
    start_x = (img_width - total_width) // 2
    y_position = 150
    max_logo_height = max(home_logo.height, away_logo.height)
    home_x = start_x
    away_x = start_x + home_logo.width + spacing
    home_y = y_position + (max_logo_height - home_logo.height)
    away_y = y_position + (max_logo_height - away_logo.height)
    img.paste(home_logo, (home_x, home_y))
    img.paste(away_logo, (away_x, away_y))

    # Рендерим текстовые блоки
    font_path = "arial.ttf"
    try:
        font = ImageFont.truetype(font_path, size=12)
        bold_font = ImageFont.truetype(font_path, size=14)
    except:
        font = ImageFont.load_default()
        bold_font = ImageFont.load_default()

    group1 = render_text_group(["Хозяева", TEAM_FULL_NAMES.get(home, home)], font)
    group2 = render_text_group(["Гости", TEAM_FULL_NAMES.get(away, away)], font)
    group3 = render_text_group(["Выборка", f"{num_matches} игр", f"с {min_year} года"], font)
    group4 = render_text_group(["Последние 10 личных встреч",
                                f"{home} {home_wins} - {away_wins} {away}" if len(last_10) == 10
                                else "Недостаточно матчей"], font)
    group5 = render_text_group([f"{home_win_percent_all}% побед во всех очных встречах",
                                f"{home_as_home_percent}% побед как хозяева",
                                f"{home_as_away_percent}% побед как гости"], font)
    group6 = render_text_group([f"{away_win_percent_all}% побед во всех очных встречах",
                                f"{away_as_away_percent}% побед как гости",
                                f"{away_as_home_percent}% побед как хозяева"], font)

    # Вставляем текстовые блоки
    img.paste(group3, ((img_width - group3.width) // 2, 210), group3)
    img.paste(group1, (home_x + (home_logo.width - group1.width) // 2, home_y - group1.height - 10), group1)
    img.paste(group2, (away_x + (away_logo.width - group2.width) // 2, away_y - group2.height - 10), group2)
    img.paste(group5, (home_x + (home_logo.width - group5.width) // 2, home_y + home_logo.height + 20), group5)
    img.paste(group6, (away_x + (away_logo.width - group6.width) // 2, away_y + away_logo.height + 20), group6)
    img.paste(group4, ((img_width - group4.width) // 2, home_y + home_logo.height - group4.height - 10), group4)

    # Создаем радарную диаграмму
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, home_normalized, 'o-', linewidth=2, label=home, color='b')
    ax.fill(angles, home_normalized, alpha=0.25, color='b')
    ax.plot(angles, away_normalized, 'o-', linewidth=2, label=away, color='r')
    ax.fill(angles, away_normalized, alpha=0.25, color='r')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    plt.title("Статистика последних 10 личных встреч", size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Сохраняем диаграмму в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    radar_img = Image.open(buf)

    # Масштабируем диаграмму
    radar_width = 600
    radar_height = int(radar_img.height * radar_width / radar_img.width)
    radar_img = radar_img.resize((radar_width, radar_height), Image.LANCZOS)

    # Вставляем диаграмму в основное изображение
    radar_x = (img_width - radar_width) // 2
    radar_y = img_height - radar_height - 150
    img.paste(radar_img, (radar_x, radar_y))

    # Закрываем временные объекты
    plt.close(fig)
    buf.close()

    # Показываем изображение
    img.show()
    return img

def main():
    """Основная функция выполнения программы."""
    try:
        warnings.filterwarnings("ignore", category=UserWarning)
        client = NHLClient(verbose=False)
        logging.info("Загружаем данные...")
        df = load_full_table()
        logging.info("Подготавливаем данные...")
        for col in ["referee_1", "referee_2", "linesman_1", "linesman_2"]:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', 'Unknown')

        if df.empty:
            logging.info("Таблица пуста")
            return

        all_teams = sorted(set(team for division in TEAM_STRUCTURE.values() for team in division["teams"]))

        # Display match count matrix
        match_counts = pd.DataFrame(0, index=all_teams, columns=all_teams, dtype=int)
        for home_team in all_teams:
            for away_team in all_teams:
                if home_team != away_team:
                    count = len(df[((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
                                   ((df["home_team"] == away_team) & (df["away_team"] == home_team))])
                    match_counts.loc[home_team, away_team] = count

        match_counts_with_sums = match_counts.copy()
        match_counts_with_sums['Total'] = match_counts_with_sums.sum(axis=1)
        match_counts_with_sums.loc['Total'] = match_counts_with_sums.sum(axis=0)

        print(f"Матрица количества сыгранных матчей между командами (с суммами):")
        print(" " * 5, end="")
        for team in all_teams:
            print(f"{team:>4}", end=" ")
        print(" Total")
        print("-" * (5 + 5 * (len(all_teams) + 1)))
        for home_team in all_teams:
            print(f"{home_team:<4}", end="|")
            for away_team in all_teams:
                count = match_counts_with_sums.loc[home_team, away_team]
                print("   -" if home_team == away_team else f"{count:>4}", end=" ")
            print(f"|{match_counts_with_sums.loc[home_team, 'Total']:>5}")
        print("-" * (5 + 5 * (len(all_teams) + 1)))
        print("Total", end="|")
        for away_team in all_teams:
            print(f"{match_counts_with_sums.loc['Total', away_team]:>4}", end=" ")
        print(f"|{match_counts_with_sums.loc['Total', 'Total'] // 2:>5}")

        print("\nДоступные команды:")
        for i in range(0, len(all_teams), 8):
            print("  ".join(f"{t:<4}" for t in all_teams[i:i + 8]))

        # Match selection
        print("\nВыберите способ выбора матча:")
        print("1. Показать ближайшие матчи по дате")
        print("2. Ввести команды вручную")
        choice = input("Введите 1 или 2: ").strip()

        home = None
        away = None
        date_str = None
        match_date = None
        total_line = None  # Инициализируем переменную для линии тоталов
        if choice == "1":
            print("\nВведите дату матча (формат: ГГГГ-ММ-ДД, например, 2025-09-21):")
            date_input = input().strip()
            try:
                match_date = datetime.strptime(date_input, "%Y-%m-%d")
                date_str = match_date.strftime("%Y-%m-%d")
                schedule = client.schedule.get_schedule(date=date_str)
                if not schedule.get('games', []):
                    print(f"На {date_str} матчи не найдены.")
                    return

                games_data = []
                for i, game in enumerate(schedule.get('games', []), 1):
                    home_team = game.get('homeTeam', {}).get('abbrev', 'N/A')
                    away_team = game.get('awayTeam', {}).get('abbrev', 'N/A')
                    start_time = pd.to_datetime(game.get('startTimeUTC', 'N/A')).strftime('%H:%M') if game.get('startTimeUTC') else 'N/A'
                    season = str(game.get('season', 'N/A'))
                    if season != 'N/A' and len(season) == 8 and season.isdigit():
                        season = f"{season[:4]}-{int(season[:4]) + 1}"
                    match_count = match_counts.loc[home_team, away_team] if home_team in match_counts.index and away_team in match_counts.columns else 0
                    games_data.append({
                        'Order': i,
                        'Home Team': home_team,
                        'Away Team': away_team,
                        'Match Count': match_count,
                        'Start Time': start_time,
                        'Season': season
                    })

                games_df = pd.DataFrame(games_data)
                print("\nДоступные матчи на", date_str)
                print(games_df.to_string(index=False))

                game_order = input("\nВведите номер матча (Order): ").strip()
                try:
                    game_order = int(game_order)
                    selected_game = games_df[games_df['Order'] == game_order]
                    if selected_game.empty:
                        print("Некорректный номер матча.")
                        return
                    home = selected_game['Home Team'].iloc[0]
                    away = selected_game['Away Team'].iloc[0]
                    # Получаем букмекерскую линию тоталов
                    total_line = get_bookmaker_total_line(match_date, home, away, API_KEY)
                    if total_line is not None:
                        logging.info(f"Букмекерская линия тоталов: {total_line}")
                    else:
                        logging.info("Букмекерская линия тоталов недоступна")
                except ValueError:
                    print("Некорректный ввод номера матча.")
                    return
            except ValueError:
                print("Некорректный формат даты.")
                return
        elif choice == "2":
            home = input("\nВыберите домашнюю команду: ").strip().upper()
            away = input("Выберите гостевую команду: ").strip().upper()
            total_line = None  # Для ручного ввода даты нет, поэтому линия недоступна
        else:
            print("Некорректный выбор. Пожалуйста, введите 1 или 2.")
            return

        if home not in all_teams or away not in all_teams:
            logging.error("Ошибка: введены некорректные коды команд.")
            return

        # Filter h2h data
        logging.info(f"Делаем h2h срез в df")
        h2h = df[((df["home_team"] == home) & (df["away_team"] == away)) |
                 ((df["home_team"] == away) & (df["away_team"] == home))].sort_values("game_date")

        # Добавляем столбец winner
        h2h['winner'] = h2h.apply(
            lambda row: row['home_team'] if row['home_score'] > row['away_score']
            else (row['away_team'] if row['away_score'] > row['home_score'] else 'Tie'),
            axis=1
        )

        if h2h.empty:
            logging.info("Очных встреч нет.")
            return

        # Apply feature engineering to h2h
        logging.info("Начинаем добавление признаков для h2h...")
        h2h = add_home_advantage_features(h2h)
        h2h = add_city_distance_feature(h2h)
        h2h = coach_factor(h2h)
        h2h = add_team_readiness(h2h)
        h2h = add_tournament_stage(h2h)
        h2h = add_lagged_totals(h2h)
        h2h = add_team_standings(h2h, TEAM_STRUCTURE)
        h2h = add_corsi_fenwick(h2h)
        h2h = add_recent_games_features(h2h, home, away)
        logging.info("Начинаем добавление температурных признаков для h2h...")
        h2h = add_temperature_feature(h2h, TEAM_CITY_COORDS)

        # Handle missing values
        for col in h2h.columns:
            if col in ["referee_1", "referee_2", "linesman_1", "linesman_2"] and h2h[col].dtype == "object":
                h2h[col] = h2h[col].fillna("Unknown")
            elif h2h[col].dtype != "object":
                h2h[col] = h2h[col].fillna(0)

        # Training mode selection
        print("\nВыберите режим обучения:")
        print("1. Выбрать конкретную модель")
        print("2. Подобрать лучшую модель")
        print("3. Ансамблировать все доступные модели")
        training_choice = input("Введите номер: ").strip()

        # Execute training based on choice
        if training_choice in ["1", "2"]:
            logging.info(f"Запуск train_evaluate с выбором {training_choice}")
            model, feat_cols, cat_cols, mae, rmse, split_idx, simplified_pred, preds_train, preds_test, arima_pred, test_data = train_evaluate(
                h2h, home, away, training_choice)
        elif training_choice == "3":
            logging.info("Запуск train_stacking_evaluate для ансамбля")
            model, feat_cols, cat_cols, mae, rmse, split_idx, simplified_pred, preds_train, preds_test, arima_pred, test_data = train_stacking_evaluate(
                h2h, home, away)
        else:
            logging.error("Некорректный выбор режима обучения.")
            return

        # Forecasting and visualization
        if simplified_pred is not None:
            y_pred = int(round(simplified_pred))
            h2h_reset = h2h.sort_values("game_date").reset_index(drop=True)
            actual_totals = h2h_reset["total_goals"].values
            mean_val = round(h2h_reset["total_goals"].mean(), 2)
            kalman_trend = calculate_kalman_filter(h2h_reset["total_goals"])
            y_pred_history = np.zeros_like(actual_totals)
        else:
            X_next, user_inputs = build_next_row(h2h, home, away, feat_cols, cat_cols, df, client, match_date=date_str)
            for col in X_next.columns[X_next.isnull().any()]:
                X_next[col] = "Unknown" if col in cat_cols else 0

            # Display feature table
            h2h_reset = h2h.sort_values("game_date").reset_index(drop=True)
            last_game = h2h_reset[feat_cols + ['game_date']].iloc[-1:].copy()
            last_game['game_date'] = last_game['game_date'].dt.strftime("%d.%m.%y")
            if isinstance(user_inputs['game_date'], pd.Timestamp):
                X_next['game_date'] = user_inputs['game_date'].strftime("%d.%m.%y")
            else:
                X_next['game_date'] = "Unknown"

            display_cols = [col for col in feat_cols if col != 'game_date']
            combined_table = pd.concat(
                [X_next[display_cols + ['game_date']].T, last_game[display_cols + ['game_date']].T], axis=1)
            combined_table.columns = [f"Next Match ({X_next['game_date'].iloc[0]})",
                                     f"Last Match ({last_game['game_date'].iloc[0]})"]
            combined_table = combined_table.sort_index()
            logging.info("Таблица признаков для следующего матча и последней игры:")
            logging.info("──────────────────────────")
            max_width = 20
            logging.info(combined_table.to_string(justify="left", col_space=max_width))
            logging.info("───────────────────────────────")

            # Plot feature importances
            if model is not None:
                logging.info("Построение графика важности признаков...")
                try:
                    if test_data is not None:
                        X_test, y_test = test_data
                        X_test_clean = X_test.copy()
                    else:
                        logging.warning("test_data отсутствует, используются данные h2h для визуализации важности признаков.")
                        X_test_clean = h2h_reset[feat_cols].copy()
                        y_test = h2h_reset["total_goals"].values
                    for col in cat_cols:
                        if col in X_test_clean.columns:
                            X_test_clean[col] = X_test_clean[col].astype(str).fillna("Unknown")
                    for col in feat_cols:
                        if col not in X_test_clean.columns:
                            X_test_clean[col] = 0 if col not in cat_cols else "Unknown"

                    preprocessor = model.named_steps['preprocessor']
                    plot_feature_importances(
                        model,
                        X_test_clean,
                        y_test,
                        feat_cols,
                        cat_cols,
                        preprocessor,
                        home,
                        away,
                        top_n=50
                    )
                except Exception as e:
                    logging.error(f"Ошибка при построении важности признаков: {e}")

            y_pred = int(round(model.predict(X_next[feat_cols])[0]))
            X_history = h2h_reset[feat_cols].copy()
            for col in X_history.columns[X_history.isnull().any()]:
                X_history[col] = "Unknown" if col in cat_cols else 0
            y_pred_history = np.round(model.predict(X_history)).astype(int)
            actual_totals = h2h_reset["total_goals"].values
            mean_val = round(h2h_reset["total_goals"].mean(), 2)
            kalman_trend = calculate_kalman_filter(h2h_reset["total_goals"])

        # Calculate confidence interval and recommendations
        if 'preds_test' in locals() and preds_test is not None and len(preds_test) > 0:
            test_actual = h2h_reset["total_goals"].iloc[split_idx:].values
            errors = test_actual - preds_test
            std_dev = np.std(errors)
        else:
            std_dev = None

        logging.info(f"Прогноз модели на следующий матч: [{y_pred}] голов")

        if std_dev is not None:
            threshold = 0.8
            lines = np.arange(round(mean_val - 1.0), round(mean_val + 1.5), 0.5)
            prob_over = 1 - norm.cdf(lines, loc=y_pred, scale=std_dev)
            prob_under = norm.cdf(lines, loc=y_pred, scale=std_dev)

            over_candidates = [(line, prob) for line, prob in zip(lines, prob_over) if prob > threshold]
            under_candidates = [(line, prob) for line, prob in zip(lines, prob_under) if prob > threshold]

            if over_candidates and not under_candidates:
                line, prob = max(over_candidates, key=lambda x: x[1])
                logging.info(f"💡 Рекомендация: Тотал Больше {line:.1f} с вероятностью {prob * 100:.1f}%")
            elif under_candidates and not over_candidates:
                line, prob = max(under_candidates, key=lambda x: x[1])
                logging.info(f"💡 Рекомендация: Тотал Меньше {line:.1f} с вероятностью {prob * 100:.1f}%")
            elif over_candidates and under_candidates:
                best_over = max(over_candidates, key=lambda x: x[1])
                best_under = max(under_candidates, key=lambda x: x[1])
                if best_over[1] > best_under[1]:
                    logging.info(f"💡 Рекомендация: Тотал Больше {best_over[0]:.1f} с вероятностью {best_over[1] * 100:.1f}%")
                else:
                    logging.info(f"💡 Рекомендация: Тотал Меньше {best_under[0]:.1f} с вероятностью {best_under[1] * 100:.1f}%")
            else:
                logging.info("⚠️ Нет уверенных рекомендаций (вероятность ниже порога 65%)")
        else:
            logging.info("⚠️ Недостаточно данных для доверительного интервала, рекомендация не выдается")

        # Рисуем график с прогнозами
        visualize_results(
            h2h_reset,
            preds_train,
            preds_test,
            kalman_trend,
            y_pred,
            mean_val,
            mae,
            rmse,
            home,
            away,
            len(h2h_reset),
            split_idx,
            total_line=total_line
        )

        # Рисуем предматчевую аналитику
        df = add_home_advantage_features(df)
        df = add_corsi_fenwick(df)
        h2h = add_home_advantage_features(h2h)
        h2h = add_corsi_fenwick(h2h)
        df = add_defense_stability(df)
        h2h = add_defense_stability(h2h)
        generate_prematch_image(home, away, h2h, df)

    except Exception as e:
        logging.info(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()

