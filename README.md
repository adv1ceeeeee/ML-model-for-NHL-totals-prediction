# 🏒 NHL Game Prediction Pipeline

Многофункциональный пайплайн на Python для прогнозирования тотала голов в матчах НХЛ.  
Учитывает географию, погоду, усталость, судей, тренеров, статистику команд и многое другое.

## 📦 Возможности

- Подключение к PostgreSQL (PgAdmin) через SQLAlchemy
- Подключение к NHL API через (`nhlpy`) для прогноза по актуальному расписанию лиги
- Расширенная инженерия признаков:
  - историческая температура (`Meteostat`)
  - усталость от переездов (`geopy`)
  - влияние тренеров, судей и времени матча
  - автоподбор фичей через (`tsfel`)
  - показатели эффективности (Corsi, Fenwick)
  - турнирное положение команд
  - показатели готовности команд (травмы, состав)
  - актуальная форма команд (`pykalman`)
- Автоматический выбор модели в зависимости от размера выборки личных матчей
- Возможность фильтровать фичи по VIX-фактору для ухода от мультиколлинеарности (для небустинговых решений)
- Поддержка формирования ансамблей через (`StackingRegressor`) и ARIMA-моделей
- Загрузка актуальных котировок топовых международных букмекеров для корректировки прогнозов через (`OddsAPI`) (нужно получить API-ключ на сайте https://www.oddsapi.co/)
- Встроенное логгирование с подсветкой (colorama) для отладки
- Кэширование тяжёлых операций (`joblib` и `memory`) и распараллеливание через (`ThreadPoolExecutor`)
- Вывод итогового отчёта через (`cairosvg`), (`PIL`) и (`io`)

## 🚀 Быстрый старт

### 1. Установка необходимых зависимостей

Создайте виртуальное окружение и установите зависимости:

```bash
pip install -r requirements.txt

pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.1
catboost>=1.2
pykalman>=0.9.5
geopy>=2.4
sqlalchemy>=2.0
psycopg2-binary>=2.9
joblib>=1.3
tsfel>=0.1.4
meteostat>=1.6
nhlpy>=0.1.5
cairosvg>=2.7
matplotlib>=3.7
colorama>=0.4
Pillow>=10.0
requests>=2.31
ipython>=8.10

```


### 2. Настройка подключения к БД

```bash
PG_HOST = "your_host"
PG_DATABASE = "your_DB"
PG_USER = "your_user"
PG_PASSWORD = "your_password"
PG_PORT = "your_port"
DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
```

Или задайте переменную окружения:

```bash
export DATABASE_URL="postgresql://your_user:your_password@your_host:your_port/your_DB"
```


### 3. Схема работы кода

1. Выбор команд через
```bash
print("1. Показать ближайшие матчи по дате")
print("2. Ввести команды вручную")
choice = input("Введите 1 или 2: ").strip()
```
- Если выбираем 1, то автоматически на сайт НХЛ по API (`nhlpy`) отправляется запрос о расписании игр на конкретную дату и подгружаются фичи для прогноза
- Если выбираем 2, то вручную вводим все команды и доступные фичи для прогноза

2. Выбирается режим обучения
```bash
print("\nВыберите режим обучения:")
print("1. Выбрать конкретную модель")
print("2. Подобрать лучшую модель")
print("3. Ансамблировать все доступные модели")
training_choice = input("Введите номер: ").strip()
```
- Если выбираем 1, то код позволяет выбрать одну из 12 моделей, в зависимости от размера выборки
- Если выбираем 2, то код автоматически подбирает оптимальную модель по метрике (`RMSE`)
- Если выбираем 3, то код ансамблирует все модели через (`StackingRegressor`), мета-модель также выбирается в зависимости от размера выборки

3. Вывод рекомендации по ставке
```bash
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
```

4. Визуализация прогноза
```bash
visualize_results(...)
```

5. Визуализация отчёта
```bash
generate_prematch_image(...)
```

Пример стороннего запуска

```bash
from nhl_model import load_full_table, add_temperature_feature, add_city_distance_feature, train_stacking_evaluate

df = load_full_table("nhl_games_extended")
df = add_temperature_feature(df, TEAM_CITY_COORDS)
df = add_city_distance_feature(df)

model, feat_cols, cat_cols, mae, rmse, *_ = train_stacking_evaluate(
    h2h_df=df, home_team="TOR", away_team="MTL"
)
```

### 4. Создание БД в PostgreSQL

Необходимые данные для работы модели:
```bash
CREATE TABLE public.RosterPlayers (
  id integer NOT NULL DEFAULT nextval('"RosterPlayers_id_seq"'::regclass),
  game_id text NOT NULL,
  team_type text CHECK (team_type = ANY (ARRAY['home'::text, 'away'::text])),
  player_name text NOT NULL,
  game_date date,
  home_team text,
  away_team text,
  position text,
  toi text,
  CONSTRAINT RosterPlayers_pkey PRIMARY KEY (id),
  CONSTRAINT RosterPlayers_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nhl_games_extended(game_id)
);
CREATE TABLE public.game_period_stats (
  id integer NOT NULL DEFAULT nextval('game_period_stats_id_seq'::regclass),
  game_id text,
  period_number integer,
  period_type text CHECK (period_type = ANY (ARRAY['REG'::text, 'OT'::text, 'SO'::text])),
  goals_home integer,
  goals_away integer,
  shots_home integer,
  shots_away integer,
  home_puck_control double precision,
  away_puck_control double precision,
  home_pim integer,
  away_pim integer,
  home_hits integer,
  away_hits integer,
  CONSTRAINT game_period_stats_pkey PRIMARY KEY (id),
  CONSTRAINT game_period_stats_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nhl_games_extended(game_id)
);
CREATE TABLE public.nhl_games_extended (
  game_id text NOT NULL,
  game_date date,
  game_type integer,
  venue text,
  period_type text,
  period_number integer,
  home_id integer,
  home_abbrev text,
  home_score integer,
  home_sog integer,
  home_name text,
  home_city text,
  away_id integer,
  away_abbrev text,
  away_score integer,
  away_sog integer,
  away_name text,
  away_city text,
  home_forwards_count integer,
  home_forwards_goals integer,
  home_forwards_assists integer,
  home_forwards_hits integer,
  home_forwards_pim integer,
  home_forwards_blockedshots integer,
  home_forwards_shifts integer,
  home_forwards_plusminus integer,
  home_forwards_giveaways integer,
  home_forwards_takeaways integer,
  home_defense_count integer,
  home_defense_goals integer,
  home_defense_assists integer,
  home_defense_hits integer,
  home_defense_pim integer,
  home_defense_blockedshots integer,
  home_defense_shifts integer,
  home_defense_plusminus integer,
  home_defense_giveaways integer,
  home_defense_takeaways integer,
  home_forwards_avg_toi double precision,
  home_defense_avg_toi double precision,
  home_forwards_total_toi double precision,
  home_defense_total_toi double precision,
  home_skaters_total_toi double precision,
  home_total_plusminus integer,
  home_total_giveaways integer,
  home_total_takeaways integer,
  home_goalies_count integer,
  home_goalies_saves integer,
  home_goalies_savepctg double precision,
  home_goalies_evenstrength_shots_against integer,
  home_goalies_powerplay_shots_against integer,
  home_goalies_shorthanded_shots_against integer,
  home_goalies_evenstrength_goals_against integer,
  home_goalies_powerplay_goals_against integer,
  home_goalies_shorthanded_goals_against integer,
  home_goalies_total_shots_against integer,
  away_forwards_count integer,
  away_forwards_goals integer,
  away_forwards_assists integer,
  away_forwards_hits integer,
  away_forwards_pim integer,
  away_forwards_blockedshots integer,
  away_forwards_shifts integer,
  away_forwards_plusminus integer,
  away_forwards_giveaways integer,
  away_forwards_takeaways integer,
  away_defense_count integer,
  away_defense_goals integer,
  away_defense_assists integer,
  away_defense_hits integer,
  away_defense_pim integer,
  away_defense_blockedshots integer,
  away_defense_shifts integer,
  away_defense_plusminus integer,
  away_defense_giveaways integer,
  away_defense_takeaways integer,
  away_forwards_avg_toi double precision,
  away_defense_avg_toi double precision,
  away_forwards_total_toi double precision,
  away_defense_total_toi double precision,
  away_skaters_total_toi double precision,
  away_total_plusminus integer,
  away_total_giveaways integer,
  away_total_takeaways integer,
  away_goalies_count integer,
  away_goalies_saves integer,
  away_goalies_savepctg double precision,
  away_goalies_evenstrength_shots_against integer,
  away_goalies_powerplay_shots_against integer,
  away_goalies_shorthanded_shots_against integer,
  away_goalies_evenstrength_goals_against integer,
  away_goalies_powerplay_goals_against integer,
  away_goalies_shorthanded_goals_against integer,
  away_goalies_total_shots_against integer,
  home_puck_control_total double precision,
  away_puck_control_total double precision,
  referee_1 text,
  referee_2 text,
  linesman_1 text,
  linesman_2 text,
  home_coach text,
  away_coach text,
  home_faceoffwinningpctg double precision,
  away_faceoffwinningpctg double precision,
  home_powerplaypctg double precision,
  away_powerplaypctg double precision,
  home_pim integer,
  away_pim integer,
  home_hits integer,
  away_hits integer,
  home_blockedshots integer,
  away_blockedshots integer,
  home_giveaways integer,
  away_giveaways integer,
  home_takeaways integer,
  away_takeaways integer,
  game_time text,
  home_team text,
  away_team text,
  neutral_site boolean,
  home_powerplay_chances integer,
  away_powerplay_chances integer,
  total_shots_home integer,
  total_shots_away integer,
  game_duration double precision,
  CONSTRAINT nhl_games_extended_pkey PRIMARY KEY (game_id)
);
CREATE TABLE public.rostersscratches (
  id integer NOT NULL DEFAULT nextval('rostersscratches_id_seq'::regclass),
  game_id text NOT NULL,
  game_date date,
  home_team text,
  away_team text,
  home_roster text,
  away_roster text,
  home_scratches text,
  away_scratches text,
  home_scratch_count integer,
  away_scratch_count integer,
  home_total_players integer,
  away_total_players integer,
  CONSTRAINT rostersscratches_pkey PRIMARY KEY (id),
  CONSTRAINT rosters_scratches_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nhl_games_extended(game_id)
);
CREATE TABLE public.scratches (
  id integer NOT NULL DEFAULT nextval('scratches_id_seq'::regclass),
  game_id text,
  team_type text CHECK (team_type = ANY (ARRAY['home'::text, 'away'::text])),
  player_name text,
  scratch_order integer,
  CONSTRAINT scratches_pkey PRIMARY KEY (id),
  CONSTRAINT scratches_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nhl_games_extended(game_id)
);
CREATE TABLE public.scratchplayers (
  id integer NOT NULL DEFAULT nextval('scratchplayers_id_seq'::regclass),
  game_id text NOT NULL,
  team_type text CHECK (team_type = ANY (ARRAY['home'::text, 'away'::text])),
  player_name text NOT NULL,
  CONSTRAINT scratchplayers_pkey PRIMARY KEY (id),
  CONSTRAINT scratchplayers_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nhl_games_extended(game_id)
);
CREATE TABLE public.stars (
  id integer NOT NULL DEFAULT nextval('stars_id_seq'::regclass),
  game_id text,
  star_number integer CHECK (star_number >= 1 AND star_number <= 3),
  player_id text,
  player_name text,
  team text,
  position text,
  goals integer,
  assists integer,
  points integer,
  CONSTRAINT stars_pkey PRIMARY KEY (id),
  CONSTRAINT stars_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.nhl_games_extended(game_id)
);
```

Получить данные можно через загрузчик ()



