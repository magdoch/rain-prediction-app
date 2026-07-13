from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Налаштування сторінки
# --------------------------------------------------

st.set_page_config(
    page_title="Rain Tomorrow Prediction",
    page_icon="🌧️",
    layout="wide"
)


# --------------------------------------------------
# Завантаження готової моделі
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "aussie_rain.joblib"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл моделі не знайдено: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    artifacts = load_artifacts()
except Exception as error:
    st.error(f"Помилка завантаження моделі: {error}")
    st.stop()


model = artifacts["model"]
imputer = artifacts["imputer"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

input_cols = artifacts["input_cols"]
numeric_cols = artifacts["numeric_cols"]
categorical_cols = artifacts["categorical_cols"]
encoded_cols = artifacts["encoded_cols"]


# --------------------------------------------------
# Заголовок
# --------------------------------------------------

st.title("🌧️ Прогноз дощу на завтра")

st.write(
    """
    Введіть поточні погодні показники. Після натискання кнопки
    введені дані пройдуть імпутацію, масштабування та кодування,
    після чого модель спрогнозує, чи піде дощ завтра.
    """
)


# --------------------------------------------------
# Варіанти категоріальних ознак
# --------------------------------------------------

location_options = [
    "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek",
    "Ballarat", "Bendigo", "Brisbane", "Cairns", "Canberra",
    "Cobar", "CoffsHarbour", "Dartmoor", "Darwin", "GoldCoast",
    "Hobart", "Katherine", "Launceston", "Melbourne",
    "MelbourneAirport", "Mildura", "Moree", "MountGambier",
    "MountGinini", "Newcastle", "Nhil", "NorahHead", "NorfolkIsland",
    "Nuriootpa", "PearceRAAF", "Penrith", "Perth", "PerthAirport",
    "Portland", "Richmond", "Sale", "SalmonGums", "Sydney",
    "SydneyAirport", "Townsville", "Tuggeranong", "Uluru",
    "WaggaWagga", "Walpole", "Watsonia", "Williamtown",
    "Witchcliffe", "Wollongong", "Woomera"
]

wind_options = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW"
]

rain_today_options = ["No", "Yes"]


# --------------------------------------------------
# Форма
# --------------------------------------------------

with st.form("weather_form"):

    st.subheader("Місце та температура")

    col1, col2, col3 = st.columns(3)

    with col1:
        location = st.selectbox(
            "Location",
            options=location_options
        )

    with col2:
        min_temp = st.number_input(
            "MinTemp, °C",
            value=10.0,
            step=0.1
        )

    with col3:
        max_temp = st.number_input(
            "MaxTemp, °C",
            value=20.0,
            step=0.1
        )

    st.subheader("Опади та сонячність")

    col1, col2, col3 = st.columns(3)

    with col1:
        rainfall = st.number_input(
            "Rainfall, mm",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col2:
        evaporation = st.number_input(
            "Evaporation",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

    with col3:
        sunshine = st.number_input(
            "Sunshine, hours",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.1
        )

    st.subheader("Вітер")

    col1, col2, col3 = st.columns(3)

    with col1:
        wind_gust_dir = st.selectbox(
            "WindGustDir",
            options=wind_options
        )

    with col2:
        wind_gust_speed = st.number_input(
            "WindGustSpeed, km/h",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

    with col3:
        wind_dir_9am = st.selectbox(
            "WindDir9am",
            options=wind_options
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        wind_dir_3pm = st.selectbox(
            "WindDir3pm",
            options=wind_options
        )

    with col2:
        wind_speed_9am = st.number_input(
            "WindSpeed9am, km/h",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

    with col3:
        wind_speed_3pm = st.number_input(
            "WindSpeed3pm, km/h",
            min_value=0.0,
            value=15.0,
            step=1.0
        )

    st.subheader("Вологість і тиск")

    col1, col2, col3 = st.columns(3)

    with col1:
        humidity_9am = st.number_input(
            "Humidity9am, %",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with col2:
        humidity_3pm = st.number_input(
            "Humidity3pm, %",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    with col3:
        pressure_9am = st.number_input(
            "Pressure9am, hPa",
            value=1015.0,
            step=0.1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        pressure_3pm = st.number_input(
            "Pressure3pm, hPa",
            value=1012.0,
            step=0.1
        )

    with col2:
        cloud_9am = st.number_input(
            "Cloud9am",
            min_value=0.0,
            max_value=8.0,
            value=4.0,
            step=1.0
        )

    with col3:
        cloud_3pm = st.number_input(
            "Cloud3pm",
            min_value=0.0,
            max_value=8.0,
            value=4.0,
            step=1.0
        )

    st.subheader("Температура протягом дня")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp_9am = st.number_input(
            "Temp9am, °C",
            value=14.0,
            step=0.1
        )

    with col2:
        temp_3pm = st.number_input(
            "Temp3pm, °C",
            value=19.0,
            step=0.1
        )

    with col3:
        rain_today = st.selectbox(
            "RainToday",
            options=rain_today_options
        )

    submitted = st.form_submit_button(
        "Спрогнозувати",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# Функція препроцесингу
# --------------------------------------------------

def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Застосовує той самий препроцесинг, який використовувався
    під час навчання готової моделі.
    """

    numeric_df = input_df[numeric_cols]
    categorical_df = input_df[categorical_cols].copy()

    # Імпутація числових значень
    numeric_imputed = imputer.transform(numeric_df)

    # Масштабування числових значень
    numeric_scaled = scaler.transform(numeric_imputed)

    numeric_processed_df = pd.DataFrame(
        numeric_scaled,
        columns=numeric_cols,
        index=input_df.index
    )

    # Заповнення пропусків у категоріальних змінних
    categorical_df = categorical_df.fillna("Unknown")

    # Кодування категоріальних змінних
    categorical_encoded = encoder.transform(categorical_df)

    if hasattr(categorical_encoded, "toarray"):
        categorical_encoded = categorical_encoded.toarray()

    categorical_processed_df = pd.DataFrame(
        categorical_encoded,
        columns=encoded_cols,
        index=input_df.index
    )

    # Об'єднання всіх ознак
    processed_df = pd.concat(
        [
            numeric_processed_df,
            categorical_processed_df
        ],
        axis=1
    )

    return processed_df


# --------------------------------------------------
# Прогнозування
# --------------------------------------------------

if submitted:

    user_data = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today
    }

    input_df = pd.DataFrame(
        [user_data],
        columns=input_cols
    )

    try:
        processed_input = preprocess_input(input_df)

        prediction = model.predict(processed_input)[0]
        probabilities = model.predict_proba(processed_input)[0]

        classes = model.classes_

        probability_map = {
            str(class_name): float(probability)
            for class_name, probability in zip(
                classes,
                probabilities
            )
        }

        yes_probability = probability_map.get("Yes", 0.0)
        no_probability = probability_map.get("No", 0.0)

        st.divider()
        st.subheader("Результат")

        if str(prediction) == "Yes":
            st.error("🌧️ Прогноз: завтра піде дощ — Yes")
        else:
            st.success("☀️ Прогноз: завтра дощу не буде — No")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Ймовірність дощу",
                f"{yes_probability:.2%}"
            )

        with col2:
            st.metric(
                "Ймовірність відсутності дощу",
                f"{no_probability:.2%}"
            )

        st.progress(yes_probability)

        with st.expander("Введені дані"):
            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Помилка прогнозування: {error}")