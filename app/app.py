import streamlit as st
import pandas as pd
import joblib
import os

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ============================================
# CUSTOM DESIGN
# ============================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef7f5, #f5f8fc);
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #123c4a;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #58727c;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(0,0,0,0.08);
}

.number {
    font-size: 32px;
    font-weight: 800;
    color: #126782;
}

.label {
    font-size: 15px;
    color: #607d86;
}

.aqi-card {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    color: white;
    margin: 20px 0;
}

.aqi-number {
    font-size: 58px;
    font-weight: 900;
}

.aqi-category {
    font-size: 27px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# PATHS
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "final_aqi_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "..", "models", "aqi_feature_columns.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "..", "data", "air_quality_city_day.csv"
)

# ============================================
# LOAD MODEL AND DATA
# ============================================

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURE_PATH)

air_data = pd.read_csv(DATA_PATH)

air_data["Date"] = pd.to_datetime(
    air_data["Date"]
)

# ============================================
# HEADER
# ============================================

st.markdown(
    '<div class="main-title">🌍 Environmental Monitoring & Air Quality Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Powered Air Quality Prediction System</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🌫️ Pollutant Inputs")

st.sidebar.write(
    "Enter pollutant concentrations to estimate AQI."
)

pm25 = st.sidebar.number_input(
    "PM2.5", min_value=0.0, value=50.0
)

pm10 = st.sidebar.number_input(
    "PM10", min_value=0.0, value=100.0
)

no = st.sidebar.number_input(
    "NO", min_value=0.0, value=10.0
)

no2 = st.sidebar.number_input(
    "NO2", min_value=0.0, value=25.0
)

nox = st.sidebar.number_input(
    "NOx", min_value=0.0, value=30.0
)

nh3 = st.sidebar.number_input(
    "NH3", min_value=0.0, value=15.0
)

co = st.sidebar.number_input(
    "CO", min_value=0.0, value=1.0
)

so2 = st.sidebar.number_input(
    "SO2", min_value=0.0, value=10.0
)

o3 = st.sidebar.number_input(
    "O3", min_value=0.0, value=30.0
)

benzene = st.sidebar.number_input(
    "Benzene", min_value=0.0, value=2.0
)

toluene = st.sidebar.number_input(
    "Toluene", min_value=0.0, value=5.0
)

xylene = st.sidebar.number_input(
    "Xylene", min_value=0.0, value=2.0
)

predict = st.sidebar.button(
    "🔍 Predict AQI",
    use_container_width=True
)

# ============================================
# AQI CATEGORY
# ============================================

def get_category(aqi):

    if aqi <= 50:
        return "Good 😊", "Air quality is considered good.", "#2e8b57"

    elif aqi <= 100:
        return "Satisfactory 🙂", "Air quality is acceptable.", "#7a9e35"

    elif aqi <= 200:
        return "Moderate 😐", "Sensitive individuals may experience discomfort.", "#d99a00"

    elif aqi <= 300:
        return "Poor 😷", "Health effects may be experienced by some people.", "#d96b27"

    elif aqi <= 400:
        return "Very Poor ⚠️", "Health alert: increased risk of respiratory effects.", "#c94c4c"

    else:
        return "Severe 🚨", "Health emergency conditions may affect the population.", "#8b1e3f"


# ============================================
# PREDICTION
# ============================================

if predict:

    input_data = pd.DataFrame(
        [[
            pm25,
            pm10,
            no,
            no2,
            nox,
            nh3,
            co,
            so2,
            o3,
            benzene,
            toluene,
            xylene
        ]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    category, message, color = get_category(
        prediction
    )

    st.subheader("🌫️ Air Quality Prediction")

    st.markdown(
        f"""
        <div class="aqi-card"
             style="background-color:{color};">

            <div>Predicted Air Quality Index</div>

            <div class="aqi-number">
                {prediction:.2f}
            </div>

            <div class="aqi-category">
                {category}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"💡 {message}"
    )

    st.subheader("📊 Pollutant Input Levels")

    pollutant_data = pd.DataFrame(
        {
            "Pollutant": feature_columns,
            "Value": input_data.iloc[0].values
        }
    )

    st.bar_chart(
        pollutant_data.set_index("Pollutant")
    )

    st.success(
        "Prediction generated using the Tuned Random Forest Machine Learning model."
    )

# ============================================
# HOME DASHBOARD
# ============================================

else:

    st.subheader(
        "🌱 Welcome to Air Quality Intelligence"
    )

    st.write(
        "Enter pollutant values in the sidebar and click "
        "**Predict AQI** to estimate air quality."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="number">26</div>
                <div class="label">Indian Cities</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="number">29,531</div>
                <div class="label">Dataset Records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">
                <div class="number">12</div>
                <div class="label">Pollutant Features</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="card">
                <div class="number">91.22%</div>
                <div class="label">Model R²</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ========================================
    # MODEL PERFORMANCE
    # ========================================

    st.subheader("🏆 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("R² Score", "0.9122")

    with c2:
        st.metric("MAE", "20.72")

    with c3:
        st.metric("RMSE", "40.09")

    st.write(
        "The final model is a Tuned Random Forest Regressor."
    )

    # ========================================
    # HISTORICAL AQI
    # ========================================

    st.markdown("---")

    st.subheader("📈 Historical AQI Analysis")

    selected_city = st.selectbox(
        "🏙️ Select a City",
        sorted(
            air_data["City"].dropna().unique()
        )
    )

    city_data = air_data[
        air_data["City"] == selected_city
    ].sort_values("Date")

    average_aqi = city_data["AQI"].mean()

    maximum_aqi = city_data["AQI"].max()

    latest_values = city_data["AQI"].dropna()

    latest_aqi = (
        latest_values.iloc[-1]
        if len(latest_values) > 0
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Average AQI",
            f"{average_aqi:.2f}"
        )

    with c2:
        st.metric(
            "Maximum AQI",
            f"{maximum_aqi:.2f}"
        )

    with c3:
        st.metric(
            "Latest Recorded AQI",
            f"{latest_aqi:.2f}"
        )

    historical = city_data[
        ["Date", "AQI"]
    ].dropna()

    if not historical.empty:

        historical = historical.set_index(
            "Date"
        )

        st.line_chart(
            historical["AQI"]
        )

    # ========================================
    # POLLUTANT TREND
    # ========================================

    st.subheader("🌫️ Pollutant Trend")

    selected_pollutant = st.selectbox(
        "Select a pollutant",
        feature_columns
    )

    pollutant_history = city_data[
        ["Date", selected_pollutant]
    ].dropna()

    if not pollutant_history.empty:

        pollutant_history = pollutant_history.set_index(
            "Date"
        )

        st.line_chart(
            pollutant_history[selected_pollutant]
        )

    # ========================================
    # ENVIRONMENTAL INSIGHTS
    # ========================================

    st.markdown("---")

    st.subheader("💡 Key Environmental Insights")

    st.info(
        "PM2.5 was the most important feature in our "
        "Random Forest model with approximately 30.42% "
        "feature importance."
    )

    st.info(
        "CO was the second most important feature with "
        "approximately 25.84% feature importance."
    )