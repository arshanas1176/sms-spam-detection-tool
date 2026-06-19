# ==========================================================
# SMS SPAM DETECTION TOOL
# Developed using:
# Streamlit + Scikit-Learn + TF-IDF + Logistic Regression
# ==========================================================

# =====================
# IMPORT LIBRARIES
# =====================

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

from streamlit_option_menu import option_menu


# =====================
# PAGE CONFIGURATION
# =====================

st.set_page_config(
    page_title="SMS Spam Detection Tool",
    page_icon="📩",
    layout="wide"
)
# =====================
# CUSTOM CSS
# =====================

st.markdown("""
    <style>
        .css-1aumxhk {
            background-color: #f0f2f6;
        }
    .metric-card {
            padding: 15px;
            border-radius: 15px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

    </style>
""", unsafe_allow_html=True)

# =====================
# LOAD TRAINED MODEL
# =====================

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# =====================
# HISTORY FUNCTION
# =====================

from datetime import datetime

def save_history(message, prediction, confidence):

    history = pd.DataFrame({
        "Message": [message],
        "Prediction": [prediction],
        "Confidence (%)": [confidence],
        "Timestamp": [
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ]
    })

    history.to_csv(
        "prediction_history.csv",
        mode="a",
        header=not os.path.exists(
            "prediction_history.csv"
        ),
        index=False
    )

# =====================
# SIDEBAR MENU
# =====================

with st.sidebar:

    selected = option_menu(
        menu_title="SMS Spam Detector",

        options=[
            "Dashboard",
            "Single Prediction",
            "Bulk Prediction",
            "History",
            "About"
        ],

        icons=[
            "house",
            "chat-dots",
            "file-earmark-arrow-up",
            "clock-history",
            "info-circle"
        ],

        default_index=0
    )


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if selected == "Dashboard":

    st.markdown("""
                <h1 style='text-align:center;color:#1f77b4;'>
                📩 SMS Spam Detection Tool
                </h1>
                <p style='text-align:center;font-size:18px;'>
                AI-Powered Real-Time SMS Spam Classification System
                </p>
                """,unsafe_allow_html=True)

    if os.path.exists("prediction_history.csv"):

        history = pd.read_csv(
            "prediction_history.csv"
        )

        total_messages = len(history)

        spam_count = len(
            history[
                history["Prediction"] == "Spam"
            ]
        )

        ham_count = len(
            history[
                history["Prediction"] == "Legitimate"
            ]
        )

    else:

        total_messages = 0
        spam_count = 0
        ham_count = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""<div style="background:#ffffff;padding:20px;border-radius:15px;text-align:center;box-shadow:0px 4px 10px rgba(0,0,0,0.15);">
                    <h4>📨 Messages Checked</h4>
                    <h1 style="color:#1f77b4;">
                    {total_messages}
                    </h1>
                    </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div style="background:#ffffff;padding:15px;border-radius:15px;text-align:center;box-shadow:0px 4px 10px rgba(0,0,0,0.15);">
                    <h4>⚠️ Spam Messages</h4>
                    <h1 style="color:#d62728;">
                    {spam_count}
                    </h1>
                    </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""<div style="background:#ffffff;padding:15px;border-radius:15px;text-align:center;box-shadow:0px 4px 10px rgba(0,0,0,0.15);">
                    <h4>✅ Legitimate Messages</h4>
                    <h1 style="color:#2ca02c;">
                    {ham_count}
                    </h1>
                    </div>""", unsafe_allow_html=True)

    st.divider()

    if total_messages > 0:

        st.subheader("Prediction Distribution")

        fig = px.pie(
            names=[
                "Spam",
                "Legitimate"
            ],
            values=[
                spam_count,
                ham_count
            ],
            title="Spam vs Legitimate Messages"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        chart_df = pd.DataFrame({
            "Category": ["Spam", "Legitimate"],
            "Count": [spam_count, ham_count]
        })
        fig2 = px.bar(
            chart_df,
            x="Category",
            y="Count",
            title="Prediction Distribution"
        )
        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    else:

        st.info(
            "No prediction data available yet."
        )


# ==========================================================
# SINGLE SMS PREDICTION
# ==========================================================

elif selected == "Single Prediction":

    st.title("📩 Single SMS Prediction")

    message = st.text_area(
        "Enter SMS Message",
        height=180
    )

    if st.button("Detect Spam"):

        if message.strip() == "":

            st.warning(
                "Please enter a message."
            )

        else:

            transformed_message = (
                vectorizer.transform(
                    [message]
                )
            )

            prediction = model.predict(
                transformed_message
            )

            probability = (
                model.predict_proba(
                    transformed_message
                )
            )

            spam_score = (
                probability[0][1]
            )

            if prediction[0] == 1:

                st.error(
                    "⚠️ SPAM MESSAGE"
                )

                result = "Spam"

            else:

                st.success(
                    "✅ LEGITIMATE MESSAGE"
                )

                result = "Legitimate"

            st.metric(
                "Spam Probability",
                f"{spam_score*100:.2f}%"
            )

            st.progress(
                float(spam_score)
            )
            if spam_score < 0.30:
                st.success("🟢 Low Risk")
            elif spam_score < 0.70:
                st.warning("🟠 Moderate Risk")
            else:
                st.error("🔴 High Risk")

            save_history(
                message,
                result,
                round(spam_score * 100, 2)
            )


# ==========================================================
# BULK SMS PREDICTION
# ==========================================================

elif selected == "Bulk Prediction":

    st.title("📂 Bulk SMS Detection")

    st.info(
        "Upload a CSV file containing a column named 'Message'"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # Read CSV safely
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8",
                engine="python",
                on_bad_lines="skip"
            )

        except Exception as e:

            st.error(
                f"Error reading CSV file: {e}"
            )

            st.stop()

        # Debug Information
        st.subheader("Detected Columns")

        st.write(
            df.columns.tolist()
        )

        st.subheader(
            "Uploaded Data Preview"
        )

        st.dataframe(
            df.head()
        )

        # Check required column
        if "Message" not in df.columns:

            st.error(
                """
                CSV must contain a column named:

                Message
                """
            )

        else:

            try:

                # Transform messages
                transformed = vectorizer.transform(
                    df["Message"].astype(str)
                )

                # Prediction
                predictions = model.predict(
                    transformed
                )

                # Probability Scores
                probabilities = model.predict_proba(
                    transformed
                )

                # Spam Probability %
                spam_scores = (
                    probabilities[:, 1] * 100
                ).round(2)

                # Add Results
                df["Prediction"] = [

                    "Spam"
                    if pred == 1
                    else "Legitimate"

                    for pred in predictions
                ]

                df["Spam Probability (%)"] = (
                    spam_scores
                )

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                # Summary Metrics
                spam_count = len(
                    df[
                        df["Prediction"]
                        == "Spam"
                    ]
                )

                ham_count = len(
                    df[
                        df["Prediction"]
                        == "Legitimate"
                    ]
                )

                total_count = len(df)

                st.divider()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Messages",
                    total_count
                )

                col2.metric(
                    "Spam Messages",
                    spam_count
                )

                col3.metric(
                    "Legitimate Messages",
                    ham_count
                )

                # Download CSV
                csv = df.to_csv(
                    index=False
                )

                st.download_button(
                    label="📥 Download Report",
                    data=csv,
                    file_name="spam_report.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Prediction Error: {e}"
                )

# ==========================================================
# HISTORY PAGE
# ==========================================================

elif selected == "History":

    st.title("📜 Prediction History")

    if os.path.exists(
        "prediction_history.csv"
    ):

        history = pd.read_csv(
            "prediction_history.csv"
        )

        search = st.text_input(
            "🔍 Search Messages"
        )

        if search:

            history = history[
                history["Message"]
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            history,
            use_container_width=True
        )

        csv = history.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

    else:

        st.warning(
            "No history found."
        )

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif selected == "About":

    st.title(
        "ℹ️ About Project"
    )

    st.markdown(
        """
        ## SMS Spam Detection Tool

        This project uses Machine Learning to classify SMS messages as:

        - Spam
        - Legitimate

        ### Technologies Used

        - Python
        - Streamlit
        - Scikit-Learn
        - TF-IDF Vectorizer
        - Logistic Regression
        - Plotly

        ### Features

        ✅ Real-Time SMS Prediction

        ✅ Confidence Score

        ✅ Bulk CSV Prediction

        ✅ Downloadable Reports

        ✅ Prediction History

        ✅ Analytics Dashboard

        ### Developer

        Azizur Rahman

        M.Sc Cyber Security
        Aligarh Muslim University
        """
    )