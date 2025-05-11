import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from io import BytesIO

st.set_page_config(page_title="Regresi Linier", layout="centered")
st.title("Regresi Linier")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview Data:")
    st.dataframe(df.head())

    try:
        columns = df.columns.tolist()
        x_col = st.selectbox("Pilih kolom sumbu X:", columns)
        y_col = st.selectbox("Pilih kolom sumbu Y:", columns)

        # Model
        X = df[[x_col]].values
        y = df[y_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Koefisien regresi
        a = model.coef_[0]
        b = model.intercept_

        # Evaluasi
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Hasil Model Regresi")
        st.write(f"Persamaan regresi: **y = {a:.4f} * x + {b:.4f}**")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Plot interaktif pakai Plotly
        st.subheader("Plot Interaktif Regresi")
        fig = go.Figure()

        # Titik data asli
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col],
                                 mode='markers', name='Data Asli'))

        # Garis regresi
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        y_line = a * x_range + b
        fig.add_trace(go.Scatter(x=x_range, y=y_line,
                                 mode='lines', name='Regresi Linier', line=dict(color='red')))

        fig.update_layout(title="Regresi Linier",
                          xaxis_title=x_col,
                          yaxis_title=y_col)

        st.plotly_chart(fig)

        # Prediksi nilai baru
        st.subheader("Prediksi Nilai Baru")
        input_x = st.number_input("Masukkan nilai x", value=0.0)
        if st.button("Prediksi y"):
            pred_y = model.predict([[input_x]])[0]
            st.success(f"Prediksi y: {pred_y:.2f}")

        # Tambahkan kolom prediksi ke data
        df['y_pred'] = model.predict(X)

        # Download CSV hasil prediksi
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="💾 Download CSV dengan hasil prediksi",
            data=csv_buffer,
            file_name="hasil_regresi.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

else:
    st.info("Silakan upload file CSV.")
