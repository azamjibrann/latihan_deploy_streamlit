import joblib 
import streamlit as st

model = joblib.load('model/model_logistic_regression.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

st.title("Aplikasi Klasifikasi Komentar")
st.write("Aplikasi Ini dibuat menggunakan Tekonologi NLP ")
input = st.text_input("Masukan Komentar Anda !")
if st.button("SUBMIT"):
    if input.strip() == "":
        st.warning("Komentar Tidak Boleh Kosong")
    else:
        vector = tfidf.transform([input])
        prediksi = model.predict(vector)[0]

        label_map = {
            0: "Negatif",
            1: "Positif"
        }
        st.subheader("Hasil Analisis Komentar")
        st.write("**Komentar :**", label_map.get(prediksi,prediksi))