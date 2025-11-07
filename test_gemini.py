import google.generativeai as genai
import streamlit as st

# Konfigurasi API key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Pilih model yang stabil (bukan -latest)
model = genai.GenerativeModel("gemini-1.5-flash")

# Tes chat sederhana
try:
    response = model.generate_content(["Hai, aku Yilita! Apa kabar?"])
    st.write(response.text)
except Exception as e:
    st.error(f"Terjadi error: {e}")
