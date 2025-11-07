import os
import json
import joblib
import difflib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any
from sklearn.linear_model import LinearRegression

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="🚘 Prediksi Harga Mobil + Chat Gemini",
    layout="wide",
    page_icon="🚗",
)

# =========================
# CSS KUSTOM
# =========================
st.markdown("""
<style>
:root {
    --primary-color: #00b4d8;
    --secondary-color: #0077b6;
    --bg-dark: #0e1117;
    --text-light: #f1f1f1;
    --shadow: 0 0 25px rgba(0,180,216,0.3);
}
body, .stApp {
    background-color: var(--bg-dark);
    color: var(--text-light);
    font-family: "Poppins", sans-serif;
}
.main-header {
    font-size: 2.4em;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(270deg, #00b4d8, #48cae4, #0077b6);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 8s ease infinite;
    margin-bottom: 25px;
}
@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.chat-container {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(0,180,216,0.15);
}
.chat-bubble-user {
    background: linear-gradient(135deg, #0077b6, #0096c7);
    color: white;
    border-radius: 20px 20px 0 20px;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
    word-wrap: break-word;
}
.chat-bubble-ai {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px 20px 20px 0;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚘 Prediksi Harga Mobil + Chat Gemini AI</div>", unsafe_allow_html=True)

# =========================
# UTIL
# =========================
def rupiah(x: float) -> str:
    try:
        return f"Rp {x:,.0f}".replace(",", ".")
    except Exception:
        return "Rp -"

@st.cache_resource
def load_model_if_exists(path: str):
    """Memuat model jika tersedia, atau buat dummy model jika gagal."""
    if os.path.exists(path):
        try:
            st.sidebar.info("✅ Model ditemukan dan dimuat dari disk.")
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"⚠ Gagal memuat model.pkl: {e}")
            st.sidebar.warning("🔄 Menggunakan model dummy sementara.")
            return LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))
    else:
        st.sidebar.warning("⚠ model.pkl tidak ditemukan. Menggunakan model dummy.")
        return LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))

def prepare_input(form_data: Dict[str, Any], example_schema: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{k: form_data.get(k, v) for k, v in example_schema.items()}])

def save_to_dataset(input_df: pd.DataFrame, prediction: float, path: str):
    df_to_save = input_df.copy()
    df_to_save["prediction"] = prediction
    if os.path.exists(path):
        try:
            dataset = pd.read_csv(path)
        except Exception:
            dataset = pd.DataFrame()
        dataset = pd.concat([dataset, df_to_save], ignore_index=True)
    else:
        dataset = df_to_save
    dataset.to_csv(path, index=False)
    return dataset

# =========================
# KONFIG GEMINI (ENV FIRST)
# =========================
gemini_api_key_env = os.environ.get("GEMINI_API_KEY", "")
try:
    import google.generativeai as genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

with st.sidebar:
    st.subheader("🤖 Gemini")
    if not HAVE_GENAI:
        st.error("Paket google-generativeai belum terpasang.\nJalankan:\n`python -m pip install google-generativeai`")
    key_status = "TERDETEKSI" if gemini_api_key_env else "TIDAK TERDETEKSI"
    st.write(f"GEMINI_API_KEY: *{key_status}*")
    manual_key = "" if gemini_api_key_env else st.text_input("API Key (opsional, tidak disimpan)", type="password")
    model_name = st.selectbox("Model LLM", ["gemini-2.5-flash","gemini-flash-latest","gemini-2.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.6, 0.1)
    max_output_tokens = st.slider("Max Output Tokens", 256, 4096, 1024, 64)
    answer_all = st.toggle("🟦 Gemini jawab SEMUA pertanyaan (disarankan)", value=True)
    list_models = st.toggle("Tampilkan daftar model yang mendukung generateContent", value=False)

def _get_gemini_client(api_key: str):
    if not HAVE_GENAI:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception:
        return None

def _active_api_key() -> str:
    return gemini_api_key_env or manual_key

def _dataset_summary(df: pd.DataFrame, max_models: int = 20) -> str:
    if df is None or df.empty:
        return "Dataset kosong/tidak dimuat."
    cols = ", ".join(df.columns.tolist()[:20])
    summary = f"Kolom: {cols}."
    if "model" in df.columns:
        uniq = sorted(df['model'].dropna().astype(str).unique().tolist())
        if uniq:
            summary += f" Contoh model: {', '.join(uniq[:max_models])}."
    if "price" in df.columns:
        try:
            mean_price = float(df['price'].mean())
            summary += f" Rata-rata price (unit asli dataset): {mean_price:,.2f}."
        except Exception:
            pass
    return summary

def gemini_chat(
    user_message: str,
    history: list[tuple[str, str]] | None = None,
    df: pd.DataFrame | None = None,
    last_prediction: float | None = None,
    last_input: dict | None = None,
) -> str:
    """
    Chat generik: arahkan SEMUA pertanyaan ke Gemini.
    Sertakan konteks ringan: ringkasan dataset, prediksi terakhir (jika ada), dan riwayat chat.
    """
    api_key = _active_api_key()
    if not api_key:
        return "⚠ Gemini belum aktif karena API Key tidak terpasang. Set GEMINI_API_KEY di environment atau isi di sidebar."

    client = _get_gemini_client(api_key)
    if client is None:
        return "⚠ Gagal menginisialisasi klien Gemini. Periksa API key atau koneksi."

    # Siapkan konteks
    ds_ctx = _dataset_summary(df)
    last_ctx = ""
    if last_prediction is not None and last_input:
        try:
            last_ctx = f"Prediksi terakhir: {json.dumps(last_input)} → £{last_prediction:,.2f} (≈ {rupiah(last_prediction*20000)})."
        except Exception:
            pass

    chat_history_text = ""
    if history:
        # gabungkan 6 pesan terakhir agar konteks tidak terlalu panjang
        tail = history[-6:]
        lines = []
        for role, msg in tail:
            r = "User" if role == "user" else "Assistant"
            lines.append(f"{r}: {msg}")
        chat_history_text = "\n".join(lines)

    system_prompt = f"""
Kamu adalah asisten AI serbaguna untuk pengguna Indonesia.
Tugasmu: jawab pertanyaan apapun secara jelas, ringkas, dan akurat.
Jika pertanyaan tentang data mobil, kamu boleh memanfaatkan ringkasan dataset di bawah.
Jika bukan tentang mobil, jawab seperti asisten umum (tanpa bergantung dataset).
Gunakan bahasa Indonesia yang natural. Sertakan langkah atau contoh jika relevan.
Hindari spekulasi berlebihan — jika ragu, jelaskan asumsi atau minta klarifikasi singkat.

KONTEKS DATASET (opsional):
{ds_ctx}

KONTEKS PREDIKSI TERAKHIR (opsional):
{last_ctx}

RIWAYAT CHAT (ringkas):
{chat_history_text}
""".strip()

    try:
        # Gunakan satu prompt gabungan
        prompt = f"{system_prompt}\n\nPERTANYAAN PENGGUNA:\n{user_message}\n"
        resp = client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens
            },
        )
        text = (getattr(resp, "text", None) or "").strip()
        return text or "Maaf, aku tidak menerima teks balasan dari Gemini."
    except Exception as e:
        return f"⚠ Gagal memanggil Gemini API: {e}"

def local_chat_response(user_message: str, last_prediction=None, last_input=None, df=None):
    """
    Fallback lokal jika Gemini mati. (Tetap dipertahankan)
    """
    msg = (user_message or "").lower().strip()

    if msg in ["model", "daftar model", "model mobil"]:
        if df is not None and not df.empty and "model" in df.columns:
            all_models = sorted(df["model"].dropna().astype(str).unique().tolist())
            return "🚗 Jenis mobil yang tersedia:\n" + ", ".join(all_models[:50])
        else:
            return "Dataset belum dimuat atau kolom 'model' tidak ditemukan."

    if "harga" in msg:
        if df is not None and not df.empty and "model" in df.columns and "price" in df.columns:
            models = df["model"].dropna().astype(str).unique().tolist()
            best_match = difflib.get_close_matches(msg, [m.lower() for m in models], n=1, cutoff=0.6)
            if best_match:
                found = best_match[0]
                matched_model = next((m for m in models if m.lower() == found), found)
                avg_price = df[df["model"].astype(str).str.lower() == matched_model.lower()]["price"].mean()
                if pd.notna(avg_price):
                    harga_rupiah = avg_price * 20000
                    return f"💰 Harga rata-rata {matched_model} sekitar {rupiah(harga_rupiah)} (≈ £{avg_price:,.2f})."
                else:
                    return f"Harga {matched_model} belum tersedia di dataset."
            else:
                # fallback generik
                return "Untuk harga model itu belum ada di dataset. Coba sebutkan detail spesifik (tahun, transmisi, kilometer)."
        else:
            return "Dataset tidak tersedia. Kamu bisa tetap bertanya hal lain kok 🙂"

    if any(x in msg for x in ["halo", "hai", "hi"]):
        return "Halo! 👋 Aku siap bantu. Tanyakan apa saja."

    return "Aku siap jawab pertanyaanmu. Jika tentang mobil, sebutkan model/tahun untuk estimasi."

# =========================
# PATH & DATA
# =========================
BASE_DIR = os.getcwd()
MODEL_PATH = "model.pkl"
DATASET_PATH = "toyota.csv"
EXAMPLE_PATH = "example_schema.json"

# ===== Sidebar: status & uploader =====
with st.sidebar:
    st.subheader("📁 Berkas")
    st.write(f"Folder kerja: {BASE_DIR}")
    st.write(f"model.pkl: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")
    st.write(f"toyota.csv: {'✅' if os.path.exists(DATASET_PATH) else '❌'}")
    st.write("---")
    up_model = st.file_uploader("Upload model.pkl (opsional)", type=["pkl"])
    if up_model is not None:
        try:
            with open(MODEL_PATH, "wb") as f:
                f.write(up_model.read())
            st.success("model.pkl tersimpan.")
        except Exception as e:
            st.error(f"Gagal menyimpan model: {e}")

    up_csv = st.file_uploader("Upload toyota.csv (opsional)", type=["csv"])
    if up_csv is not None:
        try:
            df_tmp = pd.read_csv(up_csv)
            df_tmp.to_csv(DATASET_PATH, index=False)
            st.success("toyota.csv tersimpan.")
        except Exception as e:
            st.error(f"Gagal menyimpan dataset: {e}")

    st.divider()
    if list_models and HAVE_GENAI:
        try:
            genai.configure(api_key=_active_api_key())
            names = []
            for m in genai.list_models():
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    names.append(m.name)
            if names:
                st.caption("Model yang mendukung generateContent:")
                st.code("\n".join(names), language="text")
        except Exception as e:
            st.warning(f"List model gagal: {e}")

# Load model
model = load_model_if_exists(MODEL_PATH)

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    st.sidebar.success(f"📊 Dataset dimuat ({len(df)} baris)")
except Exception as e:
    st.sidebar.warning(f"⚠ Gagal memuat dataset: {e}")
    df = pd.DataFrame()

# Load example schema (atau fallback)
try:
    with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
        example_schema = json.load(f)
except Exception:
    example_schema = {
        "model": "Avanza", "year": 2020, "transmission": "Manual",
        "mileage": 15000, "fuelType": "Bensin", "tax": 1500000,
        "mpg": 14.5, "engineSize": 1.3
    }

# =========================
# ANTARMUKA
# =========================
left, right = st.columns([1.2, 1])

# =========================
# CHAT GEMINI (TENGAH)
# =========================
st.markdown("""
<style>
.center-chat {
    display: flex;
    justify-content: center;
}
.chat-box {
    width: 85%;
    max-width: 700px;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='center-chat'><div class='chat-box'>", unsafe_allow_html=True)
    st.subheader("💬 Chat Asisten (Gemini-first)")

    # Kontrol chat
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Reset Chat"):
            st.session_state["chat_history"] = []
            st.toast("Chat direset.")
    with c2:
        st.caption("Tips: tanya apa saja. Aku akan jawab pakai Gemini.")
    with c3:
        mode_label = "Gemini-first ✅" if answer_all else "Fallback Lokal"
        st.caption(mode_label)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Tampilan chat
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        bubble = "chat-bubble-user" if role == "user" else "chat-bubble-ai"
        st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input chat
    user_msg = st.text_input("Ketik pertanyaan di sini...", key="chat_input")
    if st.button("Kirim 💬"):
        if user_msg.strip():
            st.session_state.chat_history.append(("user", user_msg))
            if answer_all:
                response = gemini_chat(
                    user_msg,
                    history=st.session_state.get("chat_history"),
                    df=df,
                    last_prediction=st.session_state.get("last_prediction"),
                    last_input=st.session_state.get("last_input"),
                )
            else:
                response = local_chat_response(
                    user_msg,
                    st.session_state.get("last_prediction"),
                    st.session_state.get("last_input"),
                    df
                )
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("✨ Aplikasi prediksi harga mobil + chat Gemini AI — Mode Gemini-first untuk menjawab semua pertanyaan ✨")