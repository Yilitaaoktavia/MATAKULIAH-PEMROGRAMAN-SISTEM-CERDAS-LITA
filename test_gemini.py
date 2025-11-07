import google.generativeai as genai

# Masukkan API key yang valid dari Google AI Studio
genai.configure(api_key="MASUKKAN_API_KEY_KAMU_DI_SINI")

# Gunakan model yang tersedia dan stabil
model = genai.GenerativeModel("gemini-1.5-flash")

# Coba generate teks
response = model.generate_content("Hai, aku Yilita! Apa kabar?")

# Tampilkan hasil
print(response.text)