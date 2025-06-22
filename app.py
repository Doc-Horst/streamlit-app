import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🧬 BioData Tutor")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Vorschau der Daten:", df.head())

    x_col = st.selectbox("Spalte für X-Achse", df.columns)
    y_col = st.selectbox("Spalte für Y-Achse", df.columns)

    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col], marker='o')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Diagramm")
    st.pyplot(fig)

    stats = df[[x_col, y_col]].describe()
    st.write("📈 Statistische Kennwerte:", stats)

    st.markdown("🧠 GPT-Analyse folgt...")
    prompt = f"""
    Erkläre die folgenden Versuchsdaten (X={x_col}, Y={y_col}):

    {df[[x_col, y_col]].to_string(index=False)}

    Welche biologischen oder physikalischen Trends ergeben sich?
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein Tutor für Biotechnologie und Physik."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=600
    )

    st.markdown(response["choices"][0]["message"]["content"])
