import streamlit as st
import os

st.set_page_config(page_title="AI Music Generator", page_icon="🎵")

st.title("🎵 AI Music Generator")
st.write("Generate music using an LSTM AI model trained on MIDI files.")

st.sidebar.header("🎼 Options")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["Classical", "Piano", "Melody"]
)

st.write("Selected Genre:", genre)

st.subheader("▶️ Generated Music")

if os.path.exists("generated_music.mid"):
    with open("generated_music.mid", "rb") as file:
        st.download_button(
            label="Download Generated Music",
            data=file,
            file_name="generated_music.mid",
            mime="audio/midi"
        )

    st.success("Generated music file found!")
else:
    st.warning("No generated music found. Run generate.py first.")

st.subheader("📊 Training Loss Graph")

if os.path.exists("loss.png"):
    st.image("loss.png", caption="Training Loss Graph")
else:
    st.info("Training loss graph will appear here after adding loss tracking.")