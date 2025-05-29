import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите страницу:", ["Анализ и модель", "Презентация"])

if page == "Анализ и модель":
    analysis_and_model_page()
elif page == "Презентация":
    presentation_page()

