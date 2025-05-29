import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    # Презентация в формате Markdown
    presentation_markdown = """
    # Бинарная классификация для предиктивного обслуживания оборудования
    ---
    ## Цель проекта
    - Построить модель машинного обучения
    - Предсказывать отказ оборудования (Target = 1) или его отсутствие (Target = 0)
    ---
    ## Датасет
    - AI4I 2020 Predictive Maintenance Dataset
    - 10 000 записей, 14 признаков
    - Типы отказов: износ, перегрев, перегрузка и др.
    ---
    ## Этапы работы
    1. Загрузка и предобработка данных
    2. Разделение на обучающую и тестовую выборки
    3. Обучение модели (Logistic Regression)
    4. Оценка: Accuracy, Confusion Matrix, ROC-AUC
    ---
    ## Streamlit-приложение
    - Основная страница: загрузка данных, обучение модели, предсказания
    - Страница презентации: текущий слайд-шоу
    ---
    ## Выводы и улучшения
    - Модель показывает хорошие результаты
    - Возможные улучшения: другие модели (XGBoost), балансировка классов, расширение признаков
    """

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["white", "black", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins
        },
        markdown_props={"data-separator-vertical": "^--$"}
    )

