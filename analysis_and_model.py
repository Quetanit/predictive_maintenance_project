import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(data):
    # Удаление ненужных столбцов
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')

    # Преобразование категориальной переменной
    data['Type'] = LabelEncoder().fit_transform(data['Type'])

    # Масштабирование числовых признаков
    scaler = StandardScaler()
    num_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data[num_features] = scaler.fit_transform(data[num_features])

    return data


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.write("**Classification Report:**")
    st.text(class_report)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)


def analysis_and_model_page():
    st.title("Анализ данных и обучение модели")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = preprocess_data(data)

        # Разделение на признаки и цель
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        st.header("Результаты модели")
        evaluate_model(model, X_test, y_test)

        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            product_type = st.selectbox("Тип продукта (L=0, M=1, H=2)", ['L', 'M', 'H'])
            air_temp = st.number_input("Температура воздуха [K]", value=300.0)
            proc_temp = st.number_input("Процессная температура [K]", value=310.0)
            rot_speed = st.number_input("Скорость вращения [rpm]", value=1500)
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", value=0)

            submit = st.form_submit_button("Предсказать")

            if submit:
                type_map = {'L': 0, 'M': 1, 'H': 2}
                input_df = pd.DataFrame([{
                    'Type': type_map[product_type],
                    'Air temperature [K]': air_temp,
                    'Process temperature [K]': proc_temp,
                    'Rotational speed [rpm]': rot_speed,
                    'Torque [Nm]': torque,
                    'Tool wear [min]': tool_wear
                }])

                scaler = StandardScaler()
                full_data = pd.concat([X, input_df], axis=0)
                full_data_scaled = scaler.fit_transform(full_data)
                input_scaled = full_data_scaled[-1].reshape(1, -1)

                pred = model.predict(input_scaled)
                proba = model.predict_proba(input_scaled)[0, 1]

                st.write(f"**Предсказание:** {'Отказ (1)' if pred[0] == 1 else 'Нет отказа (0)'}")
                st.write(f"**Вероятность отказа:** {proba:.2f}")

