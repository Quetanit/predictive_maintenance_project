import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier


def preprocess_data(data):
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    scaler = StandardScaler()
    num_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                    'Tool wear [min]']
    data[num_features] = scaler.fit_transform(data[num_features])
    return data


def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Логистическая регрессия": LogisticRegression(max_iter=1000),
        "Случайный лес": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append({
            "name": name,
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba
        })
    return results


def plot_metrics(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        ax.plot([r["name"] for r in results], [r[metric] for r in results], marker='o', label=metric)
    ax.set_ylabel("Score")
    ax.set_title("Сравнение метрик моделей")
    ax.legend()
    st.pyplot(fig)


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{res['name']} (AUC={res['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые моделей")
    ax.legend()
    st.pyplot(fig)


def plot_precision_recall_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for res in results:
        precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(recall, precision, label=res['name'])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall кривые")
    ax.legend()
    st.pyplot(fig)


def plot_confusion_matrix(y_test, y_pred, model_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    st.pyplot(fig)


def plot_feature_importance(model, X, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = X.columns
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(importances)), importances[indices], align='center')
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(feat_names[indices], rotation=45, ha='right')
        ax.set_title(f"Feature Importance: {model_name}")
        st.pyplot(fig)


def analysis_and_model_page():
    st.title("Анализ данных и сравнение моделей")
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = preprocess_data(data)
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = evaluate_models(X_train, X_test, y_train, y_test)

        # Визуализация сравнения метрик
        plot_metrics(results)
        plot_roc_curves(results, y_test)
        plot_precision_recall_curves(results, y_test)

        # Выбор лучшей модели по F1-score
        best = max(results, key=lambda x: x['f1'])
        st.success(f"**Лучшая модель:** {best['name']} (F1-score={best['f1']:.3f}, ROC-AUC={best['roc_auc']:.3f})")

        # Confusion matrix и feature importance для лучшей модели
        plot_confusion_matrix(y_test, best["y_pred"], best["name"])
        plot_feature_importance(best["model"], X, best["name"])

        # Предсказание по новым данным
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
                pred = best["model"].predict(input_scaled)
                proba = best["model"].predict_proba(input_scaled)[0, 1]
                st.write(f"**Предсказание:** {'Отказ (1)' if pred[0] == 1 else 'Нет отказа (0)'}")
                st.write(f"**Вероятность отказа:** {proba:.2f}")
