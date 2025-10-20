import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta

st.set_page_config(page_title="Roblox Accessory Price Optimizer", layout="wide")
st.title("🎩 Roblox Accessory Price Optimizer")
st.write("Загрузи CSV-файлы с продажами аксессуаров. Приложение вычислит оптимальные цены, покажет графики и динамику продаж.")

uploaded_files = st.file_uploader(
    "Загрузите CSV файлы (один или несколько):",
    type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Нормализуем имена колонок
    data.columns = data.columns.str.strip()
    required_cols = {"Asset Name", "Asset Type", "Price", "Revenue", "Date and Time"}
    if not required_cols.issubset(data.columns):
        st.error(f"Требуются колонки: {', '.join(required_cols)}. Найдено: {', '.join(data.columns)}")
        st.stop()

    # Конвертируем типы
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["Asset Name", "Price", "Revenue", "Asset Type", "Date and Time"])
    data = data[data["Revenue"] > 0]

    # Конвертируем дату
    data["Date and Time"] = pd.to_datetime(data["Date and Time"], errors="coerce")
    today = datetime.now().date()
    data = data[data["Date and Time"].dt.date < today]

    grouped = data.groupby(["Asset Name", "Asset Type", "Price"], as_index=False)["Revenue"].mean()

    def revenue_model(x, a, b, c):
        return a * x**2 + b * x + c

    results = []
    for (name, atype), df_item in grouped.groupby(["Asset Name", "Asset Type"]):
        df_item = df_item.sort_values("Price")
        min_p, max_p = df_item["Price"].min(), df_item["Price"].max()
        num_points = len(df_item)

        if num_points < 3:
            optimal_price = None
            message = "Мало тестов, добавьте новые цены"
        else:
            try:
                x, y = df_item["Price"].values, df_item["Revenue"].values
                params, _ = curve_fit(revenue_model, x, y, maxfev=5000)
                a, b, c = params
                vertex = -b / (2 * a) if a != 0 else None
                if vertex is not None and a < 0:
                    optimal_price = float(np.clip(vertex, min_p, max_p))
                    message = ""
                else:
                    optimal_price = None
                    message = f"Парабола вверх. Рекомендуем протестировать цены выше {max_p:.2f}"
            except:
                optimal_price = None
                message = "Ошибка аппроксимации. Используйте наблюдаемые данные"
        max_rev = float(df_item["Revenue"].max())
        results.append({
            "Accessory": name,
            "Type": atype,
            "Optimal Price": optimal_price,
            "Predicted Max Revenue": max_rev,
            "Min Tested Price": min_p,
            "Max Tested Price": max_p,
            "Data Points": num_points,
            "Message": message
        })

    result_df = pd.DataFrame(results).sort_values(["Type", "Accessory"])
    st.subheader("📊 Сводная таблица аксессуаров")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("🔍 Выберите аксессуар для графиков")
    search = st.text_input("Поиск по названию:")
    filtered = result_df.copy()
    if search.strip():
        filtered = filtered[filtered["Accessory"].str.contains(search.strip(), case=False, na=False)]
    options = filtered.apply(lambda row: f"{row['Type']}: {row['Accessory']}", axis=1).tolist()
    selected = st.selectbox("Аксессуар:", options)

    if selected:
        sel_type, sel_name = selected.split(": ")
        df_plot = grouped[(grouped["Asset Name"] == sel_name) & (grouped["Asset Type"] == sel_type)].sort_values("Price")

        # График Revenue vs Price
        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.scatter(df_plot["Price"], df_plot["Revenue"], label="Данные")
        if len(df_plot) >=3:
            try:
                params, _ = curve_fit(revenue_model, df_plot["Price"], df_plot["Revenue"], maxfev=5000)
                x_dense = np.linspace(df_plot["Price"].min(), df_plot["Price"].max(), 200)
                y_dense = revenue_model(x_dense, *params)
                ax1.plot(x_dense, y_dense, color="red", label="Парабола")
                vertex = -params[1]/(2*params[0])
                if params[0]<0:
                    ax1.axvline(vertex, color="green", linestyle="--", label=f"Оптимальная цена: {vertex:.2f}")
            except:
                pass
        ax1.set_xlabel("Цена")
        ax1.set_ylabel("Средний дневной Revenue")
        ax1.legend()
        st.pyplot(fig1)

        # График Revenue по дате
        df_time = data[(data["Asset Name"]==sel_name) & (data["Asset Type"]==sel_type)]
        fig2, ax2 = plt.subplots(figsize=(8,4))
        daily = df_time.groupby(df_time["Date and Time"].dt.date)["Revenue"].sum().reset_index()
        ax2.plot(daily["Date and Time"], daily["Revenue"], marker="o")
        ax2.set_xlabel("Дата")
        ax2.set_ylabel("Суммарное Revenue")
        ax2.set_title("Динамика Revenue по дате")
        st.pyplot(fig2)
