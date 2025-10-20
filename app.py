import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Roblox Accessory Price Optimizer", layout="wide")

st.title("🎩 Roblox Accessory Price Optimizer")
st.write("Загрузи несколько CSV-файлов с продажами аксессуаров — инструмент сам определит оптимальные цены и покажет графики.")

uploaded_files = st.file_uploader("Загрузи CSV файлы (один или несколько):", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Объединяем все файлы
    dfs = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Проверяем нужные колонки
    required_cols = {"Asset Name", "Price", "Revenue", "Date"}
    if not required_cols.issubset(data.columns):
        st.error(f"В таблице должны быть колонки: {', '.join(required_cols)}")
        st.stop()

    # Чистим данные
    data = data[data["Revenue"] > 0]
    data = data.dropna(subset=["Asset Name", "Price", "Revenue"])

    # Приводим типы
    data["Price"] = data["Price"].astype(float)
    data["Revenue"] = data["Revenue"].astype(float)

    # Группируем
    grouped = data.groupby(["Asset Name", "Price"], as_index=False)["Revenue"].mean()

    st.subheader("📊 Результаты анализа")

    # Для аппроксимации возьмем квадратичную модель
    def revenue_model(x, a, b, c):
        return a * x**2 + b * x + c

    results = []

    for name, df_item in grouped.groupby("Asset Name"):
        if len(df_item) < 3:
            # Недостаточно данных для аппроксимации
            optimal_price = df_item.loc[df_item["Revenue"].idxmax(), "Price"]
            max_rev = df_item["Revenue"].max()
        else:
            try:
                x = df_item["Price"].values
                y = df_item["Revenue"].values
                params, _ = curve_fit(revenue_model, x, y)
                a, b, c = params
                # Оптимальная точка вершины параболы
                optimal_price = -b / (2 * a)
                max_rev = revenue_model(optimal_price, *params)
            except:
                optimal_price = df_item.loc[df_item["Revenue"].idxmax(), "Price"]
                max_rev = df_item["Revenue"].max()

        results.append({
            "Accessory": name,
            "Optimal Price": round(optimal_price, 2),
            "Max Revenue": round(max_rev, 2),
            "Min Tested Price": df_item["Price"].min(),
            "Max Tested Price": df_item["Price"].max()
        })

    result_df = pd.DataFrame(results).sort_values("Max Revenue", ascending=False)
    st.dataframe(result_df, use_container_width=True)

    # Графики
    st.subheader("📈 Графики по аксессуарам")

    selected = st.selectbox("Выбери аксессуар для просмотра графика:", result_df["Accessory"])
    df_plot = grouped[grouped["Asset Name"] == selected]

    # Построение графика
    fig, ax = plt.subplots()
    ax.scatter(df_plot["Price"], df_plot["Revenue"], label="Данные", color="blue")

    if len(df_plot) >= 3:
        x = np.linspace(df_plot["Price"].min(), df_plot["Price"].max(), 100)
        try:
            params, _ = curve_fit(revenue_model, df_plot["Price"], df_plot["Revenue"])
            y = revenue_model(x, *params)
            ax.plot(x, y, color="red", label="Модель (аппроксимация)")
        except:
            pass

    opt_price = result_df[result_df["Accessory"] == selected]["Optimal Price"].values[0]
    ax.axvline(opt_price, color="green", linestyle="--", label=f"Оптимальная цена: {opt_price}")

    ax.set_xlabel("Цена")
    ax.set_ylabel("Средний дневной Revenue")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Готово! Оптимальные цены вычислены.")
