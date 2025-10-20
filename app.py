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
    # Read and concat files robustly
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Не удалось прочитать {getattr(file, 'name', 'файл')}: {e}")
            st.stop()
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Normalize column names (trim whitespace)
    data.columns = data.columns.str.strip()

    # Required columns (use Date and Time)
    required_cols = {"Asset Name", "Price", "Revenue", "Date and Time"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"В таблице должны быть колонки: {', '.join(required_cols)}. "
                 f"Найдено: {', '.join(data.columns)}")
        st.stop()

    # Clean data
    # Remove zero revenue rows (bug)
    data = data[pd.to_numeric(data["Revenue"], errors="coerce").notnull()]
    data["Revenue"] = data["Revenue"].astype(float)
    data = data[data["Revenue"] > 0]

    # Drop rows with missing key fields
    data = data.dropna(subset=["Asset Name", "Price", "Revenue"])

    # Ensure Price numeric
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["Price"])
    data["Price"] = data["Price"].astype(float)

    # (Optional) parse Date and Time if needed
    try:
        data["Date and Time"] = pd.to_datetime(data["Date and Time"], errors="coerce")
    except:
        pass

    # Group by asset and price — take mean daily revenue per price (if you prefer sum, change here)
    grouped = data.groupby(["Asset Name", "Price"], as_index=False)["Revenue"].mean()

    st.subheader("📊 Результаты анализа")

    # Quadratic model for revenue vs price
    def revenue_model(x, a, b, c):
        return a * x**2 + b * x + c

    results = []

    for name, df_item in grouped.groupby("Asset Name"):
        df_item = df_item.sort_values("Price")
        min_p = df_item["Price"].min()
        max_p = df_item["Price"].max()

        if len(df_item) < 3:
            # Недостаточно точек для аппроксимации — возьмём наблюдаемую лучшую цену
            best_idx = df_item["Revenue"].idxmax()
            optimal_price = float(df_item.loc[best_idx, "Price"])
            max_rev = float(df_item.loc[best_idx, "Revenue"])
            method = "observed_max"
        else:
            x = df_item["Price"].values
            y = df_item["Revenue"].values
            try:
                params, _ = curve_fit(revenue_model, x, y, maxfev=5000)
                a, b, c = params
                # вершина параболы
                vertex = -b / (2 * a) if a != 0 else None

                if vertex is not None and a < 0:
                    # парабола перевёрнутая — вершина максимум -> используем её
                    optimal_price = float(vertex)
                    max_rev = float(revenue_model(optimal_price, *params))
                    method = "vertex"
                else:
                    # парабола не подходит (a >= 0) или degenerate — ограничимся исследованием в тестированном диапазоне
                    xs = np.linspace(min_p, max_p, 200)
                    ys = revenue_model(xs, *params)
                    idx = int(np.nanargmax(ys))
                    optimal_price = float(xs[idx])
                    max_rev = float(ys[idx])
                    method = "fitted_bounded"
                # If optimal is nonsense (nan/inf), fallback to observed max
                if not np.isfinite(optimal_price) or not np.isfinite(max_rev):
                    best_idx = df_item["Revenue"].idxmax()
                    optimal_price = float(df_item.loc[best_idx, "Price"])
                    max_rev = float(df_item.loc[best_idx, "Revenue"])
                    method = "fallback_observed"
            except Exception:
                # Fit failed -> fallback to observed best
                best_idx = df_item["Revenue"].idxmax()
                optimal_price = float(df_item.loc[best_idx, "Price"])
                max_rev = float(df_item.loc[best_idx, "Revenue"])
                method = "fit_failed"

        # Bound the suggested optimal price to a reasonable range:
        # If optimal is far outside tested range ( >2x ), clamp to [min/2, max*2]
        lower_bound = min_p * 0.5
        upper_bound = max_p * 2.0
        optimal_price = float(np.clip(optimal_price, lower_bound, upper_bound))

        results.append({
            "Accessory": name,
            "Optimal Price": round(optimal_price, 2),
            "Predicted Max Revenue": round(max_rev, 2),
            "Min Tested Price": min_p,
            "Max Tested Price": max_p,
            "Data Points": len(df_item),
            "Method": method
        })

    result_df = pd.DataFrame(results).sort_values("Predicted Max Revenue", ascending=False)
    st.dataframe(result_df, use_container_width=True)

    st.subheader("📈 Графики по аксессуарам")

    selected = st.selectbox("Выбери аксессуар для просмотра графика:", result_df["Accessory"])
    df_plot = grouped[grouped["Asset Name"] == selected].sort_values("Price")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(df_plot["Price"], df_plot["Revenue"], label="Данные")

    if len(df_plot) >= 3:
        # try fit and plot curve fitted over a denser x-axis
        try:
            params, _ = curve_fit(revenue_model, df_plot["Price"], df_plot["Revenue"], maxfev=5000)
            x_dense = np.linspace(df_plot["Price"].min(), df_plot["Price"].max(), 200)
            y_dense = revenue_model(x_dense, *params)
            ax.plot(x_dense, y_dense, label="Модель (аппроксимация)", linestyle="-")
        except:
            pass

    opt_price = float(result_df[result_df["Accessory"] == selected]["Optimal Price"].values[0])
    ax.axvline(opt_price, linestyle="--", label=f"Оптимальная цена: {opt_price}")

    ax.set_xlabel("Цена")
    ax.set_ylabel("Средний дневной Revenue")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Готово! Оптимальные цены вычислены.")
