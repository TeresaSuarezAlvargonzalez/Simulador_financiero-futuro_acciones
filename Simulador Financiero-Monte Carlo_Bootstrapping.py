import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import requests
import plotly.express as px

st.set_page_config(page_title="Simulador Financiero", layout="wide")

st.title("📈 Simulador Financiero con Monte Carlo y Bootstrapping")

def calc_size(width=1000, ratio=2/4):
    height = int(width * ratio)
    return width, height



# Sidebar para parámetros
st.sidebar.header("Parámetros del Activo")
ticker = st.sidebar.text_input("Introduce el ticker (ej: AAPL, GOOG, MSFT)", value="GOOG")
start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2024-01-01"))

n_dias = st.sidebar.number_input("Número de días para simular Monte Carlo", min_value=1, max_value=365, value=30)
n_simulaciones = st.sidebar.number_input("Número de simulaciones Monte Carlo", min_value=1, max_value=1000, value=100)

# Aquí creamos un slider para el ancho global que se usará en los gráficos
width_global = st.sidebar.slider("Ancho gráfico (px)", min_value=300, max_value=2400, value=1000)

if st.sidebar.button("📥 Descargar datos y ejecutar análisis"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)['Close']
        if data.empty:
            st.error("❌ No se encontraron datos para el ticker y las fechas dadas.")
        else:
            st.success(f"✅ Datos descargados correctamente para {ticker} ({len(data)} registros)")

            width, height = calc_size(width_global)

            fig = px.line(data, title="📈 Precios históricos")
            fig.update_layout(
                width=width,
                height=height,
                margin=dict(l=40, r=40, t=40, b=40),
                showlegend=False
            )

            col1, col2, col3 = st.columns([1,6,1])
            with col2:
                st.plotly_chart(fig, use_container_width=False)

            # Precio máximo, mínimo y último día
            precio_max = float(data.max())
            precio_min = float(data.min())
            precio_hoy = float(data.iloc[-1])
            fecha_hoy = data.index[-1].date()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📉 Precio mínimo", f"${precio_min:.2f}")
            with col2:
                st.metric("📅 Precio último día", f"${precio_hoy:.2f}", help=f"Fecha: {fecha_hoy}")
            with col3:
                st.metric("📈 Precio máximo", f"${precio_max:.2f}")

            # Rendimientos diarios logarítmicos y simples
            rendimientos_diarios = np.log(data / data.shift(1)).dropna()
            rendimientos_diarios_n = data.pct_change().dropna()
            hist_data = rendimientos_diarios_n.values.flatten()
            P_t = float(data.iloc[-1])

            # Histograma de rendimientos
            width_hist, height_hist = calc_size(width_global)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=hist_data,
                nbinsx=50,
                histnorm='probability density',
                marker_color='rgba(100, 149, 237, 0.9)',
                name='Histograma'
            ))

            xmin, xmax = hist_data.min(), hist_data.max()
            x = np.linspace(xmin, xmax, 100)
            mu, std = stats.norm.fit(hist_data)
            pdf = stats.norm.pdf(x, mu, std)

            fig_hist.add_trace(go.Scatter(
                x=x,
                y=pdf,
                mode='lines',
                line=dict(color='white', width=3),
                name=f'Normal ajustada μ={mu:.4f}, σ={std:.4f}'
            ))

            conf_intervals = {'68%': 1, '95%': 2, '99%': 3}
            for label, z in conf_intervals.items():
                for direction in [-1, 1]:
                    x_pos = mu + direction * z * std
                    color = "blue" if z == 1 else "orange" if z == 2 else "red"
                    fig_hist.add_shape(
                        type="line",
                        x0=x_pos, x1=x_pos,
                        y0=0, y1=max(pdf)*1.5,
                        line=dict(color=color, width=1.5, dash="dot")
                    )
            for i, (label, z) in enumerate(conf_intervals.items()):
                x_start = mu - z * std
                x_end = mu + z * std
                y_pos = -0.6 - i * 0.5
                color = "lime" if z == 1 else "red" if z == 2 else "red"
                fig_hist.add_annotation(
                    x=x_end,
                    y=y_pos,
                    ax=x_start,
                    ay=y_pos,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=color,
                    font=dict(color="white", size=12),
                    align="center"
                )
            fig_hist.add_shape(
                type="line",
                x0=mu,
                x1=mu,
                y0=0,
                y1=max(pdf)*1.5,
                line=dict(color="yellow", width=2, dash="dashdot")
            )

            n_dias_futuro = 10
            mu_pct = mu * 100
            precio_esperado_futuro = P_t * (1 + mu) ** n_dias_futuro

            fig_hist.add_annotation(
                x=mu,
                y=max(pdf) * 0.10,
                text=(
                    f"μ diaria ≈ {mu_pct:.4f}%<br>"
                    f"Precio esperado ({n_dias_futuro} días) ≈ {precio_esperado_futuro:.2f}"
                ),
                showarrow=False,
                font=dict(color="white", size=12),
                align="center",
                bgcolor="black",
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )

            texto_precios = ""
            for label, z in conf_intervals.items():
                ret_min = mu - z * std
                ret_max = mu + z * std
                price_min = P_t * (1 + ret_min)
                price_max = P_t * (1 + ret_max)
                texto_precios += f"{label} precios aprox.: {price_min:.2f} - {price_max:.2f}<br>"

            fig_hist.add_annotation(
                x=0.99,
                y=0.90,
                xref="paper",
                yref="paper",
                text=texto_precios,
                showarrow=False,
                font=dict(color="white", size=14),
                align="left",
                borderwidth=1,
                borderpad=10,
                bgcolor="rgba(0, 0, 0, 1)",
            )

            fig_hist.update_layout(
                title=f'Distribución de Rendimientos Diarios - {ticker}',
                xaxis_title='Rendimiento Diario',
                yaxis_title='Densidad',
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white', family='Arial, sans-serif'),
                showlegend=False,
                margin=dict(t=50, b=80, l=40, r=40),
                yaxis=dict(range=[-2.5, max(pdf)*1.2]),
                width=width_hist,
                height=height_hist,
                autosize=False
            )

            col_left, col_center, col_right = st.columns([1,6,1])
            with col_center:
                st.plotly_chart(fig_hist, use_container_width=False)

            # Simulación Monte Carlo
            st.subheader("🎲 Simulación Monte Carlo de precios futuros (log-normal)")
            precio_inicial = float(data.iloc[-1])

            simulaciones = np.zeros((n_simulaciones, n_dias))
            for i in range(n_simulaciones):
                rendimientos_simulados = np.random.normal(mu, std, n_dias)
                simulaciones[i, :] = precio_inicial * np.cumprod(np.exp(rendimientos_simulados))

            media_final = np.mean(simulaciones[:, -1])
            mediana_final = np.median(simulaciones[:, -1])
            perc_5 = np.percentile(simulaciones[:, -1], 5)
            perc_95 = np.percentile(simulaciones[:, -1], 95)

            st.markdown(f"""
            **Resumen de precios finales después de {n_dias} días (simulación Monte Carlo):**

            - Precio inicial: ${precio_inicial:.2f}  
            - Media final simulada: ${media_final:.2f}  
            - Mediana final simulada: ${mediana_final:.2f}  
            - Percentil 5%: ${perc_5:.2f}  
            - Percentil 95%: ${perc_95:.2f}
            """)

            width_mc, height_mc = calc_size(width_global)
            fig_mc = go.Figure()
            for i in range(min(100, n_simulaciones)):
                color = 'blue' if simulaciones[i, -1] > precio_inicial else 'red'
                fig_mc.add_trace(go.Scatter(
                    y=simulaciones[i, :],
                    mode='lines',
                    line=dict(color=color, width=0.7),
                    opacity=0.6,
                    showlegend=False
                ))

            fig_mc.update_layout(
                title=f'Simulaciones Monte Carlo ({n_simulaciones} simulaciones, {n_dias} días)',
                xaxis_title='Días',
                yaxis_title='Precio Simulado',
                width=width_mc,
                height=height_mc,
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                margin=dict(t=50, b=50, l=40, r=40),
                autosize=False
            )

            st.plotly_chart(fig_mc, use_container_width=False)

            # Bootstrapping (muestra aleatoria con reemplazo)
            st.subheader("🔄 Simulación Bootstrap de precios futuros")

            n_bootstrap = n_simulaciones
            bootstrap_paths = np.zeros((n_bootstrap, n_dias))

            for i in range(n_bootstrap):
                indices = np.random.randint(0, len(rendimientos_diarios), n_dias)
                rend_bootstrap = rendimientos_diarios.values[indices]
                bootstrap_paths[i, :] = precio_inicial * np.cumprod(np.exp(rend_bootstrap.flatten()))

            media_bootstrap = np.mean(bootstrap_paths[:, -1])
            mediana_bootstrap = np.median(bootstrap_paths[:, -1])
            perc_5_boot = np.percentile(bootstrap_paths[:, -1], 5)
            perc_95_boot = np.percentile(bootstrap_paths[:, -1], 95)

            st.markdown(f"""
            **Resumen de precios finales después de {n_dias} días (bootstrap):**

            - Precio inicial: ${precio_inicial:.2f}  
            - Media final bootstrap: ${media_bootstrap:.2f}  
            - Mediana final bootstrap: ${mediana_bootstrap:.2f}  
            - Percentil 5%: ${perc_5_boot:.2f}  
            - Percentil 95%: ${perc_95_boot:.2f}
            """)

            width_bs, height_bs = calc_size(width_global)
            fig_bs = go.Figure()
            for i in range(min(100, n_bootstrap)):
                color = 'blue' if bootstrap_paths[i, -1] > precio_inicial else 'red'
                fig_bs.add_trace(go.Scatter(
                    y=bootstrap_paths[i, :],
                    mode='lines',
                    line=dict(color=color, width=0.7),
                    opacity=0.6,
                    showlegend=False
                ))

            fig_bs.update_layout(
                title=f'Simulaciones Bootstrap ({n_bootstrap} simulaciones, {n_dias} días)',
                xaxis_title='Días',
                yaxis_title='Precio Simulado',
                width=width_bs,
                height=height_bs,
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                margin=dict(t=50, b=50, l=40, r=40),
                autosize=False
            )

            st.plotly_chart(fig_bs, use_container_width=False)

            def get_news_newsapi(query):
                API_KEY = "e2b86727fa874751911c95e690357262"
                url = ("https://newsapi.org/v2/everything?"
                    f"q={query}&"
                    "sortBy=publishedAt&"
                    "language=en&"
                    "pageSize=10&"
                    f"apiKey={API_KEY}")
                response = requests.get(url)
                data = response.json()
                if data.get("status") != "ok":
                    st.error("Error al obtener noticias")
                    return []
                articles = data.get("articles", [])
                news_list = []
                for article in articles:
                    title = article.get("title")
                    url = article.get("url")
                    news_list.append((title, url))
                return news_list

            st.subheader(f"Noticias recientes para {ticker}")
            news = get_news_newsapi(ticker)

            if news:
                for title, url in news:
                    st.markdown(f"- [{title}]({url})")
            else:
                st.write("No se encontraron noticias para este ticker.")

    except Exception as e:
        st.error(f"Error al descargar o procesar datos: {e}")