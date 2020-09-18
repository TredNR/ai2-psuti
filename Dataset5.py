# http://localhost:8501/

import streamlit as st
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

st.title("A conceptual model for the coronavirus disease 2019 (COVID-19)")

st.subheader("Mathematical formulas")

r'''
$$S' = -\frac{\beta_0SF}{N} - \frac{\beta(t)SL}{N} - μS$$

$$E' = \frac{\beta_0SF}{N} + \frac{\beta(t)SL}{N} -(σ+μ)E$$

$$I' = σE - (\gamma+μ)I$$

$$R' = \gamma I-μR$$

$$N' = -μN$$

$$D' = d\gamma I - λD$$

$$C' = σE$$

where
$$\beta(t) = \beta_0(1-\alpha)(1-\frac{D}{N})^k$$
'''

# Функция модели
def deriv(y, t, beta0, k, mu, F, sigma, gamma, d, alpha, lmbda):
    S, E, I, R, N, D, C = y
    if callable(alpha):
        alpha = alpha(t)

    if callable(beta0):
        beta0 = beta0(t)

    beta_t = beta0 * (1 - alpha) * ((1 - D / N) ** k)
    dSdt = - (beta0 * S * F / N) - (beta_t * S * I / N) - mu * S
    dEdt = (beta0 * S * F / N) + (beta_t * S * I / N) - (sigma + mu) * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I - mu * R
    dNdt = - mu * N
    dDdt = d * gamma * I - lmbda * D
    dCdt = sigma * E
    return dSdt, dEdt, dIdt, dRdt, dNdt, dDdt, dCdt

# Очень большая установка системных параметров
st.sidebar.title("System parameters")

st.sidebar.subheader("Days")
days = st.sidebar.slider("", min_value=1, max_value=365, value=365, step=1)
t = np.linspace(0, days-1, days)

st.sidebar.subheader("[σ] Mean latent periode (1/n)")
sigma = 1 / st.sidebar.slider("n", min_value=1, max_value=10, value=3, step=1)
st.write("Sigma = ", sigma)

st.sidebar.subheader("[γ] Mean infectious period (1/n)")
gamma = 1 / st.sidebar.slider("n", min_value=1, max_value=10, value=5, step=1)
st.write("Gamma = ", gamma)

F = st.sidebar.number_input("[F] Number of zoonotic cases", value=0)
N0 = st.sidebar.number_input("[N0] Initial population size", value= 14000000)
S0 = 0.9 * N0
st.write("Initial susceptible population 0.9 *", N0, " =", S0)

# Настройка выбора параметра beta0
st.subheader("[β0] Transmission rate")
select1 = st.selectbox("Parameter selection", ["Fixed value",  "Lambda (t) with simple parameter", "Lambda (t) with equation"])
if select1 == "Fixed value":
    beta0 = st.number_input("", value=1.68)
elif select1 == "Lambda (t) with simple parameter":
    A11 = st.number_input("A11", value=0.5944, format="%.4f")
    B11 = st.number_input("B11", value=200)
    C11 = st.number_input("C11", value=1.68)
    st.write("beta0 = lambda t: ", A11, "if t < ", B11, " else ", C11)
    beta0 = lambda t: A11 if t < B11 else C11
elif select1 == "Lambda (t) with equation":
    A12 = st.number_input("A12", value=0.5944, format="%.4f")
    B12 = st.number_input("B12", value=200)
    C12 = st.number_input("C12", value=1.68)
    st.write("beta0 = lambda t: (", A12, " + (", C12, " - ", A12,") /", B12," * t ) if t <",  B12 ,"else", C12)
    beta0 = lambda t: (A12 + (C12 - A12)/B12 * t) if t < B12 else C12

# Настройка выбора параметра alpha
st.subheader("[α] Governmental action strength")
select2 = st.selectbox("Parameter selection", ["Fixed value", "Lambda (t) with simple parameter","Lambda (t) with double condition"])
if select2 == "Fixed value":
    alpha = st.number_input("", value=0.0)
elif select2 == "Lambda (t) with simple parameter":
    A21 = st.number_input("A21", value=0.84)
    B21 = st.number_input("B21", value=200)
    C21 = st.number_input("C21", value=0.42)
    st.write("alpha = lambda t:", A21, "if t >", B21, "else", C21)
    alpha = lambda t: A21 if t > B21 else C21
elif select2 == "Lambda (t) with double condition":
    A22 = st.number_input("A22", value=0.84)
    B22 = st.number_input("B22", value=200)
    C22 = st.number_input("C22", value=0.42)
    B23 = st.number_input("B23", value=50)
    A23 = st.number_input("A23", value=0)
    st.write("alpha = lambda t:", A22, "if t >", B22,"else (", C22, "if t >", B23, "else", A23, ")")
    alpha = lambda t: A22 if t > B22 else (C22 if t > B23 else A23)

k = st.sidebar.number_input("[k] Intensity of responds", value=1117.3)
mu = st.sidebar.number_input("[μ] Emigration rate", value=0.00)

d = st.sidebar.number_input("[d] Proportion of severe cases", 0.2)
lmbda = st.sidebar.number_input("[λ] Mean duration of public reaction", value=0.089, step=0.001, format="%.3f")

E0 = st.sidebar.number_input("[E0] Initial Exposed", value=1)
I0 = st.sidebar.number_input("[I0] Initial Infectious", value=0)
R0 = st.sidebar.number_input("[R0] Initial Removed", value=0)
D0 = st.sidebar.number_input("*[D0] Inital Dead", value=0)
C0 = st.sidebar.number_input("[C0] Initial Cumulative cases", value=0)

# Ссылка на документы
st.sidebar.title("Documentation")
link1 = 'https://drive.google.com/file/d/161XLGi5v31nk0AFtlYp-bAWBJmiYugYc/view?usp=sharing'
if st.sidebar.button("An article on the conceptual model about Covid in Wuhan", key=4):
    webbrowser.open_new_tab(link1)

y0 = S0, E0, I0, R0, N0, D0, C0 # Initial conditions vector

ret = odeint(deriv, y0, t, args=(beta0, k, mu, F, sigma, gamma, d, alpha, lmbda))
S, E, I, R, N, D, C = ret.T

# Цвета для графика
COLORS_MAP = {
        "Susceptible" : '-.b',
        "Exposed": '--c',
        "Infected": 'r',
        "Recovered": '--g',
        "Cumulative": 'm',
        "Population": "violet",
        "Dead": "-.k"
    }

# Показать перечень параметров
My_details = [{'F': F, 'N0': N0, 'S0': S0, 'beta0': beta0, 'alpha': alpha, 'k': k, 'mu': mu, 'sigma': sigma, 'gamma': gamma, 'd': d, 'lmbda': lmbda, 'E0': E0, 'I0': I0, 'R0': R0, 'D0':D0, 'C0':C0}]
if st.checkbox("Watch my parameters", key=1):
    st.write("Parameters", My_details)

# Построение графика
def plotseird(t, colors, **kwargs):
    f, ax = plt.subplots(1,1,figsize=(10, 6))

    for metric_name, metric_values in kwargs.items():
        color = colors.get(metric_name, 'b')
        ax.plot(t, metric_values, color, alpha=0.8, linewidth=2.5, label=metric_name)
    ax.set_xlabel('Time (days)', labelpad = 12)
    #ax.set_ylabel('Population', labelpad = 35)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend(borderpad=2.0)
    legend.get_frame().set_alpha(0.6)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.savefig('last.png')
    st.pyplot()

# Вывод графиков по нажатию checkbox
st.subheader("Show graphs")
if st.checkbox("Show graph with E, I, D", key=2):
    plotseird(t, colors=COLORS_MAP, Exposed=E, Infected=I, Dead=D)
if st.checkbox("Show graph with I, R, S", key=3):
    plotseird(t, colors=COLORS_MAP, Infected=I, Recovered=R, Susceptible=S)

