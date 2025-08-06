import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split
from sklearn.linear_model     import LinearRegression
from sklearn.metrics          import mean_squared_error, r2_score

st.set_page_config(page_title="Housing Regression", layout="wide")

st.title("🏠 California Housing: Median Income vs. House Value")

# 1) Load data (you can upload your own CSV)
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

df = load_data()

# 2) Let user pick X and y columns
st.sidebar.header("Feature Selection")
all_cols = df.columns.tolist()
all_cols.remove('ocean_proximity')
all_cols.remove('total_bedrooms')
default_x = "median_house_value"
default_y = "median_income"

x_col = st.sidebar.selectbox("Feature (X):", options=all_cols, index=all_cols.index(default_x))
y_col = st.sidebar.selectbox("Target (y):", options=all_cols, index=all_cols.index(default_y))

# 3) Train/test split slider
test_size = st.sidebar.slider("Test set size (%)", min_value=5, max_value=50, value=20, step=5) / 100.0
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

if st.sidebar.button("Train Model"):
    # Prepare data
    X = df[[x_col]]
    y = df[[y_col]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]

    st.subheader("Model Performance")
    st.markdown(f"- **MSE:** {mse:.2f}")
    st.markdown(f"- **R²:** {r2:.2f}")
    st.markdown(f"- **Equation:**  y = {coef:.2f}·x + {intercept:.2f}")

    # Plot
    st.subheader("Scatter & Regression Line")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")
    ax.scatter(X_test, y_test,  color="green", alpha=0.6, label="Test data")

    # regression line
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
    y_line = coef * x_line + intercept
    ax.plot(x_line, y_line, color="red", lw=2, label="Fit: y = {:.2f}x + {:.2f}".format(coef, intercept))

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.info("Configure your model in the sidebar and click **Train Model**.")
