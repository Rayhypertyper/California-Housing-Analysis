import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.metrics        import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('housing.csv')
# df.dropna(inplace=True) # Drops any rows with missing values



# X = pd.get_dummies(df['ocean_proximity'], prefix='ocean', drop_first=True)
X = df[['median_house_value']]
y = df[['median_income']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 # 20% of data set is used for testing while other 80 is training

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

print('coef:', model.coef_, 'intercept:', model.intercept_)

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_line = model.coef_[0] * x_line + model.intercept_

# visualize the data

m = f"{model.coef_[0][0]:.2f}"
b = f"{model.intercept_[0]:.2f}"

plt.figure(figsize=(12,6))
plt.scatter(X_train,y_train, color="blue", label="Training data")
plt.scatter(X_test, y_test, color="green", label="Test data")
plt.title('Median house value vs Median income in california')
plt.plot(x_line, y_line, color='red', linewidth=3, label=f'Regression Line: y = {m}x + {b}')
plt.xlabel('Median house value')
plt.ylabel('Median income')
plt.grid(True)
plt.legend() # shows the legend
plt.tight_layout()

plt.show()

# plt.figure(figsize=(12,6))
# plt.scatter(y_test, y_pred, alpha=0.6)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit (x=y)')
# plt.xlabel("Real  house values")
# plt.ylabel("Predicted hosue values")
# plt.title("Actual vs predicted house values")
# plt.grid(True)
# plt.legend()
# plt.show()

