import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample data (house size vs price)
X = np.array([[500], [700], [900], [1100], [1300]])
y = np.array([50, 70, 90, 110, 130])

model = LinearRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
