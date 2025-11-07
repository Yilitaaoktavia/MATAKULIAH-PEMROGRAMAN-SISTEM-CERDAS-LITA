import pickle
from sklearn.linear_model import LinearRegression

# contoh model
model = LinearRegression()
# model.fit(X_train, y_train)

# simpan model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
