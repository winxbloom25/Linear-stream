import pickle
from sklearn.linear_model import LinearRegression

X = [[5], [6], [7], [8], [9]]
y = [2, 3, 5, 8, 11]

model = LinearRegression()
model.fit(X, y)

# Save the trained model using pickle
with open('model2.pkl', 'wb') as file:
    pickle.dump(model, file)