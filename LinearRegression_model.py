import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Load the dataset

df = pd.read_csv("data\Real estate valuation data set.csv")

# Data preprocessing
# Drop any rows with missing values
df.dropna(inplace=True)

# Split features (X) and target variable (Y)
X = df.iloc[:, 1:-1]  
Y = df.iloc[:, -1]   

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# Save the trained model to a file
model_filename = "linear_model.pkl"
joblib.dump(linear_model, model_filename)

# Load the saved model from the file
loaded_model = joblib.load(model_filename)

# Make predictions on the test set using the loaded model
Y_pred = loaded_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)

# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("R-squared:", r2)