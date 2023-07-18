#!C:\Users\pc\Desktop\FemCycle_APP\mens_venv\Scripts\python.exe
import numpy as np
import  pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def predict_menstrual_cycle(periods):
    average_period = np.mean(periods)
    std_deviation = np.std(periods)
    covariance_matrix = np.cov(periods[:-1], periods[1:])
    predicted_cycle = np.random.multivariate_normal([average_period, average_period], covariance_matrix, 1)[0]
    return predicted_cycle


dataset = pd.read_csv("CycleData.csv")
# print(dataset.head())
length_of_cycle = dataset["LengthofCycle"].tolist()
print(predict_menstrual_cycle(length_of_cycle))


# Load the dataset from a CSV file
#data = pd.read_csv("CycleData.csv")

# Split the dataset into input features (X) and target variable (y)
X = dataset.drop("NextPeriodDate", axis=1)  # Exclude the target variable
y = dataset["NextPeriodDate"]

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
a = model.fit(X, y)

# Predict the next period date for the testing set
y_pred = model.predict(a)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(42, y_pred)
print("Mean Squared:", mse)


