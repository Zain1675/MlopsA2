from flask import Flask, render_template
import pandas as pd
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

app = Flask(__name__)

# Load the dataset
api_key = "YOUR_API_KEY"

# Define the URL for the API request
symbol = "AAPL"  # Replace with any stock symbol of your choice
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}"

# Send a request to the Alpha Vantage API and convert the response to a Pandas DataFrame
response = requests.get(url)
data = response.json()["Time Series (Daily)"]
df = pd.DataFrame.from_dict(data, orient="index")
df = df.astype(float)
df["target"] = df["4. close"].shift(-1)
X = df.iloc[:-1, :-1]
y = df.iloc[:-1, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


@app.route("/")
def dashboard():
    # Get the first row of the dataset
    first_row = df.head(1)
    # Convert the first row to a dictionary to make it easier to display in the template
    first_row_dict = first_row.to_dict("records")[0]
    # Render the template with the regression metrics and the first row of the dataset
    return render_template(
        "dashboard.html",
        first_row=first_row_dict,
        Mean_Squared_Error=mse,
        Root_Mean_Squared_Error=rmse,
        Mean_Absolute_Error=mae,
        R_squared=r2,
    )


if __name__ == "__main__":
    app.run(debug=True)
