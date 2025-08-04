import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the advertising dataset
df = pd.read_csv("D:/linear_regression_advertising/advertising.csv")
print(df.head())

# View Data type of each column
print(df.dtypes)

# Check for missing values
print(df.isna().sum()) 

# Visualize the given data
# Plot scatterplot for each pair of variables and their respective KDE using PairGrid
g = sns.PairGrid(df, corner = True)

# Mapping the plots
g.map_lower(sns.scatterplot)
g.map_diag(sns.kdeplot)

# Add legend
g.add_legend
plt.show()

# Train-Test Split
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 17)

# Model development
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Check model performance
print("R^2 score: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))

