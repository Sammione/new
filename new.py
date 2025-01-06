import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

print("Starting the script...")

# Load the dataset
print("Loading dataset...")
try:
    df = pd.read_csv('product_pricing_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# Preview the dataset
print("Previewing dataset:")
print(df.head())
print(f"Dataset shape: {df.shape}")
print("Missing values per column:\n", df.isnull().sum())

# Visualize Price vs Units Sold
print("Creating visualization...")
sns.scatterplot(x="Price", y="Units_Sold", hue="Product_Category", data=df)
plt.title("Price vs Units Sold")
plt.savefig("price_vs_units_sold.png")  # Save plot for review
print("Visualization saved as 'price_vs_units_sold.png'.")

# Feature Selection
print("Selecting features...")
X = df[["Price", "Competitor_Price", "Customer_Rating", "Demand_Elasticity"]]
y = df["Units_Sold"]

# Split the data
print("Splitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
print("Training the model...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Predictions and Evaluation
print("Making predictions and evaluating...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save the model
print("Saving the model...")
joblib.dump(model, "pricing_model.pkl")
print("Model saved as 'pricing_model.pkl'.")
