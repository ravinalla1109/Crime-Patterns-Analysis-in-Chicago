import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import joblib
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib  # For parallel computation using Dask with scikit-learn estimators
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load data
data = pd.read_csv(r"C:\Users\ravin\Downloads\MLRPOJ\Cleaned_Data.csv").sample(frac=0.1, random_state=42)  # Sample 10% of data for quick testing

# Assuming 'Primary Type' is the target variable
X = data.drop(['Primary Type'], axis=1)
y = data['Primary Type']

# Identify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the models to compare
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)  # Reduced n_estimators and enabled parallel processing
}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for name, model in models.items():
    # Create and fit the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} - Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Plotting the results
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Comparison of Classification Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()


# Assuming 'Latitude' and 'Longitude' are the columns in your dataset
# First, ensure there are no missing values for these operations
coords = data.dropna(subset=['Latitude', 'Longitude'])

# You might want to sample the data if it's very large
coords_sample = coords.sample(n=10000, random_state=42)  # Adjust n according to your computational capacity

# Apply K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
coords_sample['cluster'] = kmeans.fit_predict(coords_sample[['Latitude', 'Longitude']])

# Then you can plot or analyze these clusters
import matplotlib.pyplot as plt

plt.scatter(coords_sample['Longitude'], coords_sample['Latitude'], c=coords_sample['cluster'], cmap='viridis')
plt.colorbar()  # Shows color bar
plt.title('Crime Hotspots by Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Assume these are your test labels and predictions from a model
# y_test, y_pred are the true labels and predicted labels respectively

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Extract classes and their performance metrics
classes = list(report.keys())[:-3]  # Skip the last three summary rows
precision_score = [report[cls]['precision'] for cls in classes]
recall_score = [report[cls]['recall'] for cls in classes]
f1_score = [report[cls]['f1-score'] for cls in classes]

# Increase figure size
plt.figure(figsize=(20, 10))

# Decrease bar width
bar_width = 0.2

# Set position of bar on X axis
r1 = np.arange(len(precision_score))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Make the plot
plt.bar(r1, precision_score, color='blue', width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r2, recall_score, color='green', width=bar_width, edgecolor='grey', label='Recall')
plt.bar(r3, f1_score, color='orange', width=bar_width, edgecolor='grey', label='F1-score')

# Add xticks on the middle of the group bars
plt.xlabel('Classes', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(precision_score))], classes)
plt.ylabel('Scores')
plt.title('Classification Report')

# Create legend & Show graphic
plt.legend()
plt.show()


# Assuming 'Primary Type' is the column containing the classes
classes = data['Primary Type'].unique()

# Create an empty DataFrame to store the description
class_description = pd.DataFrame()

# Iterate over each class
for class_name in classes:
    # Filter the data for the current class
    class_data = data[data['Primary Type'] == class_name]
    
    # Describe the filtered data and append it to the class_description DataFrame
    class_stats = class_data.describe()
    class_stats.columns = [f'{col} ({class_name})' for col in class_stats.columns]
    class_description = pd.concat([class_description, class_stats], axis=0)

# Transpose the DataFrame for better visualization
class_description = class_description.T


# Assuming your data has numerical features you want to analyze
numerical_features = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_features.corr()

import seaborn as sns  # Make sure this import is included


# Example usage of seaborn
numerical_features = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_features.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv(r"C:\Users\ravin\Downloads\MLRPOJ\Cleaned_Data.csv")

# Fill missing values using forward fill method
data.ffill(inplace=True)

# Create a count-based hotspot label
data['Crime Count'] = data.groupby(['District', 'Year'])['ID'].transform('count')
data['Hotspot Label'] = pd.qcut(data['Crime Count'], q=[0, .50, .75, 1], labels=['Low', 'Medium', 'High'])

# Sample a subset of the data
sampled_data = data.sample(frac=0.1, random_state=42)
X = sampled_data.drop(['Primary Type', 'Hotspot Label', 'Crime Count', 'ID'], axis=1, errors='ignore')
y = sampled_data['Hotspot Label']

# Encode categorical data
le = LabelEncoder()
X = X.apply(lambda col: le.fit_transform(col) if col.dtypes == object else col)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importances
feature_importances = classifier.feature_importances_
features = X.columns
plt.figure(figsize=(10, 5))
plt.bar(features, feature_importances, color='skyblue')
plt.title('Feature Importances in RandomForest Classifier')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load data
data = pd.read_csv(r"C:\Users\ravin\Downloads\MLRPOJ\Cleaned_Data.csv")
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Safely convert date

# Assuming an intervention took place at the start of 2020
data['Year'] = data['Date'].dt.year
pre_intervention = data[data['Date'] < '2020-01-01']
post_intervention = data[data['Date'] >= '2020-01-01']

# Aggregate crime counts by year
pre_counts = pre_intervention.groupby('Year').size()
post_counts = post_intervention.groupby('Year').size()


plt.figure(figsize=(12, 6))
plt.plot(pre_counts.index, pre_counts.values, label='Pre-Intervention', marker='o')
plt.plot(post_counts.index, post_counts.values, label='Post-Intervention', marker='o')
plt.title('Crime Rates Before and After Intervention')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.legend()
plt.grid(True)
plt.show()


# T-test to compare pre and post intervention crime rates
t_stat, p_value = ttest_ind(pre_counts.values, post_counts.values)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
if p_value < 0.05:
    print("Statistically significant difference in crime rates before and after the intervention.")
else:
    print("No statistically significant difference in crime rates.")

    

