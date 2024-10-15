# Crime Patterns Analysis in Chicago

This project focuses on analyzing crime patterns in Chicago using machine learning models and statistical techniques. Here’s an overview of what was implemented in Python:

## 1. Data Preparation and Exploration
- **Dataset**: Used the "Crimes - 2001 to Present" dataset from the Chicago Police Department.
![plot6](https://github.com/user-attachments/assets/4fd91a11-29e5-45a2-8d27-68b00eb79810)

- **Data Cleaning**: Addressed missing values, inconsistencies, and formatted data for analysis.
![plot7](https://github.com/user-attachments/assets/d079b386-02e5-4816-a6b4-43b1a371ac8b)

- **Feature Engineering**: Extracted relevant features such as time, location coordinates, and crime types.
![plot0](https://github.com/user-attachments/assets/e08b8a97-db72-43fe-9a1d-1639eba6ae06)

## 2. Time Series Analysis with ARIMA
- **Model**: Implemented ARIMA(1, 0, 1) for time series forecasting of crime trends.
- **Stationarity Check**: Used the Augmented Dickey-Fuller test to ensure data stationarity.
- **Model Evaluation**: Assessed model performance and forecast accuracy using historical crime data.

## 3. Machine Learning Models
- **Classification Models**: Developed Decision Trees (DT) and Random Forests (RF) to classify crime types.
![plot01](https://github.com/user-attachments/assets/11682039-eb8b-4850-a9bd-2e5906a53ff5)

- **Feature Importance**: Evaluated and visualized the importance of features in predicting crime occurrences.
![plot1](https://github.com/user-attachments/assets/c09d7913-6f25-4c88-bee6-9424bd099a5e)

- **Model Evaluation**: Calculated precision, recall, and F1-score to measure model performance across crime categories.
![plot2](https://github.com/user-attachments/assets/d98d1a31-328d-45fa-85c1-ed9b18e7c689)

## 4. Data Visualization and Insights
- **Visualizations**: Created charts (bar charts, line graphs, maps) to illustrate crime trends, geographical hotspots, and seasonal variations.
![plot8](https://github.com/user-attachments/assets/d5ae4b28-c57a-4167-920f-544eeef5ec29)

- **Insights**: Analyzed spatial-temporal patterns in crime data to identify high-crime areas and understand factors influencing crime rates.
![plot9](https://github.com/user-attachments/assets/66b34251-9e1b-42a6-8ac9-76305ff00e3a)

![plot12](https://github.com/user-attachments/assets/fca28832-9957-4438-ab65-5ca0efe740c1)

## 5. Documentation and Reporting
- **Notebooks**: Used Jupyter notebooks for detailed analysis steps, including data exploration, model training, and evaluation.
- **Scripts**: Python scripts for data preprocessing, model building, and visualization.
- **Results**: Stored model outputs, visualizations, and performance metrics for documentation and future reference.

## Conclusion
This Python-based project enhances understanding of crime dynamics in Chicago through data-driven insights and predictive models. By leveraging machine learning and statistical techniques, it aims to support informed decision-making in public safety and urban planning.
