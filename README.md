### NovaNector_ML_Intern
## Elementary Project 1

# Housing Price Prediction Project

1. Introduction
This documentation provides details on the housing price prediction model, including its architecture, training process, and usage instructions. The goal of my project is to predict housing prices based on various features, using different machine-learning models.

2. Model Architecture

Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model has its own architecture and approach to prediction:

I. Linear Regression: This model predicts the target variable by fitting a linear relationship between the input features and the target.

II. Decision Tree Regressor: This model splits the data into subsets based on the value of input features, creating a tree-like model of decisions.

III. Random Forest Regressor: This ensemble model combines multiple decision trees to improve prediction accuracy and control over-fitting.

3. Training Process

a. Data Collection and Importing:
   - Import the dataset into a Jupyter notebook using pandas.
   - Example:
     ```python
     import pandas as pd
     data = pd.read_csv('housing_data.csv')
     ```

b. Data Exploration:
   - Analyze the dataset to find promising attributes and correlations.
   - Plot graphs to visualize data relationships using libraries like matplotlib and seaborn.
     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns
     
     sns.pairplot(data)
     plt.show()
     ```

c. Data Preprocessing:
   - Handle missing values, encode categorical variables, and standardize the data.
   - Create a pipeline to automate these steps.
     ```python
     from sklearn.pipeline import Pipeline
     from sklearn.impute import SimpleImputer
     from sklearn.preprocessing import StandardScaler
     
     pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler())
     ])
     
     data_prepared = pipeline.fit_transform(data)
     ```

d. Train-Test Splitting:
   - Split the dataset into training and testing sets to evaluate the model performance.
     ```python
     from sklearn.model_selection import train_test_split
     
     train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
     ```

e. Model Training:
   - Train the models using the training set and evaluate them using cross-validation.
     ```python
     from sklearn.linear_model import LinearRegression
     from sklearn.tree import DecisionTreeRegressor
     from sklearn.ensemble import RandomForestRegressor
     from sklearn.model_selection import cross_val_score
     
     lin_reg = LinearRegression()
     lin_reg.fit(train_set, train_set_labels)
     
     tree_reg = DecisionTreeRegressor()
     tree_reg.fit(train_set, train_set_labels)
     
     forest_reg = RandomForestRegressor()
     forest_reg.fit(train_set, train_set_labels)
     ```

f. Model Evaluation:
   - Evaluate the models using cross-validation and metrics like RMSE (Root Mean Square Error).
     ```python
     from sklearn.metrics import mean_squared_error
     import numpy as np
     
     lin_scores = cross_val_score(lin_reg, train_set, train_set_labels, scoring='neg_mean_squared_error', cv=10)
     lin_rmse_scores = np.sqrt(-lin_scores)
     
     def display_scores(scores):
         print("Scores:", scores)
         print("Mean:", scores.mean())
         print("Standard deviation:", scores.std())
     
     display_scores(lin_rmse_scores)
     ```

3. Usage Instructions

Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

** How to Use
- Ensure you have the notebook (`housing_price_prediction.ipynb`) and data file (`housing_data.csv`).
- Navigate to the project directory and open the Jupyter notebook.
     ```bash
     jupyter notebook housing_price_prediction.ipynb
     ```
- Execute the cells step by step to load data, preprocess it, train the models, and evaluate them.

4. Predict Housing Prices:
   - Use the trained model to predict housing prices for new data.
     ```python
     final_model = forest_reg
     some_data = data.iloc[:5]
     some_labels = labels.iloc[:5]
     prepared_data = pipeline.transform(some_data)
     predictions = final_model.predict(prepared_data)
     ```

5. Conclusion

This project demonstrates the complete workflow for solving a machine learning problem, from data collection to model deployment. 
