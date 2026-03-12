# Indoor-Thermal-Comfort-
# Data-Driven Assessment of Indoor Thermal Comfort

## Project Overview

This project analyzes **indoor thermal comfort conditions** using statistical analysis and machine learning techniques. The goal is to evaluate how environmental factors such as **temperature, humidity, and occupancy** affect the **Thermal Comfort Index (THI)**.

The project applies data preprocessing, visualization, and a machine learning model to identify patterns and predict thermal comfort levels.


## Objectives

* Apply data-driven methods to analyze indoor environmental conditions
* Clean and preprocess real-world datasets
* Calculate the **Thermal Comfort Index (THI)**
* Visualize relationships between environmental variables
* Build a machine learning model to predict thermal comfort



## Dataset

The dataset used in this project contains indoor environmental variables such as:

* Temperature (°C)
* Humidity (%)
* Occupancy (number of people)

These variables are used to compute the **Thermal Comfort Index (THI)** and train the machine learning model.


## THI Formula

The Thermal Comfort Index is calculated using the following formula:

THI = T − (0.55 − 0.0055 × RH) × (T − 14.5)

Where:

* **T** = Temperature
* **RH** = Relative Humidity


## Technology Stack

The project is implemented using:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn


## Machine Learning Model

The project uses the **Random Forest Regressor** model to predict THI values based on environmental variables.

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.


## Visualizations

The project generates several visualizations including:

* Temperature vs THI scatter plot
* Correlation heatmap of environmental variables
* Feature importance graph from the Random Forest model

These visualizations help understand the relationships between indoor environmental parameters.


## Project Workflow

1. Load dataset
2. Clean and preprocess data
3. Calculate Thermal Comfort Index (THI)
4. Perform data visualization
5. Train machine learning model
6. Evaluate model performance
7. Predict THI for new environmental conditions


## Model Evaluation

The performance of the machine learning model is evaluated using:

* Mean Squared Error (MSE)
* R² Score

These metrics measure prediction accuracy and model performance.

## Example Prediction

The trained model can predict THI for new indoor conditions such as:

Temperature = 27°C
Humidity = 65%
Occupancy = 10

The model outputs a predicted Thermal Comfort Index value.

## Applications

This project can be useful for:

* Smart building systems
* Indoor environment monitoring
* Energy-efficient HVAC control
* Thermal comfort research in buildings


## Author

Ajay Kumar
MCA Student
K.R. Mangalam University, Gurgaon
