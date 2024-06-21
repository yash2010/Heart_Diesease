# Heart Disease Risk Prediction System 🫀

This repository contains a machine learning model implemented using TensorFlow that predicts the risk of heart disease based on various medical and personal attributes. The model is trained on the heart disease dataset and classifies the risk level into one of five categories.

## Dataset 🧾

The dataset used for this project is stored in a CSV file named heart.csv. This file contains various medical and personal attributes including:

+ Age
+ Sex
+ Chest pain type
+ Resting blood pressure
+ Serum cholesterol
+ Fasting blood sugar
+ Resting electrocardiographic results
+ Maximum heart rate achieved
+ Exercise induced angina
+ ST depression induced by exercise relative to rest
+ The slope of the peak exercise ST segment
+ Number of major vessels colored by fluoroscopy
+ Thalassemia
  
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv)

## Files 📁
+ Heart_Disease_Prediction.py: Python script containing the implementation of the heart disease risk prediction system.
+ heart.csv: Dataset containing medical and personal attributes for the prediction.
  
## Requirements 
+ Python 3.6 or higher
+ TensorFlow
+ Pandas
+ NumPy
+ Scikit-learn
+ Matplotlib
  
You can install the required libraries using the following command:

```sh
pip install tensorflow pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone the repository:
```sh
git clone https://github.com/yash2010/Heart_Disease_Prediction.git
```

2. Navigate to the project directory:
```sh
cd Heart_Disease_Prediction
```

3. Ensure that heart.csv is in the project directory.

4. Run the Python script:
```sh
python Heart_Disease_Prediction.py
```

5. When prompted, enter the following attributes to predict the risk of heart disease:

+ Age
+ Sex (0 for female, 1 for male)
+ Chest pain type (0-3)
+ Resting blood pressure
+ Serum cholesterol
+ Fasting blood sugar > 120 mg/dl (1 for true, 0 for false)
+ Resting electrocardiographic results (0-2)
+ Maximum heart rate achieved
+ Exercise induced angina (1 for yes, 0 for no)
+ ST depression induced by exercise relative to rest
+ The slope of the peak exercise ST segment (0-2)
+ Number of major vessels colored by fluoroscopy (0-3)
+ Thalassemia (0 for normal, 1 for fixed defect, 2 for reversible defect)
+ The system will output the predicted risk level of heart disease.

## Functions

### Data Preprocessing
+ The script reads the heart.csv dataset and displays the first few rows and the data types of the columns.
+ Histograms are plotted for each feature, showing the distribution for each target class.
+ Missing values are dropped, and the feature values are standardized using StandardScaler.

### Model Training
+ The dataset is split into training, validation, and test sets.
+ A neural network is created using TensorFlow's Sequential API with the following layers:
  + Dense layer with 32 units and ReLU activation
  + Dense layer with 16 units and ReLU activation
  + Dense output layer with softmax activation
  
+ The model is compiled with Adam optimizer and categorical crossentropy loss.
+ The model is trained for 60 epochs with a batch size of 15.
  
### Model Evaluation
+ The model is evaluated on the test set, and the accuracy on training and validation sets is printed.
+ A classification report is generated to show precision, recall, and F1-score for each class.
  
### User Input for Prediction
+ The script prompts the user to input their medical and personal attributes.
+ The input data is scaled using StandardScaler.
+ The model predicts the risk level, and the result is printed.

## Visualization 📊
The script uses Matplotlib to plot histograms for each feature, showing the distribution for each target class.

## Example
After running the script and entering the required attributes, the system will output the predicted risk level of heart disease, ranging from "Low risk" to "Critical risk".
