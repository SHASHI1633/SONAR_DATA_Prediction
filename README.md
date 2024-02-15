# SONAR_DATA_Prediction

### Step 1: Import Dependencies
python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

- Importing necessary libraries: NumPy, Pandas, Logistic Regression model, and metrics from Scikit-learn.

### Step 2: Data Collection and Processing
python
sonar_data = pd.read_csv('/content/sonar data.csv', header=None)

- Reads the dataset 'sonar data.csv' into a Pandas DataFrame named sonar_data.

### Step 3: Exploratory Data Analysis
python
sonar_data.head()
sonar_data.shape
sonar_data.describe()
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()

- Displays the initial rows, shape, statistical summary, value counts of target classes ('M' for Mine, 'R' for Rock), and mean values grouped by the target column.

### Step 4: Data Preparation
python
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

- Separates features (X) and target labels (Y) from the dataset.

### Step 5: Training and Test Data Splitting
python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

- Splits the dataset into training and testing sets using train_test_split.

### Step 6: Model Training
python
model = LogisticRegression()
model.fit(X_train, Y_train)

- Initializes a Logistic Regression model and trains it using the training data (X_train, Y_train).

### Step 7: Model Evaluation
python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

- Evaluates the model's accuracy on both training and test datasets using accuracy_score.

### Step 8: Making Predictions
python
input_data = (0.0307, 0.0523, ... , 0.0124, 0.0055)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

- Defines sample input data, converts it to a NumPy array, reshapes it, and predicts its class using the trained model.

### Step 9: Predictive Output
python
if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a mine')

- Outputs whether the predicted object is a rock or a mine based on the prediction.

### Summary:
1. *Data Preparation:* Loading, exploring, separating features and labels.
2. *Model Building:* Logistic Regression model training.
3. *Model Evaluation:* Assessing model accuracy on training and test data.
4. *Making Predictions:* Using the trained model to predict the class of new data.
