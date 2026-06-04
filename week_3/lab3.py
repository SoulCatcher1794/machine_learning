# ## Setup (do not edit)
#
# This cell imports all necessary libraries for the assignment.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# Set a random state for reproducibility
RANDOM_STATE = 42

# Grade Cell: Question 1
#
# Task: Load the dataset and display its first 5 rows.
#
# Instructions:
# 1. Load the 'wisconsin_breast_cancer.csv' file into a pandas DataFrame called `df`.
# 2. Use the `.head()` method to display the first 5 rows.

df = pd.read_csv('wisconsin_breast_cancer.csv')
df.head()



# Grade Cell: Question 2
#
# Task: Prepare the data for modeling.
#
# Instructions:
# 1. Map the 'diagnosis' column to binary values: 'M' (malignant) to 1 and 'B' (benign) to 0.
# 2. Create the feature matrix `X` by dropping the 'diagnosis' column.
# 3. Create the target vector `y` from the now-encoded 'diagnosis' column.

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(columns='diagnosis')
y = df['diagnosis']



# Grade Cell: Question 3
#
# Task: Split and scale the data.
#
# Instructions:
# 1. Split `X` and `y` into `X_train`, `X_test`, `y_train`, and `y_test` with a `test_size` of 0.2 and `random_state=RANDOM_STATE`.
# 2. Initialize a `StandardScaler` and fit it on `X_train`.
# 3. Transform both `X_train` and `X_test` using the fitted scaler, naming them `X_train_scaled` and `X_test_scaled`.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Grade Cell: Question 4
#
# Task: Train a baseline logistic regression model.
#
# Instructions:
# 1. Initialize a `LogisticRegression` model, setting `random_state` to `RANDOM_STATE`.
# 2. Train the model using the scaled training data (`X_train_scaled`, `y_train`).
# 3. Store the trained model in a variable called `log_reg_baseline`.

log_reg_baseline = LogisticRegression(random_state=RANDOM_STATE)
log_reg_baseline.fit(X_train_scaled, y_train)



# Grade Cell: Question 5
#
# Task: Make predictions and calculate performance metrics.
#
# Instructions:
# 1. Use the baseline model to make predictions on the scaled test data. Store them in `y_pred_baseline`.
# 2. Calculate accuracy, precision, recall, and F1-score. Store them in `accuracy_baseline`, `precision_baseline`, `recall_baseline`, and `f1_baseline`.

y_pred_baseline = log_reg_baseline.predict(X_test_scaled)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
precision_baseline = precision_score(y_test, y_pred_baseline)
recall_baseline = recall_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline)



# Grade Cell: Question 6
#
# Task: Compute and visualize the confusion matrix for the baseline model.
#
# Instructions:
# 1. Compute the confusion matrix using `y_test` and `y_pred_baseline`. Store it in `conf_matrix_baseline`.
# 2. Use `seaborn.heatmap` to visualize the confusion matrix.

conf_matrix_baseline = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(conf_matrix_baseline, annot=True, fmt='d', cmap='Greens')



# Grade Cell: Question 7
#
# Task: Generate the ROC curve for the baseline model.
#
# Instructions:
# 1. Get the prediction probabilities for the positive class.
# 2. Compute the false positive rate (`fpr`), true positive rate (`tpr`), and thresholds.
# 3. Calculate the Area Under the ROC Curve (`roc_auc_baseline`).
# 4. Plot the ROC curve.

y_prob_baseline = log_reg_baseline.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_baseline)
roc_auc_baseline = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_baseline:.2f})')



# Grade Cell: Question 8
#
# Task: Train a regularized logistic regression model.
#
# Instructions:
# 1. Initialize a `LogisticRegression` model with `penalty='l2'`, `C=0.1`, and `random_state=RANDOM_STATE`.
# 2. Train the model on the scaled training data.
# 3. Store the trained model in `log_reg_l2`.
# 4. Make predictions on the scaled test data and calculate the accuracy, storing it in `accuracy_l2`.

log_reg_l2 = LogisticRegression(penalty='l2', C=0.1, random_state=RANDOM_STATE)
log_reg_l2.fit(X_train_scaled, y_train)
y_pred_l2 = log_reg_l2.predict(X_test_scaled)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)



# Grade Cell: Question 9
#
# Task: Compare the magnitudes of the model coefficients.
#
# Instructions:
# 1. Calculate the average absolute value of the coefficients for the baseline model (`log_reg_baseline`) and store it in `avg_coef_baseline`.
# 2. Calculate the average absolute value of the coefficients for the L2 regularized model (`log_reg_l2`) and store it in `avg_coef_l2`.

avg_coef_baseline = np.mean(np.abs(log_reg_baseline.coef_))
avg_coef_l2 = np.mean(np.abs(log_reg_l2.coef_))



# Grade Cell: Question 10
#
# Task: Train an LDA model and evaluate its accuracy.
#
# Instructions:
# 1. Initialize a `LinearDiscriminantAnalysis` model.
# 2. Train the model on the scaled training data.
# 3. Make predictions on the scaled test data.
# 4. Calculate the accuracy and store it in `accuracy_lda`.

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred_lda = lda.predict(X_test_scaled)
accuracy_lda = accuracy_score(y_test, y_pred_lda)