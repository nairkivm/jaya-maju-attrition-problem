import pandas as pd
import joblib
# Predict unknown attrition from original data

df_predict = pd.read_csv('data/df_predict.csv')
X_predict = df_predict

# Create a text_columns list
text_columns = list(X_predict.select_dtypes(include=['object']).columns)

# Add label encoded columns to text_columns
label_encoded_columns = ['Education', 'StockOptionLevel']
text_columns.extend(label_encoded_columns)

for col in text_columns:
    # Perform one-hot encoding on 'text' columns
    dummies = pd.get_dummies(X_predict[col].astype('object'), prefix=col).astype(int)

    # Concatenate the dummy variables to the original DataFrame
    X_predict = pd.concat([X_predict, dummies], axis=1)

    # Drop the original column
    X_predict.drop(col, axis=1, inplace=True)

# Drop unnecessary columns
custom_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
       'HourlyRate', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'RelationshipSatisfaction', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Life Sciences',
       'EducationField_Medical', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'Education_3', 'StockOptionLevel_0',
       'StockOptionLevel_1']
X_predict = X_predict[custom_columns]

# Apply scaling
scaler = joblib.load('./scaler.pkl')
X_predict = scaler.transform(X_predict)

# Predict unknown attrition
selected_model = joblib.load('./svm_model.pkl')
y_predict = selected_model.predict(X_predict)
df_predict['Attrition'] = y_predict