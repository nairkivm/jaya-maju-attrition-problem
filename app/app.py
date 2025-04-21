import streamlit as st
import pandas as pd
import joblib
import io

# Load model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Daftar kolom wajib
required_columns = [
    'EmployeeId', 'Attrition', 'Age', 'BusinessTravel', 'DailyRate', 'Department',
    'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction',
    'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
    'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Kolom yang akan dipakai untuk prediksi (setelah encoding)
custom_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'RelationshipSatisfaction', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
    'Department_Sales', 'EducationField_Life Sciences',
    'EducationField_Medical', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'Education_3', 'StockOptionLevel_0',
    'StockOptionLevel_1'
]

st.title("Attrition Prediction - Jaya Jaya Maju")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    if not all(col in df.columns for col in required_columns):
        st.error("CSV file must contain all required columns.")
    else:
        X_predict = df.copy()

        # Label encoding manual (Education & StockOptionLevel tetap disertakan)
        text_columns = list(X_predict.select_dtypes(include='object').columns)
        text_columns.extend(['Education', 'StockOptionLevel'])

        for col in text_columns:
            dummies = pd.get_dummies(X_predict[col].astype('object'), prefix=col).astype(int)
            X_predict = pd.concat([X_predict, dummies], axis=1)
            X_predict.drop(col, axis=1, inplace=True)

        # Pastikan hanya ambil kolom yang ada dari custom_columns
        available_columns = [col for col in custom_columns if col in X_predict.columns]
        missing_columns = [col for col in custom_columns if col not in X_predict.columns]

        for col in missing_columns:
            X_predict[col] = 0  # isi default 0 untuk kolom yang ga ada

        X_predict = X_predict[custom_columns]

        # Scaling
        X_scaled = scaler.transform(X_predict)

        # Predict
        y_pred = model.predict(X_scaled)
        df['PredictedAttrition'] = y_pred

        # Tampilkan hasil
        predicted_rate = (df['PredictedAttrition'].sum() / len(df)) * 100
        st.metric("Predicted Attrition Rate", f"{predicted_rate:.2f}%")
        st.dataframe(df)

        # Tombol download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Result",
            data=csv,
            file_name="predicted_attrition.csv",
            mime='text/csv'
        )
