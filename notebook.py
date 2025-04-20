# %% [markdown]
# # Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

# %% [markdown]
# - Nama: Mohamad Fikri Aulya Nor
# - Email: mohfikri.aulyanor@gmail.com
# - Id Dicoding: nairkivm

# %% [markdown]
# ## Persiapan

# %% [markdown]
# ### Menyiapkan library yang dibutuhkan

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from utils.constants import Constants
from utils.utils import DataUtils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# ### Menyiapkan data yang akan diguankan

# %%
# Initialize the constants
c = Constants()

# Extract all data into dictionary of pandas DataFrames
data = {}
for source_ in c.source.keys():
    data[source_] = pd.read_csv(c.source[source_])
    print(f"Loaded '{source_}' data")

# %%
# Preview the data
data['employee'].head()

# %% [markdown]
# ## Data Undestanding

# %%
# Initialize DataUtils
u = DataUtils()

# Assess the data using asses_data
u.asses_data(data['employee'], 'employee')

# %% [markdown]
# The data contains demographic details, work-related metrics and attrition flag.
# 
# * **EmployeeId** - Employee Identifier
# * **Attrition** - Did the employee attrition? (0=no, 1=yes)
# * **Age** - Age of the employee
# * **BusinessTravel** - Travel commitments for the job
# * **DailyRate** - Daily salary
# * **Department** - Employee Department
# * **DistanceFromHome** - Distance from work to home (in km)
# * **Education** - 1-Below College, 2-College, 3-Bachelor, 4-Master,5-Doctor
# * **EducationField** - Field of Education
# * **EnvironmentSatisfaction** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **Gender** - Employee's gender
# * **HourlyRate** - Hourly salary
# * **JobInvolvement** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **JobLevel** - Level of job (1 to 5)
# * **JobRole** - Job Roles
# * **JobSatisfaction** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **MaritalStatus** - Marital Status
# * **MonthlyIncome** - Monthly salary
# * **MonthlyRate** - Mounthly rate
# * **NumCompaniesWorked** - Number of companies worked at
# * **Over18** - Over 18 years of age?
# * **OverTime** - Overtime?
# * **PercentSalaryHike** - The percentage increase in salary last year
# * **PerformanceRating** - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
# * **RelationshipSatisfaction** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **StandardHours** - Standard Hours
# * **StockOptionLevel** - Stock Option Level
# * **TotalWorkingYears** - Total years worked
# * **TrainingTimesLastYear** - Number of training attended last year
# * **WorkLifeBalance** - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
# * **YearsAtCompany** - Years at Company
# * **YearsInCurrentRole** - Years in the current role
# * **YearsSinceLastPromotion** - Years since the last promotion
# * **YearsWithCurrManager** - Years with the current manager

# %%
# Viewing summary of numerical data
data['employee'].describe().T

# %% [markdown]
# ## Data Preparation / Preprocessing

# %%
# Create a copy of the data
df = data['employee'].copy()

# %% [markdown]
# In this analysis, detected outliers will be trimmed (capping) based on the 5% and 95% percentiles.

# %%
# Cap the outliers
def cap_outliers(df: pd.DataFrame, feature: str, percentile: int):
    # Doing the capping
    lower_bound = df[feature].quantile(percentile)
    upper_bound = df[feature].quantile(1-percentile) 

    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

    return df

for feature in ['MonthlyIncome', 'NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
    df = cap_outliers(df, feature, 0.01)

# Re-assess the data after capping outliers
u.asses_data(df, 'employee')

# %% [markdown]
# Clean missing values

# %%
# Fill missing values in 'Attrition' column with 0
df['Attrition'] = df['Attrition'].fillna(0)

# Ensure 'Attrition' is treated as a categorical variable
df['Attrition'] = df['Attrition'].astype('category')

# Separate the 'Attrition' column from the rest of the DataFrame
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# %%
# Create a text_columns list
text_columns = list(X.select_dtypes(include=['object']).columns)

# Add label encoded columns to text_columns
label_encoded_columns = ['Education', 'StockOptionLevel']
text_columns.extend(label_encoded_columns)

for col in text_columns:
    # Perform one-hot encoding on 'text' columns
    dummies = pd.get_dummies(X[col].astype('object'), prefix=col).astype(int)

    # Concatenate the dummy variables to the original DataFrame
    X = pd.concat([X, dummies], axis=1)

    # Drop the original column
    X.drop(col, axis=1, inplace=True)

# Preview the updated DataFrame
X.head()

# %% [markdown]
# Drop unused column

# %%
X = X.drop(columns=['EmployeeId'])

# %% [markdown]
# Dropping features with high multicollinearity to avoid redundancy

# %%
# Define correlation limit
limit = 0.95
high_correlated_columns = set()

# Create a correlation matrix
correlation_matrix = X.corr(method='pearson')

# Identify features with high multicollinearity
for col in correlation_matrix.columns:
    for key, val in correlation_matrix[col].items():
        if abs(val) > limit and col != key:  # Avoid self-correlation
            high_correlated_columns.add(key)  # Add the correlated column

print(f"Columns with high multicollinearity (>|{limit}|):\n", high_correlated_columns)

# Drop correlated columns from dataframe
X = X.drop(columns=high_correlated_columns)

# Preview the cleaned DataFrame
X.head()

# %% [markdown]
# Drop Features with (Near-)Zero Variance
# 
# Features with very low or near-zero variance are features whose values hardly change across the entire dataset. In other words, these features do not have much variation and do not provide much useful information for the model.

# %%
# Setting the variance limit
limit = 0.2
features = X.var()[X.var() / X.max() > limit].index
print(f"Fitur yang dipertahankan\n{features}")
print(f"Fitur yang didrop\n{X.var()[X.var() / X.max() <= limit].index}")

# Select features
X = X[features]

# Preview the cleaned DataFrame
X.head()

# %%
# Select all columns
columns = X.columns

# Define the number of columns per figure
columns_per_figure = 9

# Split numerical columns into chunks of 9
chunks = [columns[i:i + columns_per_figure] for i in range(0, len(columns), columns_per_figure)]

# %%
# Create a dummy dataframe
df_dummy = pd.concat([X, df['Attrition']], axis=1)

# Loop through each chunk and create a figure
for i, chunk in enumerate(chunks):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Figure {i + 1}: Relative Frequency of Histograms of Numerical Columns', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for j, column in enumerate(chunk):
        ax = axes[j]
        for attrition_value in df_dummy['Attrition'].cat.categories:
            subset = df_dummy[df_dummy['Attrition'] == attrition_value]
            ax.hist(subset[column], bins=20, alpha=0.5, label=f'Attrition {attrition_value}', density=True)
        
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Hide unused subplots
    for k in range(len(chunk), len(axes)):
        fig.delaxes(axes[k])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.show()

# %%
# Create a heatmap correlation matrix
correlation_matrix = X.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Features')
plt.show()

# %% [markdown]
# ## Modeling

# %%
# Scale the data using MinMaxScaler
scaler = MinMaxScaler()

numeric_columns = X.columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# %%
# Scale the data using StandardScaler
scaler = StandardScaler()

numeric_columns = X.columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# %%
# Split the data into training and test sets
def split_data(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X, y)

# %%
# Create models
def create_models(X_train, y_train):
    knn = KNeighborsClassifier().fit(X_train, y_train)
    print('Creating K-Nearest Neighbors (KNN) model done')
    lr = LogisticRegression().fit(X_train, y_train)
    print('Creating Logistic Regression (LR) model done')
    dt = DecisionTreeClassifier().fit(X_train, y_train)
    print('Creating Decision Tree (DT) model done')
    rf = RandomForestClassifier().fit(X_train, y_train)
    print('Creating Random Forest (RF) model done')
    svm = SVC().fit(X_train, y_train)
    print('Creating Support Vector Machine (SVM) model done')
    nb = GaussianNB().fit(X_train, y_train)
    print('Creating Naive Bayes (NB) model done')
    return knn, lr, dt, rf, svm, nb

print(f'{"-"*40}\nCreating models\n{"-"*40}')
knn, lr, dt, rf, svm, nb = create_models(X_train, y_train)
models = {
    'K-Nearest Neighbors (KNN)': knn,
    'Logistic Regression (LR)': lr,
    'Decision Tree (DT)': dt,
    'Random Forest (RF)': rf,
    'Support Vector Machine (SVM)': svm,
    'Naive Bayes (NB)': nb
}
print('done')

# %% [markdown]
# ## Evaluation

# %%
# Evaluate the models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    results = {
        'Confusion Matrix': cm,
        'True Positive (TP)': tp,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn,
        'True Negative (TN)': tn,
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1-Score': f1_score(y_test, y_test_pred),
        'MSE_train': mean_squared_error(y_train, y_train_pred),
        'MSE_test': mean_squared_error(y_test, y_test_pred)
    }
    
    return results

evaluations = {}
rows = []

print(f'{"-"*40}\nEvaluating models\n{"-"*40}')
for name, model in models.items():
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    evaluations[name] = results
    rows.append({
        'Model': name,
        'Accuracy': results['Accuracy'],
        'Precision': results['Precision'],
        'Recall': results['Recall'],
        'F1-Score': results['F1-Score'],
        'MSE_train': results['MSE_train'],
        'MSE_test': results['MSE_train']
    })
    print(f"Evaluating {name} model done")

# Convert the dictionary into a dataframe
summary_df = pd.DataFrame(rows)

# %%
# Display confusion matrix
for name, model in models.items():
    print(f'{"-"*40}\nDisplaying confusion matrix with {name} model\n{"-"*40}')
    fig, ax = plt.subplots()
    cm = evaluations[name]['Confusion Matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    ax.grid(False)

    ax.set_title(f'Confusion Matrix for with {name} model')
    plt.show()

# %%
# Display the performance matrix
summary_df

# %%
# Display learning curve graph
def show_learning_curve(name, model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training error")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation error")
    plt.title(f"Learning Curve with {name} Model")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

for name, model in models.items():
    print(f'{"-"*80}\nLearning curve with {name} model\n{"-"*80}')
    results = show_learning_curve(name, model, X_train, y_train)


