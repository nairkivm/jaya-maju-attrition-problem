class Constants:
    def __init__(self) -> None:
        self.source = {
            'employee' : 'data_sources/employee_data.csv'
        }
        self.requirements = {
            'employee': {
                'EmployeeId': 'int64',
                'Age': 'int64',
                'Attrition': 'float64',
                'BusinessTravel': 'object',
                'DailyRate': 'int64',
                'Department': 'object',
                'DistanceFromHome': 'int64',
                'Education': 'int64',
                'EducationField': 'object',
                'EmployeeCount': 'int64',
                'EnvironmentSatisfaction': 'int64',
                'Gender': 'object',
                'HourlyRate': 'int64',
                'JobInvolvement': 'int64',
                'JobLevel': 'int64',
                'JobRole': 'object',
                'JobSatisfaction': 'int64',
                'MaritalStatus': 'object',
                'MonthlyIncome': 'int64',
                'MonthlyRate': 'int64',
                'NumCompaniesWorked': 'int64',
                'Over18': 'object',
                'OverTime': 'object',
                'PercentSalaryHike': 'int64',
                'PerformanceRating': 'int64',
                'RelationshipSatisfaction': 'int64',
                'StandardHours': 'int64',
                'StockOptionLevel': 'int64',
                'TotalWorkingYears': 'int64',
                'TrainingTimesLastYear': 'int64',
                'WorkLifeBalance': 'int64',
                'YearsAtCompany': 'int64',
                'YearsInCurrentRole': 'int64',
                'YearsSinceLastPromotion': 'int64',
                'YearsWithCurrManager': 'int64'
            }
        }
        