import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('HR_comma_sep.csv')
data = data.drop_duplicates()
department_dummies = pd.get_dummies(data.Department)
salary_dummies = pd.get_dummies(data.salary)

data = pd.concat([data, department_dummies, salary_dummies], axis = 'columns')
x = data.drop(['left', 'Department', 'salary'], axis='columns')
y = data.left
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
model = LogisticRegression()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
score = model.score(x_test, y_test)
print(score)

