import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read dataset
data = pd.read_csv('data/Social_Network_Ads.csv')

# Drop unnecessary column
data = data.drop(columns=['User ID'])

# Run one-hot encoding process with get_dummies()
data = pd.get_dummies(data)

# Separate attribute and label
x = data[['Age', 'EstimatedSalary', 'Gender_Female', 'Gender_Male']]
y = data['Purchased']

# Data normalization
scaler = StandardScaler()
scaler.fit(x)

scaled_data = scaler.transform(x)
scaled_data = pd.DataFrame(scaled_data, columns=x.columns)

# Divide data into training and testing
x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1)

# Train model
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

# Model accuracy test
print(model.score(x_test, y_test))
