import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
df_forest = pd.read_csv("forestfires.csv")
df_forest['area'] = np.log(df_forest['area'] + 1)

from sklearn.preprocessing import LabelEncoder
categorical = list(df_forest.select_dtypes(include = ["object"]).columns)

for i, column in enumerate(categorical) :
    label = LabelEncoder()
    df_forest[column] = label.fit_transform(df_forest[column])


outcome = df_forest['area']
features = df_forest.iloc[:, :-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size = 0.15, random_state = 196)

from sklearn.svm import SVR
model_4 = SVR(C = 100, kernel = 'linear')
model_4.fit(X_train, y_train)

prediction = model_4.predict(X_test)
mean_squared_error(y_test, prediction)
prediction = np.exp(prediction - 1)

import pickle
pickle_out = open("firemodel.pkl", "wb")
pickle.dump(model_4, pickle_out)
pickle_out.close()
