import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(X, y=None):
    # Find the columns with continuous values
    continuous_attributes = X.select_dtypes(include=['float64']).columns.tolist()
    # continuous_attributes.remove('Meal_Count')

    # Standardize the continuous attributes
    for attribute in continuous_attributes:
        X[attribute] = StandardScaler().fit_transform(X[[attribute]])

    # The columns with categorical values to be encoded using label encoding
    categorical_attributes = ['H_Cal_Consump', 'Alcohol_Consump', 'Smoking', 'Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn']
    for attribute in categorical_attributes:
        label_encoder = LabelEncoder()
        X[attribute] = label_encoder.fit_transform(X[attribute])

    # The columns with categorical values to be encoded using one-hot encoding
    categorical_attributes = ['Gender', 'Transport']

    for attribute in categorical_attributes:
        encoded_attribute = pd.get_dummies(X[attribute], prefix=attribute)
        X.drop([attribute], axis=1, inplace=True)
        X = pd.concat([X, encoded_attribute], axis=1)

    df = X

    if y is not None:
        # Convert the string labels to integer labels
        for i in range(len(y)):
            y[i] = int(y[i][-1])

        df['Body_Level'] = y

    return df