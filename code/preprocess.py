import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df):
    # Find the columns with continuous values
    continuous_attributes = df.select_dtypes(include=['float64']).columns.tolist()
    # continuous_attributes.remove('Meal_Count')

    # Standardize the continuous attributes
    for attribute in continuous_attributes:
        df[attribute] = StandardScaler().fit_transform(df[[attribute]])

    # The columns with categorical values to be encoded using label encoding
    categorical_attributes = ['H_Cal_Consump', 'Alcohol_Consump', 'Smoking', 'Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn']
    for attribute in categorical_attributes:
        label_encoder = LabelEncoder()
        df[attribute] = label_encoder.fit_transform(df[attribute])

    # The columns with categorical values to be encoded using one-hot encoding
    categorical_attributes = ['Gender', 'Transport']

    for attribute in categorical_attributes:
        encoded_attribute = pd.get_dummies(df[attribute], prefix=attribute)
        df.drop([attribute], axis=1, inplace=True)
        df = pd.concat([df, encoded_attribute], axis=1)


    if 'Body_Level' in df.columns:
        # Convert the Body_Level column to integers
        # map 'Body Level 1' to 1, 'Body Level 2' to 2, 'Body Level 3' to 3
        df['Body_Level'] = df['Body_Level'].map({'Body Level 1': 1, 'Body Level 2': 2, 'Body Level 3': 3, 'Body Level 4': 4})

    return df

def remove_uncorrelated_features(df):
    # df = df.drop(['Veg_Consump', 'Meal_Count', 'Smoking'], axis=1)
    df = df.drop(['Meal_Count'], axis=1)
    return df