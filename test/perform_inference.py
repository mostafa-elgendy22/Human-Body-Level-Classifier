import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df):
    # Find the columns with continuous values
    continuous_attributes = df.select_dtypes(include=['float64']).columns.tolist()

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

    return df

def remove_uncorrelated_features(df):
    df = df.drop(['Meal_Count'], axis=1)
    return df


df = pd.read_csv('test.csv')
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

df = remove_uncorrelated_features(preprocess(df))

one_hot_encoded = ['Transport_Automobile',	'Transport_Bike'	'Transport_Motorbike',	'Transport_Public_Transportation','Transport_Walking']
columns = df.columns.tolist()

for column in one_hot_encoded:
    if column not in columns:
        df[column] = 0

X_test = df.to_numpy()
y_pred = model.predict(X_test)


y_pred = [f'Body Level {i}' for i in y_pred]

# Write the predictions to a txt file
with open('preds.txt', 'w') as file:
    for prediction in y_pred:
        file.write(prediction + '\n')