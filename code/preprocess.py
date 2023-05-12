import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(X, y):
    # Find the columns with categorical values
    categorical_attributes = X.select_dtypes(include=['object']).columns.tolist()


    # Find the columns with continuous values
    continuous_attributes = X.select_dtypes(include=['float64']).columns.tolist()

    # Normalize the continuous attributes to be in the range of 0 to 1
    for attribute in continuous_attributes:
        X[attribute] = MinMaxScaler().fit_transform(X[[attribute]])

    # Check if the continuous attributes are normalized
    for attribute in continuous_attributes:
        assert X[attribute].min() == 0
        assert (X[attribute].max() >= 0.999999999999999 and X[attribute].max() <= 1.0)

    # Convert the categorical attributes to one-hot encoding
    for attribute in categorical_attributes:
        encoded_attribute = pd.get_dummies(X[attribute], prefix=attribute)
        X.drop([attribute], axis=1, inplace=True)
        X = pd.concat([X, encoded_attribute], axis=1)

    df = X

    # Convert the string labels to integer labels
    for i in range(len(y)):
        y[i] = y[i][-1]

    df['Body_Level'] = y

    return df
