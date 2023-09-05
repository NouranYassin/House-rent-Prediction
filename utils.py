# main
import numpy as np
import pandas as pd
import os
# secondary
from datasist.structdata import detect_outliers


## sklearn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


# getting data
Train_path = os.path.join(os.getcwd(), 'House_Rent_Dataset.csv')
df = pd.read_csv(Train_path)
df.columns = df.columns.str.replace(' ', '_')
df['Posted_On'] = pd.to_datetime(df['Posted_On'])
lst = []
for i in df['Point_of_Contact']:
    lst.append(i.split()[1])
# apply
df['Point_of_Contact'] = lst
df['day_posted_on'] = df['Posted_On'].dt.day
df['month_Posted_on'] = df['Posted_On'].dt.month
df['year_posted_on'] = df['Posted_On'].dt.year

# minimize "size" feature to 4 unique values


def Size(value):
    try:
        if value < 100:
            return "small"
        elif value < 500:
            return "Intermediate"
        elif value < 1000:
            return "UpperIntermediate"
        else:
            return "large"
    except:
        np.nan


df['size_category'] = df['Size'].apply(Size)

handle_cols = ['Size']
for col in handle_cols:
    ids_outliers = detect_outliers(data=df, n=0, features=[col])
    col_median = df[col].median()

    df.loc[ids_outliers, col] = col_median


df.drop(columns=['Posted_On', 'Floor', 'Area_Locality',
        'year_posted_on'], axis=1, inplace=True)
df['BHK'] = df['BHK'].astype(str)
df['day_posted_on'] = df['day_posted_on'].astype(str)
df['month_Posted_on'] = df['month_Posted_on'].astype(str)
df['Bathroom'] = df['Bathroom'].astype(str)

# Split to features and target
X = df.drop(columns='Rent', axis=1)
y = df['Rent']

# Split to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=45)

# slice the lists
num_cols = X_train.select_dtypes(include='number').columns.tolist()

categ_cols_nominal = ['Area_Type', 'City', 'Tenant_Preferred',
                      'Point_of_Contact', 'BHK', 'Bathroom', 'month_Posted_on', 'day_posted_on']
categ_cols_ordinal = ['Furnishing_Status', 'size_category']


# My Pipeline:


# Numerical -------------> Impute (median), standrize (RobustScaler)
# Categorical: Nominal --> Impute (mode), Encoding Label Encoding
# Categorical: Ordinal --> Impute (mode), Encoding Ordinal Encoding

# For Numerical
num_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ('Imputer', SimpleImputer(strategy='median')),
    ('standrdize', RobustScaler())
])


# For Categorical: Nominal
categ_nominal_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols_nominal)),
    ('imputer', SimpleImputer(
        strategy='most_frequent')),
    ('Encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# For Categorical: Ordinal
categ_ordinal_pipeline1 = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols_ordinal)),
    ('imputer', SimpleImputer(
        strategy='most_frequent')),
    ('encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])


# Combine all
all_pipeline = FeatureUnion(transformer_list=[
    ('numerical', num_pipeline),
    ('categ_nominal', categ_nominal_pipeline),
    ('categ_ordinal1', categ_ordinal_pipeline1),


])

# apply
all_pipeline.fit_transform(X_train)

def process_new(X_new):

    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns[:-1]

    df_new['BHK'] = df_new['BHK'].astype(str)
    df_new['Size'] = df_new['Size'].astype(float)
    df_new['Area_Type'] = df_new['Area_Type'].astype(str)
    df_new['City'] = df_new['City'].astype(str)
    df_new['Furnishing_Status'] = df_new['Furnishing_Status'].astype(str)
    df_new['Tenant_Preferred'] = df_new['Tenant_Preferred'].astype(str)
    df_new['Bathroom'] = df_new['Bathroom'].astype(str)
    df_new['Point_of_Contact'] = df_new['Point_of_Contact'].astype(str)
    df_new['day_posted_on'] = df_new['day_posted_on'].astype(int)
    df_new['month_Posted_on'] = df_new['month_Posted_on'].astype(int)
    df_new['size_category'] = df_new['Size'].apply(Size).astype(str)


    return all_pipeline.transform(df_new)
