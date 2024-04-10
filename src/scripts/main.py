import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycaret.classification as pc
import seaborn as sns
from imblearn.over_sampling import SMOTEN, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scripts.mapping_answers_dict import CR8_smoke_dict, OR45_dict
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


def read_dataframes(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_list = file_list[2:]
    dataframes = {file: pd.read_csv(os.path.join(folder_path, file)) for file in file_list}
    return dataframes


def get_common_columns(dataframes):
    common_columns = dataframes[list(dataframes.keys())[0]].columns
    for df in dataframes.values():
        common_columns = common_columns.intersection(df.columns)
    return common_columns


def preprocess_dataframes(dataframes, common_columns):
    merged_df = pd.DataFrame()
    for i, (file, df) in enumerate(dataframes.items()):
        df = df[common_columns]
        # df.insert(0, "State", file.split(" ")[2])
        df.insert(0, "State", i)
        merged_df = pd.concat([merged_df, df])
    return merged_df


def map_values_and_drop_columns(df, mapping_dict, column_name, new_column_name=None):
    df[new_column_name] = df[column_name].map(mapping_dict)
    df.drop(columns=[column_name], inplace=True)


# def main():
# Read dataframes from GYTS folder
dataframes = read_dataframes("./data/raw/GYTS/")

# Find common columns in all dataframes
common_columns = get_common_columns(dataframes)

# Preprocess dataframes
merged_df = preprocess_dataframes(dataframes, common_columns)


# Rename columns
merged_df.rename(columns={"CR1": "Age"}, inplace=True)
merged_df.rename(columns={"CR2": "Gender"}, inplace=True)
merged_df["CR8"] = merged_df["CR8"].map(CR8_smoke_dict)
merged_df.rename(columns={"CR8": "Smoke"}, inplace=True)
merged_df.rename(columns={"OR45": "SmokingParents"}, inplace=True)
merged_df.rename(columns={"OR46": "SmokingFriends"}, inplace=True)
merged_df.rename(columns={"OR1": "WorkingParents"}, inplace=True)
merged_df.rename(columns={"CR22": "SeenSmokerInSchool"}, inplace=True)
merged_df.rename(columns={"CR21": "SeenSmokerInPublicPlace"}, inplace=True)
merged_df.rename(columns={"CR20": "SeenSmokerInEnclosedPlace"}, inplace=True)
merged_df.rename(columns={"CR19": "SeenSmokerInHome"}, inplace=True)
merged_df.rename(columns={"CR5": "TriedCigarette"}, inplace=True)
merged_df.rename(columns={"CR6": "AgeFirstCigarette"}, inplace=True)


# Keep only the desired columns
# merged_df = merged_df[["State", "Gender", "Age", "Smoke", "SmokingParents", "SmokingFriends", "WorkingParents",
#                        "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace",
#                        "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"]]
merged_df = merged_df[["State", "Gender", "Age", "Smoke", "SmokingParents", "SmokingFriends", "WorkingParents",
                       "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace",
                       "SeenSmokerInHome"]]

# Drop rows with missing values
merged_df = merged_df.dropna()

merged_df['SmokingFather'] = merged_df['SmokingParents'].apply(lambda x: True if OR45_dict[x] in ['Both', 'Father only'] else False)
merged_df['SmokingMother'] = merged_df['SmokingParents'].apply(lambda x: True if OR45_dict[x] in ['Both', 'Mother only'] else False)
merged_df = merged_df.drop(columns=['SmokingParents'])

# Convert columns to categorical
merged_df['State'] = merged_df['State'].astype('int').astype('category')
merged_df["Gender"] = merged_df["Gender"].astype('int').astype('category')
merged_df["Age"] = merged_df["Age"].astype('int').astype('category')
merged_df["Smoke"] = merged_df["Smoke"].astype('int').astype('category')
#merged_df["SmokingParents"] = merged_df["SmokingParents"].astype('int').astype('category')
merged_df["SmokingFriends"] = merged_df["SmokingFriends"].astype('int').astype('category')
merged_df["WorkingParents"] = merged_df["WorkingParents"].astype('int').astype('category')
merged_df["SeenSmokerInSchool"] = merged_df["SeenSmokerInSchool"].astype('int').astype('category')
merged_df["SeenSmokerInPublicPlace"] = merged_df["SeenSmokerInPublicPlace"].astype('int').astype('category')
merged_df["SeenSmokerInEnclosedPlace"] = merged_df["SeenSmokerInEnclosedPlace"].astype('int').astype('category')
merged_df["SeenSmokerInHome"] = merged_df["SeenSmokerInHome"].astype('int').astype('category')
# merged_df["TriedCigarette"] = merged_df["TriedCigarette"].astype('category')
# merged_df["AgeFirstCigarette"] = merged_df["AgeFirstCigarette"].astype('category')

# Convert to boolean
merged_df['SmokingFather'] = merged_df['SmokingFather'].astype('bool')
merged_df['SmokingMother'] = merged_df['SmokingMother'].astype('bool')

# Save the preprocessed dataframe to a CSV file
merged_df.to_csv("./data/processed/merged_GYTS.csv", index=False)

merged_df_encoded = merged_df

# One hot encoding
# merged_df_encoded = pd.get_dummies(merged_df, drop_first=True,
#                                    columns=["State", "Gender", "SmokingParents", "SmokingFriends",
#                                             "WorkingParents", "SeenSmokerInSchool", "SeenSmokerInPublicPlace",
#                                             "SeenSmokerInEnclosedPlace", "SeenSmokerInHome", "TriedCigarette",
#                                             "AgeFirstCigarette"])

# Save the encoded dataframe to a CSV file
merged_df_encoded.to_csv("./data/processed/merged_GYTS_encoded.csv", index=False)

# # Correlation matrix
# corr = merged_df_encoded.corr()
# fig, ax = plt.subplots(figsize=(40, 30))
# sns.heatmap(corr, annot=True, annot_kws={"size": 8}, linewidths=.5, ax=ax)
# plt.savefig("./data/processed/correlation_matrix.svg")
# # plt.show()

# Set up a Machine Learning model to predict if a person smokes or not
train, test = train_test_split(merged_df_encoded, test_size=0.2, random_state=42)
test.reset_index(drop=True, inplace=True)

# ONE HOT ENCODING
# X = merged_df_encoded.drop(columns=["Smoke"])
# y = merged_df_encoded["Smoke"]

X = train.drop(columns=["Smoke"])
y = train["Smoke"]

# Convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

# SMOTE Oversampling
smote = SMOTEN()
# smote = ADASYN()
# smote = RandomUnderSampler()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert array to dataframe
y_resampled = pd.DataFrame(y_resampled, columns=['Smoke'])
# y = pd.DataFrame(y, columns=['Smoke'])

# remove index
# X.reset_index(drop=True, inplace=True)
# y.reset_index(drop=True, inplace=True)
X_resampled.reset_index(drop=True, inplace=True)

df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
# df_resampled = pd.concat([X, y], axis=1)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train, test = train_test_split(df_resampled, test_size=0.2, random_state=42)

setup = pc.setup(data=df_resampled,
                      target='Smoke',
                      session_id=123,
                      normalize=True,
                      transformation=True,
                      remove_multicollinearity=True, multicollinearity_threshold=0.95, max_encoding_ohe=0)

pc.compare_models()

# Extra Trees Classifier
model = pc.create_model('et')

pc.plot_model(model, plot='auc')
pc.plot_model(model, plot='pr')
pc.plot_model(model, plot='feature')
pc.plot_model(model, plot='confusion_matrix')

final_rf = pc.finalize_model(model)
# final_rf
pc.predict_model(final_rf)
# print(final_rf)

unseen_predictions = pc.predict_model(final_rf, data=test)
print(unseen_predictions.head())

pc.interpret_model(model, plot='summary')



# sns.pairplot(data=merged_df_encoded , corner=True)
# plt.show()


# if __name__ == "__main__":
#     main()
