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

# Define mapping dictionaries

# How old are you?
CR1_dict = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17}

# What is your sex?
CR2_dict = {1: "Male", 2: "Female"}

# Do you want to stop smoking now?
# CR15_dict = {1: False, 2: False, 3: True, 4: True}

# Please think about the days you smoked cigarettes during the past 30 days (one month). How many cigarettes did you usually smoke per day?
CR8_smoke_dict = {1: False, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True}
CR8_dict = {1: "0", 2: "Less than 1", 3: "1", 4: "2 to 5", 5: "6 to 10", 6: "11 to 20", 7: "More than 20"}

# Do your parents smoke tobacco?
OR45_dict = {1: "None", 2: "Both", 3: "Father only", 4: "Mother only", 5: "Don't know"}

# Do any of your closest friends smoke tobacco?
OR46_dict = {1: "None of them", 2: "Some of them", 3: "Most of them", 4: "All of them"}

# Do your parents work?
OR1_dict = {1: "Father only", 2: "Mother only", 3: "Both", 4: "Neither", 5: "Don't know"}

# During the past 30 days, did you see anyone smoke inside the school building or outside on the school property?
CR22_dict = {1: "Yes", 2: "No"}

# During the past 7 days, on how many days has anyone smoked in your presence, at outdoor public places (playgrounds, sidewalks, entrances to buildings, parks, beaches, swimming pools)?
CR21_dict = {1: "0 days", 2: "1 to 2 days", 3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}

# During the past 7 days, on how many days has anyone smoked in your presence, inside enclosed public
CR20_dict = {1: "0 days", 2: "1 to 2 days", 3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}

# During the past 7 days, on how many days has anyone smoked in your presence, inside your home
CR19_dict = {1: "0 days", 2: "1 to 2 days", 3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}

# Have you ever tried smoking cigarettes?
CR5_dict = {1: "Yes", 2: "No"}

# How old were you when you first tried smoking?
CR6_dict = {1: "Never tried", 2: "7 years old or younger", 3: "8 or 9 years old", 4: "10 or 11 years old", 5: "12 or 13 years old", 6: "14 or 15 years old", 7: "16 years old or older"}

# rename columns
merged_df.rename(columns={"CR1": "Age"}, inplace=True)
merged_df.rename(columns={"CR2": "Gender"}, inplace=True)
# merged_df.rename(columns={"CR8": "Smoke"}, inplace=True)
merged_df.rename(columns={"OR45": "SmokingParents"}, inplace=True)
merged_df.rename(columns={"OR46": "SmokingFriends"}, inplace=True)
merged_df.rename(columns={"OR1": "WorkingParents"}, inplace=True)
merged_df.rename(columns={"CR22": "SeenSmokerInSchool"}, inplace=True)
merged_df.rename(columns={"CR21": "SeenSmokerInPublicPlace"}, inplace=True)
merged_df.rename(columns={"CR20": "SeenSmokerInEnclosedPlace"}, inplace=True)
merged_df.rename(columns={"CR19": "SeenSmokerInHome"}, inplace=True)
merged_df.rename(columns={"CR5": "TriedCigarette"}, inplace=True)
merged_df.rename(columns={"CR6": "AgeFirstCigarette"}, inplace=True)

# Map values and drop unnecessary columns
# map_values_and_drop_columns(merged_df, CR1_dict, "CR1","Age")
# map_values_and_drop_columns(merged_df, CR2_dict, "CR2","Gender")
# # map_values_and_drop_columns(merged_df, CR15_dict, "CR15")
map_values_and_drop_columns(merged_df, CR8_smoke_dict, "CR8", "Smoke")
# map_values_and_drop_columns(merged_df, OR45_dict, "OR45","SmokingParents")
# map_values_and_drop_columns(merged_df, OR46_dict, "OR46","SmokingFriends")
# map_values_and_drop_columns(merged_df, OR1_dict, "OR1","WorkingParents")
# map_values_and_drop_columns(merged_df, CR22_dict, "CR22","SeenSmokerInSchool")
# map_values_and_drop_columns(merged_df, CR21_dict, "CR21","SeenSmokerInPublicPlace")
# map_values_and_drop_columns(merged_df, CR20_dict, "CR20","SeenSmokerInEnclosedPlace")
# map_values_and_drop_columns(merged_df, CR19_dict, "CR19","SeenSmokerInHome")
# map_values_and_drop_columns(merged_df, CR5_dict, "CR5","TriedCigarette")
# map_values_and_drop_columns(merged_df, CR6_dict, "CR6","AgeFirstCigarette")

# Keep only the desired columns
# merged_df = merged_df[["State", "Gender", "Age", "Smoke", "SmokingParents", "SmokingFriends", "WorkingParents",
#                        "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace",
#                        "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"]]
merged_df = merged_df[["State", "Gender", "Age", "Smoke", "SmokingParents", "SmokingFriends", "WorkingParents",
                       "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace",
                       "SeenSmokerInHome"]]
# Drop rows with missing values
merged_df = merged_df.dropna()

# convert columns to int
# merged_df['Age'] = merged_df['Age'].astype('int')

# convert columns to categorical
merged_df['State'] = merged_df['State'].astype('int').astype('category')
merged_df["Gender"] = merged_df["Gender"].astype('int').astype('category')
merged_df["Age"] = merged_df["Age"].astype('int').astype('category')
merged_df["Smoke"] = merged_df["Smoke"].astype('int').astype('category')
merged_df["SmokingParents"] = merged_df["SmokingParents"].astype('int').astype('category')
merged_df["SmokingFriends"] = merged_df["SmokingFriends"].astype('int').astype('category')
merged_df["WorkingParents"] = merged_df["WorkingParents"].astype('int').astype('category')
merged_df["SeenSmokerInSchool"] = merged_df["SeenSmokerInSchool"].astype('int').astype('category')
merged_df["SeenSmokerInPublicPlace"] = merged_df["SeenSmokerInPublicPlace"].astype('int').astype('category')
merged_df["SeenSmokerInEnclosedPlace"] = merged_df["SeenSmokerInEnclosedPlace"].astype('int').astype('category')
merged_df["SeenSmokerInHome"] = merged_df["SeenSmokerInHome"].astype('int').astype('category')
# merged_df["TriedCigarette"] = merged_df["TriedCigarette"].astype('category')
# merged_df["AgeFirstCigarette"] = merged_df["AgeFirstCigarette"].astype('category')

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

train, test = train_test_split(merged_df_encoded, test_size=0.2, random_state=42)

test.reset_index(drop=True, inplace=True)
# Set up a Machine Learning model to predict if a person smokes or not
# X = merged_df_encoded.drop(columns=["Smoke"])
# y = merged_df_encoded["Smoke"]
X = train.drop(columns=["Smoke"])
y = train["Smoke"]

# Convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

smote = SMOTEN()
# smote = ADASYN()
# smote = RandomUnderSampler()
X_resampled, y_resampled = smote.fit_resample(X, y)

# #convert array to dataframe
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

exp_clf102 = pc.setup(data=df_resampled,
                      target='Smoke',
                      session_id=123,
                      normalize=True,
                      transformation=True,
                      remove_multicollinearity=True, multicollinearity_threshold=0.95, max_encoding_ohe=0)

pc.compare_models()

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

# setup parameter space
# parameters = {'criterion':['gini','entropy'],
#             'max_depth':np.arange(1,21).tolist()[0::2],
#             'min_samples_split':np.arange(2,11).tolist()[0::2],
#             'max_leaf_nodes':np.arange(3,26).tolist()[0::2]}

# parameters = {'criterion': Categorical(['gini','entropy']),
#           'max_depth': Integer(1,21,prior='log-uniform'),
#           'min_samples_split': Real(1e-3,1.0,prior='log-uniform'),
#           'max_leaf_nodes': Integer(3,26,prior='uniform')}

# # create an instance of the grid search object
# g1 = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_jobs=-1)
# g1 = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=1000, random_state=42, n_jobs=-1)
# g1 = BayesSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=10, random_state=42, n_jobs=-1)

# # conduct grid search over the parameter space
# start_time = time.time()
# # g1.fit(X_train, y_train)
# duration = time.time() - start_time

# show best parameter configuration found for classifier
# cls_params1 = g1.best_params_
# print(cls_params1)

# Define the model
# model = KNeighborsClassifier()
# model = ExtraTreesClassifier(n_estimators = 5, criterion ='entropy', max_features = 2, random_state=42)
# model = GradientBoostingClassifier(n_estimators=300,
#                                     learning_rate=0.05,
#                                     random_state=101,
#                                     max_features=5 )
# model = KNeighborsClassifier(n_neighbors=100)
# model = GaussianNB()
# model = DecisionTreeClassifier()
# model = g1.best_estimator_

# Train the model
# model.fit(X_train, y_train)

# Make predictions
# y_pred = model.predict(X_test)

# print(model.weights)
# Computing the importance of each feature
# feature_importance = model.feature_importances_

# Normalizing the individual importances
# feature_importance_normalized = np.std([tree.feature_importances_ for tree in
#                                         model.estimators_],
#                                         axis = 0)

# Plotting a Bar Graph to compare the models
# plt.bar(X.columns, feature_importance_normalized)
# plt.xlabel('Feature Labels')
# plt.ylabel('Feature Importances')
# plt.title('Comparison of different Feature Importances')
# plt.show()

# Evaluate the model
# print('Accuracy score:', accuracy_score(y_test,y_pred))
# print('Precision score:', precision_score(y_test,y_pred))
# print('Recall score:', recall_score(y_test,y_pred))
# print('F1 score:', f1_score(y_test,y_pred))
# print('Computation time:', duration)

# # Confusion matrix
# confusion_mat = confusion_matrix(y_test, y_pred)
# print(confusion_mat)

# # Correlation matrix
# corr = merged_df_encoded.corr()
# fig, ax = plt.subplots(figsize=(40, 30))
# sns.heatmap(corr, annot=True, annot_kws={"size": 8}, linewidths=.5, ax=ax)
# plt.savefig("./data/processed/correlation_matrix.svg")
# # plt.show()

# sns.pairplot(data=merged_df_encoded , corner=True)
# plt.show()


# if __name__ == "__main__":
#     main()
