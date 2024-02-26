import pandas as pd 
import os

# df = pd.read_sas("./data/raw/GATS/GATS_Greece_National_2013_SAS/GREECE_PUBLIC_USE_11Mar2015.sas7bdat")

# df.to_csv("./data/processed/GATS_Greece_National_2013_SAS.csv")
# print(df.head())


# list file in folder ./data/raw/GYTS/
fileList = [f for f in os.listdir("./data/raw/GYTS/") if f.endswith('.csv')]

# remove greece and finland
fileList = fileList[2:]

# print(fileList)

dataframes = {file: pd.read_csv("./data/raw/GYTS/" + file) for file in fileList}
    
# check which columns are in common in all files and print them
common_columns = dataframes[fileList[0]].columns
for df in dataframes.values():
    common_columns = common_columns.intersection(df.columns)
print(common_columns)

# common_columns = common_columns.delete([0,-1,-2])



merged_df = pd.DataFrame()
for file, df in dataframes.items():
    # select only the common columns
    df = df[common_columns]
    
    # add a column to each dataframe with the name of the file
    df.insert(0, "State", file.split(" ")[2])
    # df["State"] = file.split(" ")[2]
    # append the dataframe to the merged_df
    merged_df = pd.concat([merged_df, df])
    
    
# save the merged dataframe to a csv file
# merged_df.to_csv("./data/processed/merged_GYTS.csv", index=False)

# age question
CR1_dict = {
    1: 11,
    2: 12,
    3: 13,
    4: 14,
    5: 15,
    6: 16,
    7: 17,
}

CR2_dict = {
    1: "Male",
    2: "Female",
}

# CR15_dict = {
#     1: False,
#     2: False,
#     3: True,
#     4: True,
# }

CR8_smoke_dict = {
    1: False,
    2: True,
    3: True,
    4: True,
    5: True,
    6: True,
    7: True
}

CR8_dict = {
    1: "0",
    2: "Less than 1",
    3: "1",
    4: "2 to 5",
    5: "6 to 10",
    6: "11 to 20",
    7: "More than 20"
}

OR45_dict = {
    1: "None",
    2: "Both",
    3: "Father only",
    4: "Mother only",
    5: "Don't know"
}

OR46_dict = {
    1: "None of them", 
    2: "Some of them", 
    3: "Most of them", 
    4: "All of them", 
}
    
OR1_dict = {
    1: "Father only",
    2: "Mother only",
    3: "Both",
    4: "Neither",
    5: "Don't know"
}

CR22_dict = {
    1: "Yes",
    2: "No",
}

CR21_dict = {
    1: "0 days",
    2: "1 to 2 days",
    3: "3 to 4 days",
    4: "5 to 6 days",
    5: "7 days",
}

CR20_dict = {
    1: "0 days",
    2: "1 to 2 days",
    3: "3 to 4 days",
    4: "5 to 6 days",
    5: "7 days",
}

CR19_dict = {
    1: "0 days",
    2: "1 to 2 days",
    3: "3 to 4 days",
    4: "5 to 6 days",
    5: "7 days",
}

CR5_dict = {
    1: "Yes",
    2: "No"
}

CR6_dict = {
    1: "Never tried",
    2: "7 years old or younger",
    3: "8 or 9 years old",
    4: "10 or 11 years old",
    5: "12 or 13 years old",
    6: "14 or 15 years old",
    7: "16 years old or older"
}

merged_df["Age"] = merged_df["CR1"].map(CR1_dict)
merged_df.drop(columns=["CR1"], inplace=True)

merged_df["Gender"] = merged_df["CR2"].map(CR2_dict)
merged_df.drop(columns=["CR2"], inplace=True)


# merged_df["Smoke"] = merged_df["CR15"].map(CR15_dict)
# merged_df.drop(columns=["CR15"], inplace=True)


merged_df["CigarettesPerDay"] = merged_df["CR8"].map(CR8_dict)
merged_df["Smoke"] = merged_df["CR8"].map(CR8_smoke_dict)
merged_df.drop(columns=["CR8"], inplace=True)

merged_df["SmokingParents"] = merged_df["OR45"].map(OR45_dict)
merged_df.drop(columns=["OR45"], inplace=True)

merged_df["SmokingFriends"] = merged_df["OR46"].map(OR46_dict)
merged_df.drop(columns=["OR46"], inplace=True)

merged_df["WorkingParents"] = merged_df["OR1"].map(OR1_dict)
merged_df.drop(columns=["OR1"], inplace=True)

# During the past 30 days, did you see anyone smoke inside the school building or outside on school property (yard, playground, garden, parking, etc.)?
merged_df["SeenSmokerInSchool"] = merged_df["CR22"].map(CR22_dict)
merged_df.drop(columns=["CR22"], inplace=True)

# During the past 7 days, on how many days has anyone smoked in your presence, at any outdoor public place (such as playgrounds, parks, beaches, entrance to buildings, stadium, bus stops)?
merged_df["SeenSmokerInPublicPlace"] = merged_df["CR21"].map(CR21_dict)
merged_df.drop(columns=["CR21"], inplace=True)

# During the past 7 days, on how many days has anyone smoked in your presence, inside any enclosed public place, other than your home (such as shops, restaurants, shopping malls, movie theaters)?
merged_df["SeenSmokerInEnclosedPlace"] = merged_df["CR20"].map(CR20_dict)
merged_df.drop(columns=["CR20"], inplace=True)

# During the past 7 days, on how many days has anyone smoked inside your home, in your presence?
merged_df["SeenSmokerInHome"] = merged_df["CR19"].map(CR19_dict)
merged_df.drop(columns=["CR19"], inplace=True)

# Have you ever tried or experimented with cigarette smoking, even one or two puffs?
merged_df["TriedCigarette"] = merged_df["CR5"].map(CR5_dict)
merged_df.drop(columns=["CR5"], inplace=True)

# How old were you when you first tried a cigarette?
merged_df["AgeFirstCigarette"] = merged_df["CR6"].map(CR6_dict)
merged_df.drop(columns=["CR6"], inplace=True)


# keep only the columns we are interested in
# merged_df = merged_df[["State", "Gender", "Age", "Smoke", "CigarettesPerDay", "SmokingParents", "SmokingFriends", "WorkingParents", "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace", "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"]]
merged_df = merged_df[["State", "Gender", "Age", "Smoke", "SmokingParents", "SmokingFriends", "WorkingParents", "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace", "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"]]

# remove all rows with missing values
merged_df = merged_df.dropna()

print(merged_df.head())
# save the merged dataframe to a csv file
merged_df.to_csv("./data/processed/merged_GYTS.csv", index=False)


# One hot encoding
# merged_df = pd.get_dummies(merged_df, columns=["State", "Gender", "CigarettesPerDay", "SmokingParents", "SmokingFriends", "WorkingParents", "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace", "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"])
merged_df = pd.get_dummies(merged_df, drop_first=True , columns=["State", "Gender", "SmokingParents", "SmokingFriends", "WorkingParents", "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace", "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"])

# save the merged dataframe to a csv file
merged_df.to_csv("./data/processed/merged_GYTS_encoded.csv", index=False)
                                



# set a Machine Learning model to predict if a person smokes or not
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# define the features and the target
X = merged_df.drop(columns=["Smoke"])
y = merged_df["Smoke"]

from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

#view transformed values
print(y)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the model
model = RandomForestClassifier(n_estimators=100)

# train the model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# view the feature importances
feature_importances = model.feature_importances_
feature_names = X.columns
importances = list(zip(feature_names, feature_importances))

# sort the feature importances by importance

importances.sort(key=lambda x: x[1], reverse=True)
print(importances)

# plot the feature importances
import matplotlib.pyplot as plt
import numpy as np

# horizontal bar chart
x_values = list(range(len(feature_importances)))
plt.bar(x_values, feature_importances, orientation = 'vertical')
plt.xticks(x_values, feature_names, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# Correlation matrix of model
import seaborn as sns
corr = merged_df.corr()

fig, ax = plt.subplots(figsize=(40,30))         # Sample figsize in inches
sns.heatmap(corr, annot=True,annot_kws={"size": 8}, linewidths=.5, ax=ax)
# sns.heatmap(corr, annot=True)
# save svg
plt.savefig("./data/processed/correlation_matrix.svg")
plt.show()




    
    
    
    





  


