import time
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    # Read dataframes from GYTS folder
    dataframes = read_dataframes("./data/raw/GYTS/")

    # Find common columns in all dataframes
    common_columns = get_common_columns(dataframes)

    # Preprocess dataframes
    merged_df = preprocess_dataframes(dataframes, common_columns)

    # Define mapping dictionaries
    
    CR1_dict = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17}
    CR2_dict = {1: "Male", 2: "Female"}
    # CR15_dict = {1: False, 2: False, 3: True, 4: True}
    CR8_smoke_dict = {1: False, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True}
    CR8_dict = {1: "0", 2: "Less than 1", 3: "1", 4: "2 to 5", 5: "6 to 10", 6: "11 to 20", 7: "More than 20"}
    OR45_dict = {1: "None", 2: "Both", 3: "Father only", 4: "Mother only", 5: "Don't know"}
    OR46_dict = {1: "None of them", 2: "Some of them", 3: "Most of them", 4: "All of them"}
    OR1_dict = {1: "Father only", 2: "Mother only", 3: "Both", 4: "Neither", 5: "Don't know"}
    CR22_dict = {1: "Yes", 2: "No"}
    CR21_dict = {1: "0 days", 2: "1 to 2 days", 3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}
    CR20_dict = {1: "0 days", 2: "1 to 2 days", 3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}
    CR19_dict = {1: "0 days",2: "1 to 2 days",3: "3 to 4 days",4: "5 to 6 days",5: "7 days"}
    CR5_dict = {1: "Yes", 2: "No"}
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
    map_values_and_drop_columns(merged_df, CR8_smoke_dict, "CR8","Smoke")
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
    merged_df = merged_df[["State", "Gender", "Age", "Smoke", "SmokingParents", "SmokingFriends", "WorkingParents",
                           "SeenSmokerInSchool", "SeenSmokerInPublicPlace", "SeenSmokerInEnclosedPlace",
                           "SeenSmokerInHome", "TriedCigarette", "AgeFirstCigarette"]]

    # Drop rows with missing values
    merged_df = merged_df.dropna()

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

    # Set up a Machine Learning model to predict if a person smokes or not
    X = merged_df_encoded.drop(columns=["Smoke"])
    y = merged_df_encoded["Smoke"]

    # Convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y = lab.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # setup parameter space
    # parameters = {'criterion':['gini','entropy'],
    #             'max_depth':np.arange(1,21).tolist()[0::2],
    #             'min_samples_split':np.arange(2,11).tolist()[0::2],
    #             'max_leaf_nodes':np.arange(3,26).tolist()[0::2]}
    
    
    
    parameters = {'criterion': Categorical(['gini','entropy']),
              'max_depth': Integer(1,21,prior='log-uniform'),
              'min_samples_split': Real(1e-3,1.0,prior='log-uniform'),
              'max_leaf_nodes': Integer(3,26,prior='uniform')}
    

    # # create an instance of the grid search object
    # g1 = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_jobs=-1)
    # g1 = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=1000, random_state=42, n_jobs=-1)
    # g1 = BayesSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=10, random_state=42, n_jobs=-1)

    # conduct grid search over the parameter space
    start_time = time.time()
    # g1.fit(X_train, y_train)
    duration = time.time() - start_time

    # show best parameter configuration found for classifier
    # cls_params1 = g1.best_params_
    # print(cls_params1)

    # Define the model
    model = KNeighborsClassifier()
    # model = KNeighborsClassifier(n_neighbors=100)
    # model = GaussianNB()
    # model = DecisionTreeClassifier()
    # model = g1.best_estimator_

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    print(model.weights)

    # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    print('Accuracy score:', accuracy_score(y_test,y_pred))
    print('Precision score:', precision_score(y_test,y_pred, average=None))
    print('Recall score:', recall_score(y_test,y_pred, average=None))
    print('F1 score:', f1_score(y_test,y_pred, average=None))
    print('Computation time:', duration)

    # Confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)

    # Correlation matrix
    corr = merged_df_encoded.corr()
    fig, ax = plt.subplots(figsize=(40, 30))
    sns.heatmap(corr, annot=True, annot_kws={"size": 8}, linewidths=.5, ax=ax)
    plt.savefig("./data/processed/correlation_matrix.svg")
    # plt.show()

if __name__ == "__main__":
    main()
