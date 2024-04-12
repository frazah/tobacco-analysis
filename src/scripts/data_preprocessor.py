import os
import pandas as pd

import mapping_answers_dict as map_dict


# Function to read dataframes from a folder
def read_dataframes(folder_path):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {file: pd.read_csv(os.path.join(folder_path, file)) for file in file_list}
    return dataframes


# Function to get common columns from dataframes
def get_common_columns(dataframes):
    common_columns = set(dataframes[next(iter(dataframes))].columns)
    for df in dataframes.values():
        common_columns &= set(df.columns)
    return list(common_columns)


# Function to preprocess dataframes
def preprocess_dataframes(dataframes, common_columns):
    merged_df = pd.concat([df[common_columns].assign(State=file.split(" ")[2]) for file, df in dataframes.items()])
    return merged_df


# Read dataframes from the folder
dataframes = read_dataframes("../../data/raw/GYTS/")

# Find common columns
common_columns = get_common_columns(dataframes)

# Preprocess dataframes
dataset = preprocess_dataframes(dataframes, common_columns)


# Rename columns and map values
dataset.rename(columns=map_dict.column_mappings, inplace=True)
for column, mapping in map_dict.column_mappings.items():
    if mapping in dataset.columns:
        dataset[mapping] = dataset[mapping].map(eval(f"map_dict.{column}_dict"))

# Keep only the desired columns
columns_to_keep = list(map_dict.column_mappings.values())
columns_to_keep.append("State")
dataset = dataset[columns_to_keep]

# Create additional columns
dataset['SmokingFather'] = dataset['SmokingParents'].isin(['Both', 'Father only'])
dataset['SmokingMother'] = dataset['SmokingParents'].isin(['Both', 'Mother only'])
dataset['WorkingFather'] = dataset['WorkingParents'].isin(['Both', 'Father only'])
dataset['WorkingMother'] = dataset['WorkingParents'].isin(['Both', 'Mother only'])

# Drop unnecessary columns
dataset.drop(columns=['SmokingParents', 'WorkingParents'], inplace=True)

# Convert columns to categorical
categorical_columns = ["State", "Gender", "Age", "SmokingFriends", "SeenSmokerInPublicPlace",
                       "SeenSmokerInEnclosedPlace", "SeenSmokerInHome", "AttractiveSmoker",
                       "HardQuitSmoke", "SmokerConfidentInCelebrations", "SchoolWarnings",
                       "SeenHealthWarnings", "AntiTobaccoInEvents", "HarmfulPassiveSmoke"]
dataset[categorical_columns] = dataset[categorical_columns].astype('category')

# Convert to boolean
boolean_columns = ["Smoke", "SeenSmokerInSchool", "ParentWarnings", "AntiTobaccoInMedia",
                   "BanTobaccoOutdoors", "SmokingFather", "SmokingMother", "WorkingFather",
                   "WorkingMother"]
dataset[boolean_columns] = dataset[boolean_columns].astype('bool')

# drop rows with missing values
dataset.dropna(inplace=True)

# Save the preprocessed dataframe to a CSV file
dataset.to_csv("../../data/processed/GYTS_dataset.csv", index=False)
