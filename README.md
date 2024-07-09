# Tobacco smoking prediction model

## Overview

This study aims to analyze the factors associated with tobacco consumption among adolescent students, in order to better understand this phenomenon and inform prevention policies. We will use data analysis techniques and machine learning to identify the factors associated with tobacco consumption among adolescent students

## Installation and Setup

Instructions on setting up the project environment:

1. Clone the repository: `git clone https://gitlab.com/frazah/tobacco-analysis`
2. Install dependencies: `pip install -r requirements.txt`

## Data

- **Raw Data**: Location `./data/raw`
The raw data includes the GYTS (Global Youth Tobacco Survey) dataset.
It includes survey answers about tobacco consumption and habits aswell as demographic information.
Each column represents the survey question, each question has an ID (we will select only the relevant ones).
Each row contains all answers of each survey subject.

- **Processed Data**: GYTS data has been converted in csv format through the Microsoft Access export wizard.
Then using pandas (in the `data_preprocessor.py` script) it has been exported in a new file with readable columns and values (`GYTS_dataset.csv`)

- **Models**:
  The models are saved in the `./models` folder. The models are saved in the pickle format. In addition to the model, there is the final model explainer in the folder, which allows you to view interactive dashboards without having to recalculate SHAP values each run.

## Notebooks

The notebooks are located in the `./src/notebooks` folder. The notebooks are used to preprocess the data, train the models, and evaluate the models. The notebooks are named according to the task they perform:

- `data_analysis.ipynb`: This notebook is used to analyze the data and identify the features that are most important for the model.
- `predictive_model.ipynb`: This notebook is used to train the model and evaluate its performance.

## Scripts

The scripts are located in the `./src/scripts` folder. The scripts are used to preprocess the data and for utility functions. The scripts are named according to the task they perform:

- `data_preprocessor.py`: This script is used to preprocess the data and save the processed data to a new file in the `./data/processed` folder.
- `mapping_answers_dict.py`: This script is used to map the answers to the questions in the dataset to a dictionary.
- `custom_functions.py`: This script contains customize pycaret functions.

## Usage
How to run the project:

- Step-by-step instructions:
       1. Select python 3.11.4 kernel in current IDE
       2. Run the script `src/scripts/data_preprocessor.py`.
       3. Run the script `src/notebooks/data_analysis.ipynb`.
       4. Run the script `src/notebooks/predictive_model.ipynb`.

## Structure

- `/data`: Contains raw and processed data.
- `/src`: Source code for the project.
  - `/scripts`: Individual scripts or modules.
  - `/notebooks`: Jupyter notebooks or similar.
- `/tests`: Test cases for your application.
- `/docs`: Additional documentation in text format (e.g., LaTeX or Word).
- `/public`: Folder where GitLab pages will write static website. 
- `index.html`: Documentation in rich format (e.g., HTML, Markdown, JavaScript), will populate `public`.

## Contact

Students:
  - [Leonardo Dessì](mailto:leonardo.dessi@studio.unibo.it)
  - [Francesco Corigliano](mailto:francesco.coriglian2@studio.unibo.it)
