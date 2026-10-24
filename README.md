# Tobacco smoking prediction model

## Overview

This project analyzes the factors associated with tobacco consumption among adolescent students and develops a machine-learning model to predict smoking behavior. The goal is exploratory and preventive: the results can help identify relevant patterns in youth tobacco use, but they should not be interpreted as causal effects or as a clinical screening tool.

## Report

### Research question

Which demographic, social, environmental and tobacco-related factors are associated with smoking among adolescents, and how accurately can smoking behavior be predicted from survey responses?

### Data

The analysis uses the WHO **Global Youth Tobacco Survey (GYTS)**. The combined dataset contains **18,344 students** from national surveys conducted in:

| Country | Survey year |
| --- | ---: |
| Portugal | 2013 |
| Poland | 2016 |
| Romania | 2017 |
| Italy | 2018 |

The target variable is the smoking indicator (`Smoke`). Predictors include age, gender, country, smoking among friends and parents, exposure to second-hand smoke, perceptions of smokers, school warnings and anti-tobacco messages. Categorical variables are one-hot encoded, ordinal variables are ordinal-encoded, and class weights are used to account for class imbalance.

### Analyses performed

1. **Data preparation**: import of the original GYTS files, harmonization of columns and answers, and creation of a single analysis dataset.
2. **Exploratory analysis**: distributions of smoking behavior and associations between the target and survey variables.
3. **Model comparison**: evaluation of several classifiers with stratified 10-fold cross-validation, including logistic regression, decision trees, SVM, LightGBM, Extra Trees and Random Forest.
4. **Model improvement**: tuning and blending of the strongest tree-based models, followed by threshold selection.
5. **Interpretability**: a SHAP-based explainer dashboard is generated to inspect global and individual predictions.

The strongest simple associations with the smoking indicator in the processed association table are smoking friends (**0.350**), exposure to smoking in public places (**0.274**), exposure in enclosed places (**0.259**), age (**0.237**) and perceived harmfulness of second-hand smoke (**0.243**). These are associations in survey data, not evidence that one variable causes another.

### Results

The final model is a **Voting Classifier** combining LightGBM and Extra Trees (with the selected probability threshold set to 0.50). On the held-out test set it achieved:

| Metric | Test score |
| --- | ---: |
| Accuracy | 0.8686 |
| AUC | 0.8585 |
| Recall | 0.5800 |
| Precision | 0.6040 |
| F1 score | 0.5918 |
| Matthews correlation coefficient | 0.5136 |

The headline accuracy is relatively high, but recall and F1 are more informative for the less frequent smoking class. The model therefore captures useful predictive signal while still missing a proportion of smokers; it should be used for analysis and education, not for automated decisions about individual students.

### Reproducibility and outputs

- The full report is available in [`report.pdf`](./report.pdf).
- The processed association matrix is available as [`association_table.csv`](./data/processed/association_table.csv), with a visual version in [`association_table.png`](./data/processed/association_table.png).
- The trained model and SHAP dashboard files are stored in [`data/models/`](./data/models/).
- The complete workflow is documented in [`data_analysis.ipynb`](./src/notebooks/data_analysis.ipynb) and [`predictive_model.ipynb`](./src/notebooks/predictive_model.ipynb).

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
  The models are saved in the `./data/models` folder in pickle format. The folder also contains the final model explainer, which allows you to view interactive dashboards without recalculating SHAP values on every run.

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
