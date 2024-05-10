# Project Title: Tobacco smoking prediction model

## Overview
This study aims to analyze the factors associated with tobacco consumption among adolescent students, in order to better understand this phenomenon and inform prevention policies. We will use data analysis techniques and machine learning to identify the factors associated with tobacco consumption among adolescent students

## Installation and Setup
Instructions on setting up the project environment:
1. Clone the repository: `git clone [repository link]`
2. Install dependencies: `pip install -r requirements.txt`

## Data

- **Raw Data**: Location "./data/raw"
The raw data includes the GYTS (Global Youth Tobacco Survey) dataset.
It includes survey answers about tobacco consumption and habits aswell as demographic information.
Each column represents the survey question, each question has an ID (we will select only the relevant ones).
Each row contains all answers of each survey subject.


- **Processed Data**: 
GYTS data has been converted in csv format through the Microsoft Access export wizard.
Then using pandas (in the data_preprocessor.py script) it has been exported in a new file with readable columns and values (GYTS_dataset.csv)


## Usage
How to run the project:
1. Step-by-step instructions.
  -Select python 3.11.4 kernel in current IDE
  -Run the scripts in the src/notebooks folder
  -The explainer dashboard interface will open locally on the address provided

## Structure
- `/data`: Contains raw and processed data.
- `/src`: Source code for the project.
  - `/scripts`: Individual scripts or modules.
  - `/notebooks`: Jupyter notebooks or similar.
- `/tests`: Test cases for your application.
- `/docs`: Additional documentation in text format (e.g., LaTeX or Word).
- `/public`: Folder where GitLab pages will write static website. 
- `index.html`: Documentation in rich format (e.g., HTML, Markdown, JavaScript), will populate `public`.

## Contribution
Guidelines for contributing to the project (if applicable).

## License
State the license or leave it as default (if applicable).

## Contact
Your contact information for students or others to reach out with questions.
