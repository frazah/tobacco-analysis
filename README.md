# Project Title: Tobacco smoking prediction model

## Overview
Briefly describe what this project is about. Include the main objective and potential impact.

## Installation and Setup
Instructions on setting up the project environment:
1. Clone the repository: `git clone [repository link]`
2. Install dependencies: `pip install -r requirements.txt`

## Data

- **Raw Data**: Location "./data/raw"
The raw data includes 2 folders, GYTS (Global Youth Tobacco Survey) and GATS (Global Adult Tobacco Survey).
They include datasets of survey answers about tobacco consumption and habits aswell as demographic information.
GATS datasets are in SAS format, GYTS are in MDB format.
Each column represents the survey question, each question has an ID (we will select only the relevant ones).
Each row contains all answers of each survey subject.

Furthermore, there's another datasets for smoking habits in the UK that is already in csv format.

- **Processed Data**: 
GATS data can be used directly through panda libraries, while the GYTS ones have been converted in csv format through the Microsoft Access export wizard.


## Usage
How to run the project:
1. Step-by-step instructions.
2. Example commands.

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
