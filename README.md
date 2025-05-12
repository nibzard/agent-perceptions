# O'Reilly Survey Data Preparation

This repository contains scripts and data for preparing and analyzing survey responses for the O'Reilly survey preprint project.

## Project Structure

- `scripts/01_prepare.py`: Cleans and processes the raw survey data, expands JSON answers, extracts region info, and outputs tidy data files.
- `data/raw/`: Place raw survey data files here (e.g., `survey_responses_rows_20250512.csv`).
- `data/`: Output directory for cleaned data and extracted text.

## Setup

1. **Install dependencies**

   This project requires Python 3.8+ and the following packages:
   - pandas

   You can install dependencies with:
   ```bash
   pip install pandas
   ```

2. **Prepare data**

   Place your raw survey CSV file in the `data/raw/` directory. By default, the script expects a file named `survey_responses_rows_20250512.csv`.

3. **Run the preparation script**

   ```bash
   python scripts/01_prepare.py
   ```

   This will generate:
   - `data/clean_survey.parquet`: Cleaned and expanded survey data.
   - `data/q11.txt`: Free-text responses to question 11.

## Notes
- Update the `RAW` variable in `scripts/01_prepare.py` if your raw data file has a different name or location.
- The script automatically creates the `data/` directory if it does not exist.

## Using Conda (Recommended)

1. **Create the environment**

   Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.

   ```bash
   conda env create -f environment.yml
   conda activate ai_survey
   ```

2. **Run the preparation script**

   ```bash
   python scripts/01_prepare.py
   ```

## Using Docker

1. **Build the Docker image**

   ```bash
   docker build -t oreilly-survey .
   ```

2. **Run the container**

   Mount your project directory to persist data:

   ```bash
   docker run -it --rm -v "$PWD":/workspace oreilly-survey
   ```

3. **Run the preparation script inside the container**

   ```bash
   python scripts/01_prepare.py
   ```

## License

- **Code:** MIT License (see LICENSE file)
- **Data:** CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International)