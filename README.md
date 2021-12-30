# Setup

- Tested with Python 3.8+ on MacOS / Ubuntu
- Recommended: create virtual environment for Python
    - `$ python3 -m venv ./venv`
    - `$ source venv/bin/activate`
- `$ pip install -r requirements.txt`

# Usage

Prior to the usage the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset must be decompressed and placed in the data folder.

The various versions of the applications (`main*.py`) can be startet with: `$ streamlit run main.py`.

The `occlusion.py` script can be used to generate visual explanations for the application to use.

The evaluation directory contains the data of the summative evaluation and the script to generate descriptive statistics and visualizations.