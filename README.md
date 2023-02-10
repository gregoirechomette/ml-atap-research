# Machine learning models for assessment of local asteroid risks


## Installation of code and required libraries
Clone the git repo from the command line:
```
git clone https://github.com/gregoirechomette/ml-atap-research
```

Install the libraries and dependencies (requires python3 and pip3):
```
pip3 install virtualenv
python3 -m virtualenv .venv
cd .venv
source bin/activate
bin/pip install -r ../requirements.txt
```

## Use the pyton notebooks

The notebooks ```regression.ipynb``` and ```classification.ipynb``` are use to train and test several machine learning models, and to generate sensitivity analyses.