# Initialisation Tutorial for first-time user

## 1. Clone repository

1. Set up git accounts
    - $ git config --global user.name "John Doe"
    - $ git config --global user.email <johndoe@example.com>
2. Clone repository from github

## 2. Install dependencies

### 2.1. Virtual Environment

1. Install homebrew
2. Install pyenv to create virtual machine
    - $ brew install pyenv
    - $ brew install pyenv-virtualenv
3. To install Python version 3.10.12 and activate
    - $ pyenv install 3.10.12
    - $ pyenv virtualenv 3.10.12 <given_name>
    - $ pyenv local <given_name>
4. Check Python interpreter version
    1. Pycharm > Python Interpreter > Add new Interpreter > Add local interpreter > Virtualenv Environment > Location: ~
       /Raw_Material_Price_Prediction/.venv
    2. Check interpreter name as : "Python 3.10 given_name"

### 2.2. Install requirements

1. To install requirements from requirements.txt
    - pip install -r /path_to/requirements.txt