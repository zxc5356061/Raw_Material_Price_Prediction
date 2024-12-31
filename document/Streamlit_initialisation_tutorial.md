# Initialisation Tutorial for first-time user to develop the web application

## 1. Clone repository from bitbucket

1. bitbucket > setting > Personal settings > App passwords > Create app password (to create personal access token)
2. Set up git accounts
    - $ git config --global user.name "John Doe"
    - $ git config --global user.email <johndoe@example.com>
3. Clone repository from bitbucket

## 2. Install dependencies

### 2.1. Virtual Environment

1. Install homebrew
2. Install pyenv to create virtual machine
    - $ brew install pyenv
    - $ brew install pyenv-virtualenv
3. To install Python version 3.10.0 and activate
    - $ pyenv virtualenv 3.10.0 <name>
    - $ pyenv local <name>
4. Check Python interpreter version
    1. Pycharm > Python Interpreter > Add new Interpreter > Add local interpreter > Virtualenv Environment > Location: ~
       /data_team_support_hub/.venv
    2. Check interpreter name as : "Python 3.10.0 <name>"

### 2.2. Install requirements

1. To install requirements from requirements.txt
    - pip install -r /path_to/src/common_packages/requirements.txt
    - pip install -r /path_to/src/data_team_support_hub/requirements.txt

### 2.3. Install and set up AWS CLI

1. [Install or update the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. [Set up the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)
    - Long-term credentials

    ```bash
    $ aws configure
    AWS Access Key ID [None]: xxx
    AWS Secret Access Key [None]: xxx
    Default region name [None]: eu-central-1
    Default output format [None]: json  
    ```

## 3. Run web application locally

```bash
invoke streamlit
```