import os
from pathlib import Path
from dotenv import load_dotenv

DIRECTORY_PARENT = "fastapi_app"
DIRECTORY_WP3 = os.path.join(DIRECTORY_PARENT, "data", "demand").replace("\\", "/")
FULL_PATH_PROFILES = os.path.join(DIRECTORY_WP3, "1-hour_mean_365_days_all_users.parquet").replace("\\", "/")
FULL_PATH_DISTRIBUTIONS = os.path.join(DIRECTORY_WP3, "zonal_consumption_distributions.parquet").replace("\\", "/")

def check_vars(var_name):
    if os.environ.get(var_name) == 'tbd' or os.environ.get(var_name) is None:
        user_input = input(f"{var_name} is not set. Please enter a value:")
        os.environ[f"{var_name}"] = f"value_{user_input}"

var_list = ['MAIL_ADDRESS',
            'HEADER_ADDRESS',
            'LOGGER_RECEIVING_MAIL',
            'MAIL_HOST',
            'MAIL_PORT',
            'PW',
            'MAIL_PW',
            'KEY_FOR_ACCESS_TOKEN']

if os.environ.get('DOCKERIZED') is None:
    load_dotenv(dotenv_path='fastapi_app.env')
    load_dotenv(dotenv_path='mail.env')
    secret_path = next((path for path in Path.cwd().parents if path.name == DIRECTORY_PARENT), None)
    secret_path = os.getcwd() if secret_path is None else str(secret_path.parent)
    if secret_path is None:
        raise FileNotFoundError(f"Could not find directory {DIRECTORY_PARENT}")

    file_paths = {'PW': os.path.join(os.getcwd(), 'secrets', 'secret.txt'),
                  'MAIL_PW': os.path.join(secret_path, 'secrets', 'mail_secret.txt'),
                  'KEY_FOR_ACCESS_TOKEN': os.path.join(secret_path, 'secrets', 'key_for_token.txt')}
    for var_name, file_path in file_paths.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                os.environ[var_name] = file.read().strip()

for var in var_list:
    check_vars(var)

DB_RETRY_COUNT = int(os.environ.get('DB_RETRY_COUNT'))
RETRY_DELAY = float(os.environ.get('DB_RETRY_DELAY'))
TOKEN_ALG = os.environ.get('TOKEN_ALG')
KEY_FOR_ACCESS_TOKEN = os.environ.get('KEY_FOR_ACCESS_TOKEN')
ACCESS_TOKEN_EXPIRE_MINUTES = int(float(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES')))
ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS = int(float(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS')))
DB_USER_NAME=os.environ.get('DB_USER_NAME')
PW=os.environ.get('PW')
DB_HOST=os.environ.get('DB_HOST')
DB_PORT=os.environ.get('DB_PORT')
DB_NAME=os.environ.get('DB_NAME')
DOMAIN=os.environ.get('DOMAIN')