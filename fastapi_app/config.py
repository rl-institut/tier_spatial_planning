import os
from pathlib import Path
from dotenv import load_dotenv

APP_DIR = "fastapi_app"
DIRECTORY_WP3 = os.path.join(APP_DIR, "data", "demand").replace("\\", "/")
FULL_PATH_PROFILES = os.path.join(DIRECTORY_WP3, "1-hour_mean_365_days_all_users.parquet").replace("\\", "/")
FULL_PATH_DISTRIBUTIONS = os.path.join(DIRECTORY_WP3, "zonal_consumption_distributions.parquet").replace("\\", "/")

mail_vars = ['MAIL_ADDRESS',
             'HEADER_ADDRESS',
             'MAIL_HOST',
             'MAIL_PORT',
             'MAIL_PW']

var_list = ['PW',
            'KEY_FOR_ACCESS_TOKEN',
            'EXAMPLE_USER_PW']

if os.environ.get('DOCKERIZED') is None or bool(os.environ.get('DOCKERIZED')) is False:
    # If the environment variable 'DOCKERIZED' is not set, it is assumed that the app is running outside a docker container.
    # In this case, we need to read the environment files and docker secrets files manually.
    load_dotenv(dotenv_path='fastapi_app.env')
    load_dotenv(dotenv_path='mail.env')


root_path = os.getcwd()
secret_path = os.path.join(root_path, 'secrets')
if not os.path.exists(secret_path):
    root_path = next((path for path in Path.cwd().parents if path.name == APP_DIR), None)
    secret_path = os.path.join(root_path, 'secrets') if root_path is not None else None
if not os.path.exists(secret_path):
    raise FileNotFoundError(f"Could not find directory the directory /secrets/")

file_paths = {'PW': os.path.join(secret_path, 'secret.txt'),
              'MAIL_PW': os.path.join(secret_path, 'mail_secret.txt'),
              'KEY_FOR_ACCESS_TOKEN': os.path.join(secret_path, 'key_for_token.txt'),
              'EXAMPLE_USER_PW': os.path.join(secret_path, 'example_user_secret.txt'),
              'CDS_API_KEY': os.path.join(secret_path, 'cds_api_key.txt'),
              }

for var_name, file_path in file_paths.items():
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            os.environ[var_name] = file.read().strip()

def is_set(var_name):
    if os.environ.get(var_name) == 'tbd' or os.environ.get(var_name) is None:
        return False
    else:
        return True

for var in mail_vars:
    if is_set(var) is False:
        print("To set up the mail service, you need to define the environment variable \"{}\".\nWithout the mail"
              " service, the web app won't be able to send activation links during user registration.\n".format(var))

if is_set('PW') is False:
    print("To set up the database, you need to define the environment variable PW")

if is_set('KEY_FOR_ACCESS_TOKEN') is False:
    print("To set up the user access token, you need to define the environment variable KEY_FOR_ACCESS_TOKEN")

if is_set('EXAMPLE_USER_PW') is False:
    print("To set up the default_example user, you need to define the environment variable EXAMPLE_USER_PW")


DB_RETRY_COUNT = int(os.environ.get('DB_RETRY_COUNT'))
RETRY_DELAY = float(os.environ.get('DB_RETRY_DELAY'))
TOKEN_ALG = os.environ.get('TOKEN_ALG')
KEY_FOR_ACCESS_TOKEN = os.environ.get('KEY_FOR_ACCESS_TOKEN')
ACCESS_TOKEN_EXPIRE_MINUTES = int(float(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES')))
ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS = int(float(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS')))
ACCESS_TOKEN_EXPIRE_MINUTES_EXTENDED = int(float(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES_EXTENDED')))
DB_USER_NAME=os.environ.get('DB_USER_NAME')
PW=os.environ.get('PW')
DB_HOST=os.environ.get('DB_HOST')
DB_PORT=os.environ.get('DB_PORT')
DB_NAME=os.environ.get('DB_NAME')
DOMAIN=os.environ.get('DOMAIN')
EXAMPLE_USER_PW=os.environ.get('EXAMPLE_USER_PW')
MAIL_PW=os.environ.get('MAIL_PW')
MAIL_ADDRESS=os.environ.get('MAIL_ADDRESS')
MAIL_ADDRESS_LOGGER=os.environ.get('MAIL_ADDRESS_LOGGER')
HEADER_ADDRESS=os.environ.get('HEADER_ADDRESS')
CDS_API_KEY=os.environ.get('CDS_API_KEY')
DOCKERIZED=os.environ.get('DOCKERIZED')