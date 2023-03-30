import os
from fastapi_app.tools.crypt import Crypt

SALT = 'gAAAAABj-OFE1NLL7IAQN2aXe_9_48MgF-pJNrM1NlrN6Sz6z-z4MO6QLI3WOICcQebVCNVCKCeM7V-1Q9AL7bfqYe3oAMM_Z5Ec3f_JBTuAgwvFNASUCC8='
MAIL_ADRESS = 'gAAAAABj-OFEj3BrVQRUALZjULbt6l4_PxGcCqBXkPZKAUl_MtJopCQLBEX2JnZaeQY61eRv3NS3QQ3WyU36UkNsISJxbISN2APRw-gmN2X_eb4dbHL-Qvk='
MAIL_PW = 'gAAAAABj-OFEfgAV5WlcOPz8pzZ4s5GR-atSI0WMLzXZ4uOiGoRP-5ERMgIoixBren_PT6gLyXAU3EZUahn7AMgKpEbbGN2h5LyfCNZNNwTzySBOjCBlBsg='
KEY_FOR_TOKEN = 'gAAAAABj-OFE8BgJLmeBSyL-5d_Ly3IxxdcgmJmXaUqIfv-D-toDOsjpA4JVE5TSfyXXFeQjYK6uLhanopgiLunL02Ug_zJ9Xs-DSHgcdoh4FIj0AQD7Y10='


if os.environ.get('PW') is not None:
    db_host = "mysql"  # use name of service in docker-compose.yml
    db_name = "peoplesun_user_db"
    db_user_name = "root"
    with open(os.environ['PW']) as file:
        PW = file.read()
        crypt = Crypt(PW)
        SALT = crypt.decrypt(SALT)
        MAIL_ADRESS = crypt.decrypt(MAIL_ADRESS).replace(' ', '')
        MAIL_PW = crypt.decrypt(MAIL_PW)
        KEY_FOR_TOKEN = crypt.decrypt(KEY_FOR_TOKEN)
        del os.environ['PW']
        del crypt
else:
    print('WARNING: no password provided')
    from fastapi_app.db.dev_config import db_host, db_name, db_user_name, PW, MAIL_PW, SALT, ACCESS_TOKEN_EXPIRE_MINUTES, \
        KEY_FOR_TOKEN, TOKEN_ALG, MAIL_ADRESS
db_port = 3306
ACCESS_TOKEN_EXPIRE_MINUTES=180
ACCESS_TOKEN_EXPIRE_MINUTES_EXTENDED=60*24*2
ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS=180
DOMAIN='https://peoplesun.energietechnik.tu-berlin.de'
MAIL_HOST='mail.gmx.net'
MAIL_PORT=587
TOKEN_ALG = 'HS256'
directory_parent = "fastapi_app"
directory_database = os.path.join(directory_parent, "data", "database").replace("\\", "/")
full_path_demands = os.path.join(directory_database, "demands.csv").replace("\\", "/")
directory_inputs = os.path.join(directory_parent, "data", "inputs").replace("\\", "/")
full_path_timeseries = os.path.join(directory_inputs, "timeseries.csv").replace("\\", "/")