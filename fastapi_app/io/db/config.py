import os
from fastapi_app.tools.crypt import Crypt

SALT = 'gAAAAABj-OFE1NLL7IAQN2aXe_9_48MgF-pJNrM1NlrN6Sz6z-z4MO6QLI3WOICcQebVCNVCKCeM7V-1Q9AL7bfqYe3oAMM_Z5Ec3f_JBTuAgwvFNASUCC8='
MAIL_ADDRESS = 'gAAAAABkyNDcV-TRqpB0Gg2muwFy-qcsg5vQbYnWOZOoPROaZp8mTzn0u-P77adRk0de7-c-lLZM_amaupgn2_1Ae1P_dbzr8VjgH-gIE_w4nHvksGfRf3AZ8XAaqJNOeIH479jk8ZBo'
FROM_ADDRESS = 'gAAAAABkyNFU49gHsJLIGHudXmyRONPlxMPI9lUqdlV004xnl6KWdCcXBb01nIpniREXfjtdXC0wy14kXYKFWP1j55YJDVupd9jIaFkRjXSS7_ndoi8VL08='
MAIL_ADDRESS_LOGGER = 'gAAAAABkm-nPFnh8LDlwsgnpZehRqkSlieqZIuB2lp7xatcSvFlj4fp_4l5TkugFJ-D44uRffagTMmcSRw_u_M6m_y5jJw1Yjw_qQ-1TiM3Pc9gEAos9TeI='
MAIL_PW = 'gAAAAABkyNEp1bXDEior8pe_Fhshdezg7V6bfsD89kjmjtOa9pMUka97FtibagDP6RsmjOeencZiK9uMlQFGQtwgI03Ds2cuaKsqxClysVnjiB5zuc88BGY='
KEY_FOR_TOKEN = 'gAAAAABj-OFE8BgJLmeBSyL-5d_Ly3IxxdcgmJmXaUqIfv-D-toDOsjpA4JVE5TSfyXXFeQjYK6uLhanopgiLunL02Ug_zJ9Xs-DSHgcdoh4FIj0AQD7Y10='


if os.environ.get('PW') is not None:
    db_host = "mysql"  # use name of service in docker-compose.yml
    db_name = "peoplesun_user_db"
    db_user_name = "root"
    with open(os.environ['PW']) as file:
        PW = file.read()
        crypt = Crypt(PW)
        SALT = crypt.decrypt(SALT)
        MAIL_ADDRESS = crypt.decrypt(MAIL_ADDRESS).replace(' ', '')
        FROM_ADDRESS = crypt.decrypt(FROM_ADDRESS).replace(' ', '')
        MAIL_ADDRESS_LOGGER = crypt.decrypt(MAIL_ADDRESS_LOGGER)
        MAIL_PW = crypt.decrypt(MAIL_PW)
        KEY_FOR_TOKEN = crypt.decrypt(KEY_FOR_TOKEN)
        del os.environ['PW']
        del crypt
else:
    print('WARNING: no password provided')
    from fastapi_app.io.db.dev_config import db_host, db_name, db_user_name, PW, MAIL_PW, SALT, \
        ACCESS_TOKEN_EXPIRE_MINUTES, \
        KEY_FOR_TOKEN, TOKEN_ALG, MAIL_ADDRESS, MAIL_ADDRESS_LOGGER, FROM_ADDRESS
db_port = 3306
ACCESS_TOKEN_EXPIRE_MINUTES=180
ACCESS_TOKEN_EXPIRE_MINUTES_EXTENDED=60*24*2
ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS=180
DOMAIN='https://offgridplanner.org'
MAIL_HOST='mail.gmx.net'
MAIL_PORT=587
TOKEN_ALG = 'HS256'
directory_parent = "fastapi_app"
directory_database = os.path.join(directory_parent, "data", "database").replace("\\", "/")
full_path_demands = os.path.join(directory_database, "demands.csv").replace("\\", "/")
directory_inputs = os.path.join(directory_parent, "data", "inputs").replace("\\", "/")
full_path_timeseries = os.path.join(directory_inputs, "timeseries.csv").replace("\\", "/")
directory_wp3 = os.path.join(directory_parent, "data", "wp3").replace("\\", "/")
full_path_profiles = os.path.join(directory_wp3, "1-hour_mean_365_days_all_users.parquet").replace("\\", "/")
full_path_distributions = os.path.join(directory_wp3, "zonal_consumption_distributions.parquet").replace("\\", "/")
RETRY_COUNT = 10
RETRY_DELAY = 0.5