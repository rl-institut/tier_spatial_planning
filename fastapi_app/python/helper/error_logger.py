import logging
import os
import socket
import traceback

from fastapi_app.python import config
from fastapi_app.python.helper.mail import send_mail

"""
This module implements a custom logger that records application errors to a log file and also emails these error 
details, including user and request information, for enhanced monitoring and debugging.
"""

directory = os.getcwd() + '/logs'

if not os.path.exists(directory):
    os.makedirs(directory)


class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def error_log(self, exc, request, user_name):
        user_name = user_name if isinstance(user_name, str) else ''
        host = socket.gethostname()
        try:
            msg = type(exc).__name__ if hasattr(exc, '__class__') else ''
        except Exception:
            msg = ''
        msg += ': ' + str(exc) if hasattr(exc, '__str__') else ''
        msg += '; ' + user_name
        msg += '; ' + str(request.url) if hasattr(request, 'url') else ''
        try:
            msg += '; ' + str(request.scope) if hasattr(request, 'scope') else ''
        except Exception:
            msg += ''
        msg += '\n\n'
        try:
            if hasattr(exc, '__str__') and isinstance(exc, Exception):
                msg += traceback.format_exc()
        except Exception:
            pass
        msg += '\n\n'
        try:
            send_mail(subject='Error - {} - {}'.format(user_name, host), msg=msg, to_address=config.MAIL_ADDRESS_LOGGER)
        except Exception:
            pass
        self.error(msg)


logger = CustomLogger('error_logger')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(directory + '/error_logs.txt')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
