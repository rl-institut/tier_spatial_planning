import smtplib
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi_app.python import config

"""
This module is utilized primarily during user registration in the application to send activation links. It is also used 
for sending password reset emails or notifications for changes in user email. Additionally, the module is employed to 
dispatch emails upon the completion of project calculations, provided the user has enabled email notifications. 
Moreover, it serves a critical function in error logging by sending relevant email alerts.
"""

def send_mail(to_address, msg, subject='Activate your PeopleSun-Account'):
    smtp_server = config.MAIL_HOST
    smtp_port = config.MAIL_PORT
    smtp_username = config.MAIL_ADDRESS
    smtp_password = config.MAIL_PW
    message = MIMEMultipart()
    message["From"] = config.HEADER_ADDRESS
    message["To"] = to_address if '@' in to_address else config.MAIL_ADDRESS_LOGGER
    message["Subject"] = subject
    message.attach(MIMEText(msg, "plain"))
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        try:
            server.login(smtp_username, smtp_password)
            server.sendmail(config.MAIL_ADDRESS, message["To"], message.as_string())
        except smtplib.SMTPAuthenticationError as e:
            print('\n{}\n{}'.format(e, config.MAIL_ADDRESS.replace('@', '')))
            warnings.warn(str(e), category=UserWarning)
