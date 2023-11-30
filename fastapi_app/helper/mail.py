import smtplib
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi_app import config


def send_mail(to_adress, msg, subject='Activate your PeopleSun-Account'):
    smtp_server = 'exchange.tu-berlin.de'
    smtp_port = 587
    smtp_username = config.MAIL_ADDRESS
    smtp_password = config.MAIL_PW
    message = MIMEMultipart()
    message["From"] = config.HEADER_ADDRESS
    message["To"] = to_adress if '@' in to_adress else config.MAIL_ADDRESS_LOGGER
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
