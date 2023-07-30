import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi_app.io.db import config


def send_mail(to_adress, msg, subject='Activate your PeopleSun-Account'):
    from fastapi_app.io.db.config import MAIL_PW
    smtp_server = config.MAIL_HOST
    smtp_port = config.MAIL_PORT
    smtp_username = config.MAIL_ADDRESS
    smtp_password = MAIL_PW
    message = MIMEMultipart()
    message["From"] = config.MAIL_ADDRESS
    message["To"] = to_adress
    message["Subject"] = subject
    message.attach(MIMEText(msg, "plain"))
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        try:
            server.login(smtp_username, smtp_password)
            server.sendmail(config.MAIL_ADDRESS, to_adress, message.as_string())
        except smtplib.SMTPAuthenticationError as e:
            raise Exception(config.MAIL_ADDRESS.replace('@', ''))