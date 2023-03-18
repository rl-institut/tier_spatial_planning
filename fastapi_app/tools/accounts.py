import re
import uuid
import logging
import logging.handlers
from importlib import reload
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker, scoped_session
from fastapi_app.db.models import User
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from fastapi_app.db import config
from fastapi.security.utils import get_authorization_scheme_param
from fastapi_app.db.database import get_async_session_maker
from fastapi_app.db import queries, inserts


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Hasher:
    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)


async def is_valid_credentials(user):
    if not is_valid_email(user):
        return False, 'Please enter a valid email address'
    if not await is_mail_unregistered(user):
        return False, "Mail already registered"
    if not is_valid_password(user):
        return False, 'The password needs to be at least 8 characters long'
    return True, 'Please click the activation link we sent to your email'


def is_valid_email(user):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.match(regex, user.email)


async def is_mail_unregistered(user):
    if await queries.get_user_by_username(user.email) is None:
        return True
    else:
        return False


def is_valid_password(user):
    if len(user.password) < 8:
        return False
    else:
        return True


def create_guid():
    guid = str(uuid.uuid4()).replace('-', '')[0:12]
    return guid


def send_activation_link(mail, guid):
    url = '{}/activation_mail/guid={}'.format(config.DOMAIN, guid)
    msg = f"A PeopleSun account was created with this email.\nIf you want to activate the account follow the link:\n\n" \
          f"{url}\n\nOtherwise ignore this message."
    send_mail(mail, msg)


def send_mail(to_adress, msg):
    from fastapi_app.db.config import MAIL_PW
    logging.shutdown()
    reload(logging)
    logging.basicConfig(level=logging.CRITICAL,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d %b %Y %H:%M:%S')
    mail_logger = logging.getLogger("sendmail")
    mail_logger.propagate = False
    smtp_handler = logging.handlers.SMTPHandler(mailhost=(config.MAIL_HOST, config.MAIL_PORT),
                                                fromaddr=config.MAIL_ADRESS,
                                                toaddrs=[to_adress],
                                                subject='Activate your PeopleSun-Account',
                                                credentials=(config.MAIL_ADRESS, MAIL_PW),
                                                timeout=2.0,
                                                secure=())
    mail_logger.addHandler(smtp_handler)
    mail_logger.critical(msg)


async def activate_mail(guid):
    user = await queries.get_user_by_guid(guid)
    if user is not None:
        user.is_confirmed = True
        user.guid = ''
        await inserts.merge_model(user)
        send_email_with_activation_status(user)


def send_email_with_activation_status(user):
    if user is not None:
        if user.is_confirmed:
            msg = 'Your PeopleSun-Account ist activated'
        else:
            msg = 'Something went wrong with your PeopleSuN account activation'
        send_mail(user.email, msg)


async def authenticate_user(username: str, password: str):
    user = await queries.get_user_by_username(username)
    if user is None or Hasher.verify_password(password, user.hashed_password) is False:
        del password
        return False
    del password
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.KEY_FOR_TOKEN, algorithm=config.TOKEN_ALG)
    return encoded_jwt


async def _get_user_from_token(token):
    try:
        payload = jwt.decode(token, config.KEY_FOR_TOKEN, algorithms=[config.TOKEN_ALG])
        username = payload.get("sub")
    except JWTError:
        return None
    # if isinstance(db, scoped_session):
    query = select(User).where(User.email == username)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        user = res.scalars().first()
    return user


async def get_user_from_cookie(request):
    token = request.cookies.get("access_token")
    scheme, param = get_authorization_scheme_param(token)
    user = await _get_user_from_token(token=param)
    return user


if __name__ == '__main__':
    pass
