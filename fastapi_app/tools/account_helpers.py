import re
import uuid
import asyncio
import base64
import random
from typing import Tuple
from captcha.image import ImageCaptcha
from passlib.context import CryptContext
from fastapi_app.db import sa_tables
from fastapi_app import config
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from fastapi.security.utils import get_authorization_scheme_param
from fastapi_app.db import async_inserts, async_queries
from fastapi_app.tools.mails import send_mail


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


def is_valid_password(user_or_password):
    if hasattr(user_or_password, 'password') and len(user_or_password.password) <= 8:
        return False
    if isinstance(user_or_password, str) and len(user_or_password) < 8:
        return False
    else:
        return True


def create_guid():
    guid = str(uuid.uuid4()).replace('-', '')[:12]
    return guid


def send_activation_link(mail, guid):
    url = '{}/activation_mail?guid={}'.format(config.DOMAIN, guid)
    msg = f"A PeopleSun account was created with this email.\nIf you want to activate the account follow the link:\n\n"\
          f"{url}\n\nOtherwise ignore this message."
    send_mail(mail, msg)


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
            msg = 'Your PeopleSun-Account is activated.'
        else:
            msg = 'Something went wrong with your PeopleSuN account activation.'
        send_mail(user.email, msg)


async def authenticate_user(username: str, password: str):
    user = await queries.get_user_by_username(username)
    if user is None:
        del password
        return False, 'Incorrect username or password'
    if user.is_confirmed is False:
        del password
        return False, 'Email not confirmed'
    if user is None or Hasher.verify_password(password, user.hashed_password) is False:
        del password
        return False, 'Incorrect username or password'
    del password
    return True, user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.KEY_FOR_ACCESS_TOKEN, algorithm=config.TOKEN_ALG)
    return encoded_jwt


async def get_user_from_cookie(request):
    for i in range(2):
        token = request.cookies.get("access_token")
        scheme, param = get_authorization_scheme_param(token)
        user = await queries._get_user_from_token(token=param)
        if user is not None:
            return user

async def generate_captcha_image() -> Tuple[str, str]:
    captcha_text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=6))
    captcha = ImageCaptcha()
    loop = asyncio.get_running_loop()
    captcha_data = await loop.run_in_executor(None, captcha.generate, captcha_text)
    base64_image = base64.b64encode(captcha_data.getvalue()).decode('utf-8')
    return captcha_text, base64_image

async def create_default_user_account():
    if await queries.get_user_by_username('default_example') is None:
        user = models.User(email='default_example',
                           hashed_password=Hasher.get_password_hash(config.EXAMPLE_USER_PW),
                           guid='',
                           is_confirmed=True,
                           is_active=False,
                           is_superuser=False)
        await inserts.merge_model(user)
        user = models.User(email='admin',
                           hashed_password=Hasher.get_password_hash(config.PW),
                           guid='',
                           is_confirmed=True,
                           is_active=False,
                           is_superuser=True)
        await inserts.merge_model(user)


if __name__ == '__main__':
    pass
