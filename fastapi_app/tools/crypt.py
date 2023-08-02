import os
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet


class Crypt:

    def __init__(self, pw=None, salt=None):
        pw = bytes(pw, "utf-8") if isinstance(pw, str) else pw
        salt = bytes(salt, "utf-8") if isinstance(salt, str) else salt if salt else bytes("fix", "utf-8")
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(),
                            length=32,
                            salt=salt,
                            iterations=100000,
                            backend=default_backend())
        self.fernet = Fernet(base64.urlsafe_b64encode(kdf.derive(pw)))
        del pw, salt


    @staticmethod
    def prepare_str(string_or_byte):
        if isinstance(string_or_byte, str):
            return bytes(string_or_byte, "utf-8")
        elif isinstance(string_or_byte, bytes):
            return string_or_byte


    def encrypt(self, string_or_byte):
        bstring = Crypt.prepare_str( string_or_byte)
        if bstring:
            return self.fernet.encrypt(bstring).decode()


    def decrypt(self, string_or_byte):
        bstring = Crypt.prepare_str( string_or_byte)
        if bstring:
            return self.fernet.decrypt(bstring).decode()


if __name__ == '__main__':
    pw = ''
    s = Crypt(pw).encrypt('encrypt this string')
    print(s)
