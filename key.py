import secrets

SECRET_KEY = secrets.token_hex(32)
JWT_SECRET = secrets.token_hex(32)

print("SECRET_KEY =", SECRET_KEY)
print("JWT_SECRET =", JWT_SECRET)