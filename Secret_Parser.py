import json

def get_Secrets():
    with open('Secrets.json') as File:
        secrets = json.load(File)
    return secrets