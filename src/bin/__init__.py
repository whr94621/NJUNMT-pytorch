import os

def auto_mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)