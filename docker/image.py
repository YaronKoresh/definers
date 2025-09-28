import os

os.system("apt update -y")
os.system("apt install -y git")
os.system('pip install --no-cache-dir --force-reinstall "definers @ git+https://github.com/YaronKoresh/definers.git"')

exec("""
from definers import start
start("image")
""")