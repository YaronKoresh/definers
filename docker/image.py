import os

os.system("apt-get update -y")
os.system("apt-get install -y git")
os.system('pip install --no-cache-dir --force-reinstall "definers @ git+https://github.com/YaronKoresh/definers.git"')

exec("""
from definers import start
start("image")
""")