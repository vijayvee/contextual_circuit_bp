import os
from setuptools import setup, find_packages
from db import credentials
from subprocess import call
from utils import logger as log


setup(
    name="Contextual_DCN",
    version="0.1",
    packages=find_packages(),
)

params = credentials.postgresql_connection()
sys_password = credentials.machine_credentials()['password']
os.popen('sudo -u postgres createuser -sdlP %s' % params['user'], 'w').write(sys_password)
os.popen(
    'sudo -u postgres createdb %s -O %s' % (
        params['database'],
        params['user']),
        'w').write(sys_password)

log.info('Installed required packages and created DB.')

