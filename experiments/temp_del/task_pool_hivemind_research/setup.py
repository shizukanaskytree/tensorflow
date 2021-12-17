import codecs
import glob
import hashlib
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import urllib.request

from pkg_resources import parse_requirements, parse_version
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='atom', # task pool research
    version='0.0.1', 
    packages=find_packages(),
)
