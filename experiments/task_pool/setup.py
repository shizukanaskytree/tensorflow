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

def proto_compile(output_path):
    import grpc_tools.protoc

    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=atom/proto",
        f"--python_out={output_path}",
        f"--grpc_python_out={output_path}",
    ] + glob.glob("atom/proto/*.proto")

    code = grpc_tools.protoc.main(cli_args)
    if code:  # hint: if you get this error in jupyter, run in console for richer error message
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")
    # Make pb2 imports in generated scripts relative
    for script in glob.iglob(f"{output_path}/*.py"):
        with open(script, "r+") as file:
            code = file.read()
            file.seek(0)
            file.write(re.sub(r"\n(import .+_pb2.*)", "from . \\1", code))
            file.truncate()


class Develop(develop):
    def run(self):
        super().run()
        # proto_compile(os.path.join("atom", "proto"))

setup(
    name='atom', # grpc timed storage 
    version='0.0.1', 
    packages=find_packages(),
    cmdclass={"develop": Develop}
)
