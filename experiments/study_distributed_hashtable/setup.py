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

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

here = os.path.abspath(os.path.dirname(__file__))
# os.path.dirname: Return the directory name of pathname path. 具体是从 `python` 执行开始处数的 folder dir name.
# __file__: 'setup.py'; 
# * __file__ is the pathname of the file from which the module was loaded.
#
# os.path.dirname(__file__): ''; 
# os.path.abspath(os.path.dirname(__file__)): '/home/wxf/tf2/tensorflow/experiments/study_distributed_hashtable'

# https://stackoverflow.com/questions/6323860/sibling-package-imports
# packages = find_packages()
# print(packages)
# ['utils', 'dht', 'p2p', 'p2p.p2p_daemon_bindings']


P2PD_VERSION = "v0.3.6"
P2PD_CHECKSUM = "627d0c3b475a29331fdfd1667e828f6d"
LIBP2P_TAR_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/archive/refs/tags/{P2PD_VERSION}.tar.gz"
P2PD_BINARY_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/releases/download/{P2PD_VERSION}/p2pd"


class Develop(develop):
    def run(self):
        self.reinitialize_command("build_py", build_lib=here)
        self.run_command("build_py")
        super().run()


def build_p2p_daemon():
    result = subprocess.run("go version", capture_output=True, shell=True).stdout.decode("ascii", "replace")
    m = re.search(r"^go version go([\d.]+)", result)

    if m is None:
        raise FileNotFoundError("Could not find golang installation")
    version = parse_version(m.group(1))
    if version < parse_version("1.13"):
        raise EnvironmentError(f"Newer version of go required: must be >= 1.13, found {version}")

    with tempfile.TemporaryDirectory() as tempdir:
        dest = os.path.join(tempdir, "libp2p-daemon.tar.gz")
        # repo url: LIBP2P_TAR_URL, https://github.com/learning-at-home/go-libp2p-daemon
        # repo is a go project: Go: 99.9%, Makefile:0.1%
        urllib.request.urlretrieve(LIBP2P_TAR_URL, dest)

        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(tempdir)

        # p2pd 是最后编译的 binary 文件
        result = subprocess.run(
            f'go build -o {shlex.quote(os.path.join(here, "distributedhashtable", "cli", "p2pd"))}',
            cwd=os.path.join(tempdir, f"go-libp2p-daemon-{P2PD_VERSION[1:]}", "p2pd"),
            shell=True,
        )

        if result.returncode:
            raise RuntimeError(
                "Failed to build or install libp2p-daemon:" f" exited with status code: {result.returncode}"
            )


def proto_compile(output_path):
    import grpc_tools.protoc

    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=distributedhashtable/proto",
        f"--python_out={output_path}",
        f"--grpc_python_out={output_path}",
    ] + glob.glob("distributedhashtable/proto/*.proto")

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


def md5(fname, chunk_size=4096):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_p2p_daemon():
    install_path = os.path.join(here, "distributedhashtable", "cli")
    binary_path = os.path.join(install_path, "p2pd")
    if not os.path.exists(binary_path) or md5(binary_path) != P2PD_CHECKSUM:
        print("Downloading Peer to Peer Daemon")
        urllib.request.urlretrieve(P2PD_BINARY_URL, binary_path)
        os.chmod(binary_path, 0o777)
        if md5(binary_path) != P2PD_CHECKSUM:
            raise RuntimeError(f"Downloaded p2pd binary from {P2PD_BINARY_URL} does not match with md5 checksum")


class BuildPy(build_py):
    user_options = build_py.user_options + [("buildgo", None, "Builds p2pd from source")] # 增加一个编译的 option `buildgo`, 默认值是 `None`, option 的说明是 "Builds p2pd from source"

    def initialize_options(self):
        super().initialize_options()
        self.buildgo = False

    def run(self):
        if self.buildgo:
            build_p2p_daemon()
        else:
            download_p2p_daemon()

        super().run()
        # print('self.build_lib: ', self.build_lib)
        proto_compile(os.path.join("distributedhashtable", "proto"))

# 可以选择 `python setup.py build_py`, 或者 `python setup.py develop`
# 编译开启 `self.buildgo`, `def build_p2p_daemon` 的命令是 `python setup.py build_py --buildgo`
setup(
    name='distributedhashtable', 
    version='0.0.1', 
    packages=find_packages(),
    cmdclass={"build_py": BuildPy, "develop": Develop},
    package_data={"distributedhashtable": ["proto/*", "hivemind_cli/*"]},
    include_package_data=True
)

