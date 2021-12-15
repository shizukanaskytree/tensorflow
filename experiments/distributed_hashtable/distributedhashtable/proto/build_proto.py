import glob
import re
import os


def proto_compile(output_path):
    import grpc_tools.protoc

    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=.",
        f"--python_out={output_path}",
        f"--grpc_python_out={output_path}",
    ] + glob.glob("./*.proto")

    code = grpc_tools.protoc.main(cli_args)
    if (
        code
    ):  # hint: if you get this error in jupyter, run in console for richer error message
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")

    # Make pb2 imports in generated scripts relative
    for script in glob.iglob(f"{output_path}/*.py"):
        with open(script, "r+") as file:
            code = file.read()
            file.seek(0)

            """
            what is /1 in regex -- Google.
            
            The backreference \1 (backslash one) references the first capturing group. 
            \1 matches the exact same text that was matched by the first capturing group. 
            The / before it is a literal character. It is simply the forward slash in 
            the closing HTML tag that we are trying to match.
            """
            file.write(re.sub(r"\n(import .+_pb2.*)", "from . \\1", code))
            file.truncate()


proto_compile(".")
