# Description:
# Package for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

load("//tensorflow/python/tools/api/generator:api_gen.bzl", "gen_api_init_files")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "KERAS_API_INIT_FILES")
load("//tensorflow/python/tools/api/generator:api_init_files_v1.bzl", "KERAS_API_INIT_FILES_V1")

gen_api_init_files(
    name = "keras_python_api_gen",
    api_name = "keras",
    api_version = 1,
    output_files = KERAS_API_INIT_FILES_V1,
    output_package = "tensorflow.python.keras.api",
    package_deps = [
        "//tensorflow/python/keras",
        "//tensorflow/python:no_contrib",
    ],
    packages = [
        "tensorflow.python",
        "tensorflow.python.keras",
    ],
)
# 1.
# gen_api_init_files 函数说明:
# tensorflow/python/tools/api/generator/api_gen.bzl
# def gen_api_init_files(
#         name,
#         output_files = TENSORFLOW_API_INIT_FILES,
#         root_init_template = None,
#         srcs = [],
#         api_name = "tensorflow",
#         api_version = 2,
#         compat_api_versions = [],
#         compat_init_templates = [],
#         packages = ["tensorflow.python", "tensorflow.lite.python.lite"],
#         package_deps = ["//tensorflow/python:no_contrib"],
#         output_package = "tensorflow",
#         output_dir = "",
#         root_file_name = "__init__.py"):
#     """Creates API directory structure and __init__.py files.
#
#     Creates a genrule that generates a directory structure with __init__.py
#     files that import all exported modules (i.e. modules with tf_export
#     decorators).
#
#     Args:
#       name: name of genrule to create.
#       output_files: List of __init__.py files that should be generated.
#         This list should include file name for every module exported using
#         tf_export. For e.g. if an op is decorated with
#         @tf_export('module1.module2', 'module3'). Then, output_files should
#         include module1/module2/__init__.py and module3/__init__.py.
#       root_init_template: Python init file that should be used as template for
#         root __init__.py file. "# API IMPORTS PLACEHOLDER" comment inside this
#         template will be replaced with root imports collected by this genrule.
#       srcs: genrule sources. If passing root_init_template, the template file
#         must be included in sources.
#       api_name: Name of the project that you want to generate API files for
#         (e.g. "tensorflow" or "estimator").
#       api_version: TensorFlow API version to generate. Must be either 1 or 2.
#       compat_api_versions: Older TensorFlow API versions to generate under
#         compat/ directory.
#       compat_init_templates: Python init file that should be used as template
#         for top level __init__.py files under compat/vN directories.
#         "# API IMPORTS PLACEHOLDER" comment inside this
#         template will be replaced with root imports collected by this genrule.
#       packages: Python packages containing the @tf_export decorators you want to
#         process
#       package_deps: Python library target containing your packages.
#       output_package: Package where generated API will be added to.
#       output_dir: Subdirectory to output API to.
#         If non-empty, must end with '/'.
#       root_file_name: Name of the root file with all the root imports.
#     """



gen_api_init_files(
    name = "keras_python_api_gen_compat_v1",
    api_name = "keras",
    api_version = 1,
    output_dir = "_v1/",
    output_files = KERAS_API_INIT_FILES_V1,
    output_package = "tensorflow.python.keras.api._v1",
    package_deps = [
        "//tensorflow/python/keras",
        "//tensorflow/python:no_contrib",
    ],
    packages = [
        "tensorflow.python",
        "tensorflow.python.keras",
    ],
)

gen_api_init_files(
    name = "keras_python_api_gen_compat_v2",
    api_name = "keras",
    api_version = 2,
    output_dir = "_v2/",
    output_files = KERAS_API_INIT_FILES,
    output_package = "tensorflow.python.keras.api._v2",
    package_deps = [
        "//tensorflow/python/keras",
        "//tensorflow/python:no_contrib",
    ],
    packages = [
        "tensorflow.python",
        "tensorflow.python.keras",
    ],
)
