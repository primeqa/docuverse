import sys
import os
import platform

from itertools import chain
from setuptools import setup, find_packages

if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. Please switch to 64-bit Python."
    )
    sys.exit(-1)


python_min_version = (3, 7, 0)
python_max_version = (3, 13, 0)

if python_min_version >= sys.version_info or python_max_version < sys.version_info:
    print(
        f"You are using Python {platform.python_version()}. {'.'.join(map(str, python_min_version))} >= Python < {'.'.join(map(str, python_max_version))} is required."
    )
    sys.exit(-1)

################################################################################
# constants
################################################################################
cwd = os.path.dirname(os.path.abspath(__file__))


################################################################################
# Version, description and package_name
################################################################################
package_name = os.getenv("PACKAGE_NAME", "docuverse")
package_type = os.getenv("PACKAGE_TYPE", "wheel")
with open(os.path.join(cwd, "VERSION"), "r", encoding="utf-8") as version_file:
    version = version_file.read().strip()
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as readme_file:
    long_description = readme_file.read()


################################################################################
# Build
################################################################################
print(f"Building wheel {package_name}-{version}")

include_packages = list(
    chain.from_iterable(map(lambda name: [name, f"{name}.*"], ["docuverse"]))
)

# _deps = {
#     "rouge-score~=0.1.2": ["install"],
#     "pyyaml~=6.0.1": ["install"],
#     "elasticsearch~=8.13.2": ["dev"],
#     "sentence-transformers~=3.0.1": ["install"],
#     "python-dotenv~=1.0.1": ["install"],
#     "faiss-cpu~=1.7.2": ["install", "gpu"],
#     "faiss-gpu~=1.7.2": ["gpu"],
#     "gitpython~=3.1.27": ["install", "gpu"],
#     "ipykernel~=6.13.0": ["notebooks"],
#     "ipywidgets~=7.7.0": ["notebooks"],
#     "jsonlines~=3.0.0": ["install", "gpu"],
#     "ninja~=1.10.2.3": ["install", "gpu"],
#     "numpy~=1.21.5": ["install", "gpu"],
#     "packaging~=21.3": ["install", "gpu"],
#     "pandas~=1.4.0": ["install", "gpu"],
#     "psutil~=5.9.0": ["install", "gpu"],
#     "pyarrow~=7.0.0": ["install", "gpu"],
#     "pyserini~=0.20.0": ["install", "gpu"],
#     "pytest~=7.1.1": ["tests"],
#     "pytest-cov~=3.0.0": ["tests"],
#     "pytest-mock~=3.7.0": ["tests"],
#     "pytest-rerunfailures~=10.2": ["tests"],
#     "scikit-learn~=1.0.2": ["install", "gpu"],
#     "signals~=0.0.2": ["install", "gpu"],
#     "stanza~=1.4.0": ["install", "gpu"],
#     "torch>=2.0": ["install", "gpu"],
#     "tox~=3.24.5": ["tests"],
#     "transformers~=4.24.0": ["install", "gpu"],
#     "sentence_transformers": ["install", "gpu"],
#     "tensorboard":["install", "gpu"],
#     "sentencepiece~=0.1.96": ["install", "gpu"],
#     "ujson~=5.1.0": ["install"],
#     "tqdm~=4.64.0": ["install", "gpu"],
#     "frozendict": ["install", "gpu"],
#     "nlp": ["install", "gpu"],
#     "protobuf~=3.20.0": ["install", "gpu"],
#     "tabulate~=0.8.9": ["install", "gpu"],
#     "rouge_score": ["install", "gpu"],
#     "rouge~=1.0.1":["install"],
#     "myst-parser~=0.17.2": ["docs"],
#     "pydata-sphinx-theme~=0.9.0": ["docs"],
#     "sphinx~=4.4.0": ["docs"],
#     "sphinx_design~=0.2.0": ["docs"],
#     "recommonmark~=0.7.1": ["docs"],
#     "grpcio~=1.48.1": ["install", "gpu"],
#     "grpcio-tools~=1.48.1": ["install", "gpu"],ok
#     "fastapi~=0.85.0": ["install", "gpu"],
#     "uvicorn~=0.18.0": ["install", "gpu"],
#     "cachetools~=5.2.0": ["install", "gpu"],
#     "sqlitedict~=2.0.0": ["install", "gpu"],
#     "openai~=0.27.0": ["install", "gpu"],
#     "nltk~=3.8.1": ["install", "gpu"],
#     "python-dotenv~=1.0.1": ["install"]
# }
_deps = {}
if _deps == {}:
    with open(os.path.join(cwd, "requirements.txt"), "r", encoding="utf-8") as requirements_file:
        for line in requirements_file:
            _deps[line.strip()] = ['install']

extras_names = ["install"]
extras = {extra_name: [] for extra_name in extras_names}
for dep_package_name, dep_package_required_by in _deps.items():
    if not dep_package_required_by:
        raise ValueError(
            f"Package '{dep_package_name}' has no associated extras it is required for. "
            f"Choose one or more of: {extras_names}"
        )

    for extra_name in dep_package_required_by:
        try:
            extras[extra_name].append(dep_package_name)
        except KeyError as ex:
            raise KeyError(f"Extras name '{extra_name}' not in {extras_names}") from ex
extras["all"] = list(_deps)

install_requires = extras["install"]

setup(
    name=package_name,
    url="https://github.com/primeqa/docuverse",
    version=version,
    author="PrimeQA/DocUVerse Team",
    author_email="primeqa@us.ibm.com",
    description="State-of-the-art Retrieval/Search engine models, including ElasticSearch, ChromaDB, Milvus, and PrimeQA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache",
    keywords="Question Answering (QA), Machine Reading Comprehension (MRC), Information Retrieval (IR)",
    packages=find_packages(".", include=include_packages),
    include_package_data=True,
    package_data={"": ["*.cpp", "*.cu"]},
    python_requires=">=3.10.0, <3.13.0",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
