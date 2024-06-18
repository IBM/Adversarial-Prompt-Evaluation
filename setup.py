import codecs
import os

from setuptools import find_packages, setup

install_requires = [
    "python-dotenv",
    "ibm-generative-ai",
    "sentence_transformers",
    "pandas",
    "rouge_score",
    "nltk",
    "mauve-text",
    "shap",
    "textattack",
    "jupyter",
    "chromadb",
    "fschat[model_worker,webui]",
    "llm-guard",
    "openai",
    "langkit"
]

dev_requires = [
    "pytest",
    "pylint >= 3.1.0",
    "mypy",
    "isort",
    "black",
]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="ape",
    version=get_version("src/ape/__init__.py"),
    description="Adversarial Prompt Evaluation",
    author="DRL IBM",
    author_email="<email>",
    maintainer="DRL IBM",
    maintainer_email="<email>",
    license="MIT",
    install_requires=install_requires,
    include_package_data=True,
    python_requires=">=3.10",
    extras_require={
        "dev": dev_requires,
    }
)
