# Python version family used in the original experiments.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tools are needed for dependencies without Python 3.14
# wheels, including immutables used by Snakemake.
# libgomp1 supports compiled scientific packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install environments before copying the project so Docker
# can cache dependency installation when source code changes.
COPY requirements-science.txt requirements-workflow.txt ./

RUN python -m venv /opt/qsar-science \
    && /opt/qsar-science/bin/python -m pip install \
       --no-cache-dir -r requirements-science.txt \
    && /opt/qsar-science/bin/python -m pip check

RUN python -m venv /opt/qsar-workflow \
    && /opt/qsar-workflow/bin/python -m pip install \
       --no-cache-dir -r requirements-workflow.txt \
    && /opt/qsar-workflow/bin/python -m pip check

# Install automated testing tools in the scientific environment.
# This separate layer preserves cached scientific dependencies.
COPY requirements-test.txt ./

RUN /opt/qsar-science/bin/python -m pip install \
       --no-cache-dir -r requirements-test.txt \
    && /opt/qsar-science/bin/python -m pip check

# Include the Git-tracked project sources and archived results.
# Local environments, databases, and artifacts are excluded.
COPY . .

# Existing scripts expect these relative paths.
RUN ln -s /opt/qsar-science /app/.venv \
    && ln -s /opt/qsar-workflow /app/.workflow-venv

# Basic checks during image construction.
RUN .venv/bin/python -c \
    "import rdkit, sklearn, mlflow, optuna; print('Scientific packages: OK')" \
    && .workflow-venv/bin/snakemake --version

# The default workflow rebuilds data in an isolated directory
# and checks the preserved research results.
CMD [".workflow-venv/bin/snakemake", "--cores", "1"]
