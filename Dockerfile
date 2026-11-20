FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV PYTHONPATH=""
ARG PROJECT_DIR=/workspace/coc_auto_labeling
ENV PROJECT_DIR=${PROJECT_DIR}

RUN apt-get update -y \
    && apt-get -y install \
        git \
        curl \
        unzip \
        ffmpeg \
        libglm-dev \
        libopenmpi-dev \
        libsm6 \
        libxext6 \
        ninja-build \
        python3 \
        python3-pip \
        python-is-python3 \
        tzdata \
        vim \
        htop \
        rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use the same Python 3.12 family as the locked standalone runtime.
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12.11 \
    && uv venv --python 3.12.11 --seed /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements-lock.txt /tmp/requirements-lock.txt

RUN pip install --upgrade pip
# The trajdata ref declares protobuf==3.19.4, while the locked
# runtime stack uses newer packages that require protobuf 6.x. Install trajdata
# outside pip's dependency resolver so the exact Git ref is still used.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    -r /tmp/requirements-lock.txt
RUN pip install --no-cache-dir --no-deps \
    "trajdata @ git+https://github.com/NVlabs/trajdata.git@coc_autolabeling-1.0.0"

COPY . ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

RUN pip install --no-cache-dir --no-deps -e .

RUN git config --global --add safe.directory ${PROJECT_DIR}

CMD ["bash"]
