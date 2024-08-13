ARG CUDA_VERSION=12.4.1
#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
       ccache software-properties-common python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip git \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /workspace

COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt \
    && rm -rf /root/.cache/pip

ARG torch_cuda_arch_list='6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BASE BUILD IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM base AS build

ARG PYTHON_VERSION=3

COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt \
    && rm -rf /root/.cache/pip

COPY kernels kernels
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
COPY pyproject.toml pyproject.toml
COPY aphrodite aphrodite

ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=8
ENV NVCC_THREADS=${nvcc_threads}

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist \
    && rm -rf /root/.cache/pip /root/.cache/ccache

#################### DEV IMAGE ####################
FROM base as dev

COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-dev.txt \
    && rm -rf /root/.cache/pip
#################### DEV IMAGE ####################


#################### Aphrodite installation IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS aphrodite-base
ARG CUDA_VERSION=12.4.1
WORKDIR /aphrodite-workspace

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip git vim \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN --mount=type=bind,from=build,src=/workspace/dist,target=/aphrodite-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install dist/*.whl --verbose \
    && rm -rf /root/.cache/pip
#################### Aphrodite installation IMAGE ####################


#################### OPENAI API SERVER ####################
FROM aphrodite-base AS aphrodite-openai

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install accelerate hf_transfer 'modelscope!=1.15.0' \
    && rm -rf /root/.cache/pip

ENV NUMBA_CACHE_DIR=$HOME/.numba_cache

ENTRYPOINT ["python3", "-m", "aphrodite.endpoints.openai.api_server"]
#################### OPENAI API SERVER ####################
