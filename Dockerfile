# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG TORCH_INDEX=""
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/data/hf \
    TRITON_CACHE_DIR=/data/triton HOME=/tmp USER=flide

COPY dist/flide-*.whl /wheels/
RUN if [ -n "$TORCH_INDEX" ]; then pip install "torch>=2.12,<2.15" torchvision --index-url "$TORCH_INDEX"; fi \
    && pip install /wheels/flide-*.whl torchvision \
    && rm -r /wheels \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin flide \
    && mkdir -p /data/hf /data/triton && chmod 1777 /data/hf /data/triton

USER flide
EXPOSE 8000
ENTRYPOINT ["flide"]
CMD ["--help"]
