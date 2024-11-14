# Base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

ARG MODEL_URL
ENV MODEL_URL=${MODEL_URL}

ARG CIVITAI_TOKEN
ENV CIVITAI_TOKEN=${CIVITAI_TOKEN}

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
    build-essential vim git wget ffmpeg libsm6 libxext6

# Update and upgrade the system packages (Worker Template)
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Fetch the model using environment variable
COPY builder/model_fetcher.py /model_fetcher.py
RUN python /model_fetcher.py --model_url ${MODEL_URL} --civitai_token ${CIVITAI_TOKEN}
RUN rm /model_fetcher.py

# Add src files (Worker Template)
ADD src .

CMD python3 -u /handler.py --civitai_token=${CIVITAI_TOKEN}
