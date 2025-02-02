# refer from : https://hub.docker.com/r/nvidia/cuda/tags
# please confirm base image tag name from above URL
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set shell option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Update and upgrade
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Setting timezone, language
RUN apt-get update &&\
    apt-get install -y --no-install-recommends locales &&\
    #  Add locale configuration to prevent build from stopping
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime &&\
    apt-get install -y --no-install-recommends \
    software-properties-common tzdata &&\
    locale-gen ja_JP ja_JP.UTF-8 &&\
    update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8 &&\
    add-apt-repository universe &&\
    rm -rf /var/lib/apt/lists/*
ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

# Install basic packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \
    wget \
    nano \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for building python and pkgs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    build-essential \
    python3-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install pkgs for PyQt5
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinput0 \
    libxcb-xkb1 \
    libxcb-xrm0 && \
    rm -rf /var/lib/apt/lists/*

# Add user and group, then allow sudo
ARG UID
ARG GID
ARG PASSWORD
ARG USER_NAME
ARG GROUP_NAME
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
RUN groupadd -g ${GID} ${GROUP_NAME} && \
    useradd -l -u ${UID} -g ${GID} -G sudo -s /bin/bash -m ${USER_NAME} && \
    echo "${USER_NAME}:${PASSWORD}" | chpasswd && \
    echo "Defaults visiblepw" >> /etc/sudoers && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install python3 environment tools
RUN add-apt-repository universe && \
    # ↑ prevent error from "Unable to locate package pip"
    apt-get update && \
    apt-get install python3-pip python3.10-venv -y --no-install-recommends && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# Switch to user
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# Set git user name and email
RUN git config --global user.name ${GIT_USER_NAME} && \
    git config --global user.email ${GIT_USER_EMAIL}

# Switch to user
ARG USER_NAME
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy setup.sh and run
COPY setup.sh /home/${USER_NAME}/setup.sh
RUN bash ~/setup.sh


CMD ["/bin/bash"]
