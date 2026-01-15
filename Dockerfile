# Method of Moderation Dockerfile
# Single Source of Truth: reproduce/docker/setup.sh
#
# Build:   docker build -t method-of-moderation .
# Run:     docker run -it --rm method-of-moderation
# Jupyter: docker run -it --rm -p 8888:8888 method-of-moderation jupyter lab --ip=0.0.0.0 --no-browser

FROM mcr.microsoft.com/devcontainers/python:3.12

# Metadata
LABEL org.opencontainers.image.title="Method of Moderation"
LABEL org.opencontainers.image.description="Development environment for Method of Moderation REMARK"
LABEL org.opencontainers.image.source="https://github.com/econ-ark/method-of-moderation"

# Environment
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory with correct ownership
RUN mkdir -p /workspace && chown vscode:vscode /workspace
WORKDIR /workspace

# Copy files with correct ownership
COPY --chown=vscode:vscode . /workspace/

# Run setup as vscode user
USER vscode
RUN chmod +x /workspace/reproduce/docker/setup.sh && \
    bash /workspace/reproduce/docker/setup.sh

# Set runtime environment
ENV PATH="/workspace/.venv/bin:/home/vscode/.local/bin:$PATH"
ENV PYTHONPATH="/workspace/code"

# Expose common ports
EXPOSE 8888 8866

# Default command
CMD ["/bin/bash"]
