# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . /app/traffic_signal_env/server/
COPY ../models.py /app/traffic_signal_env/models.py

# Create __init__.py files so Python treats folders as packages
RUN touch /app/traffic_signal_env/__init__.py
RUN touch /app/traffic_signal_env/server/__init__.py

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Run the server
CMD ["python", "-m", "uvicorn", "traffic_signal_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]