# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:22.04


RUN DEBIAN_FRONTEND=noninteractive apt update --fix-missing -y
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y 
RUN DEBIAN_FRONTEND=noninteractive apt install python3-pip npm git libsndfile1-dev python3-numba python3-opencv python3-psycopg2 python3-numpy python3-grpc-tools python3-grpcio ffmpeg python3-imageio -y
RUN DEBIAN_FRONTEND=noninteractive apt install libmariadbclient-dev-compat imagemagick python3-sklearn -y

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV NUMBA_CACHE_DIR=/tmp/

# Install pip requirements

RUN python3 -m pip install imageio-ffmpeg
RUN python3 -m pip install msgpack
RUN python3 -m pip install numpy
RUN python3 -m pip install PyYAML
RUN python3 -m pip install redis 

# RUN python3 -m pip install scikit-image
RUN python3 -m pip install librosa 
RUN python3 -m pip install ffmpeg-python

WORKDIR /app/analyser/
COPY . /app/analyser/

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app/analyser/
USER appuser

ENV PYTHONPATH="/app/analyser/"
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python3", "/app/analyser/server.py", "--config", "config.yml"]
# CMD ["python", "backend.py"]
