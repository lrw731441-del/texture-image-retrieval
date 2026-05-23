FROM python:3.12-slim-bookworm

WORKDIR /app

# Use Tencent Cloud APT mirror for faster downloads in China
RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.cloud.tencent.com/debian|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://deb.debian.org/debian|http://mirrors.cloud.tencent.com/debian|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use Tencent Cloud PyPI mirror for faster pip installs
RUN pip install --no-cache-dir \
    -i https://mirrors.cloud.tencent.com/pypi/simple \
    --trusted-host mirrors.cloud.tencent.com \
    -r requirements.txt

COPY *.py ./
COPY templates/ templates/
COPY images/ images/
COPY data/ data/

EXPOSE 8000

CMD ["python", "app.py"]
