FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/usr/local/bin:$PATH" \
    APP_HOME=/app

# Install essential packages and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    git \
    vim \
    nano \
    htop \
    netcat \
    iputils-ping \
    net-tools \
    dnsutils \
    tzdata \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 directly from the Ubuntu repositories
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y
    

# Create and set working directory
WORKDIR $APP_HOME

# Copy requirements file
COPY . .

# Install Python dependencies
# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt --no-deps && \
    pip3 install --no-cache-dir fastapi==0.115.9 && \
    pip3 install --no-cache-dir smolagents[litellm]

# Copy application code
# COPY . .

# Expose port
EXPOSE 9000

# Set up healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Set entrypoint
RUN chmod +x /app/run.sh
ENTRYPOINT ["/app/run.sh"]