# Build shit
FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt /app/

# Install dependencies and keep pip cache persistent
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Final leftover shit
FROM python:3.10-slim-buster

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the app
COPY . /app/

CMD ["python", "app.py"]
