FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Install dependencies directly (skip the builder stage)
COPY requirements.txt /app/
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

CMD ["python", "app.py"]
