FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Install dependencies (by default torch with GPU, it's ok if you don't have GPU it'll still work, i think)
COPY requirements.txt /app/
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Run the app (--debug if env var is set)
CMD ["python3", "app.py", "--debug"]
