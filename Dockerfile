FROM continuumio/miniconda3

WORKDIR /app

# Copy the environment.yml into the container
COPY environment.yml /app/

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install any additional dependencies or final setups
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app/

# Set the default command
CMD ["python", "app.py"]
