# Use a Conda-compatible base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Install dependencies using the conda environment file
RUN conda env create -f environment.yml

# Copy the rest of the project files
COPY . .

# Expose port 8000
EXPOSE 8000

# The command to run the app in the new environment
CMD ["conda", "run", "-n", "my-fastapi-app", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]