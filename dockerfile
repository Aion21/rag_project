FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment files
COPY environment.yml .
COPY requirements.txt .

# Create conda environment
RUN conda env create -f environment.yml

# Activate environment in shell
SHELL ["conda", "run", "-n", "rag_project", "/bin/bash", "-c"]

# Copy source code
COPY . .

# Create necessary folders
RUN mkdir -p data chroma_db

# Install additional pip packages
RUN conda run -n rag_project pip install -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0

# Startup command
CMD ["conda", "run", "--no-capture-output", "-n", "rag_project", "python", "main.py"]