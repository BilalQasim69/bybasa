# Use RunPod's official CUDA-enabled base image
FROM runpod/base:0.4.0-cuda11.8.0

# Set the Hugging Face token as an environment variable
ENV HUGGINGFACE_TOKEN=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

# Install Python dependencies from requirements.txt
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Copy all source files (including LoRA weights) to /workspace
COPY src/ /workspace/

# Set the working directory to /workspace
WORKDIR /workspace

# Run the worker with Python 3.11
CMD ["python3.11", "-u", "handler.py"]
