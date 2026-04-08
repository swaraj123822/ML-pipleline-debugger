# Hugging Face Spaces — OpenEnv ML Pipeline Debugger
# Tag: openenv

FROM python:3.12-slim

LABEL org.opencontainers.image.title="ML Pipeline Debugger"
LABEL org.opencontainers.image.description="OpenEnv RL environment for LLM ML pipeline debugging"
LABEL tags="openenv"

WORKDIR /app

# Install server dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source — models.py and client.py live at root
COPY models.py       /app/models.py
COPY client.py       /app/client.py
COPY server/         /app/server/
COPY openenv.yaml    /app/openenv.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Start FastAPI server via uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]