FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

WORKDIR /app
COPY requirements.txt requirements-edge.txt pyproject.toml README.md /app/
COPY ai_core /app/ai_core
COPY edge /app/edge
RUN python3 -m pip install --no-cache-dir -r requirements-edge.txt

CMD ["python3", "edge/runtime/run_edge_inference.py", "--source", "0"]
