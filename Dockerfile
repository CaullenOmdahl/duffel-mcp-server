FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY duffel_mcp/pyproject.toml .
RUN pip install fastmcp httpx pydantic

# Copy server code
COPY duffel_mcp/server.py .

# Expose port for SSE transport
EXPOSE 8080

# Run with SSE transport
CMD ["python", "server.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]
