FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY duffel_mcp/pyproject.toml .
RUN pip install fastmcp httpx pydantic starlette uvicorn

# Copy server code
COPY duffel_mcp/server.py .

# Set checkout base URL (Railway injects RAILWAY_PUBLIC_DOMAIN)
ENV CHECKOUT_BASE_URL=""

# Expose port for SSE transport + checkout pages
EXPOSE 8080

# Run with SSE transport (includes checkout routes)
CMD ["python", "server.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]
