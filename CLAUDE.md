# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Duffel MCP server that enables LLMs to search for flights, retrieve offers, and create bookings through the Duffel flight booking API. Single-file FastMCP server with intelligent optimization strategies and integrated checkout flow.

## Commands

```bash
# Install dependencies
pip install fastmcp httpx pydantic starlette uvicorn
# Or with uv
uv pip install -e duffel_mcp/

# Run server (stdio transport - default for CLI)
python duffel_mcp/server.py

# Run server (SSE transport - for web/Railway deployment with checkout)
python duffel_mcp/server.py --transport sse --host 0.0.0.0 --port 8080

# Enable debug logging
python duffel_mcp/server.py --debug

# Test API key permissions
DUFFEL_API_KEY_LIVE="your_key" python duffel_mcp/test_api_key.py

# Docker build & run
docker build -t duffel-mcp .
docker run -p 8080:8080 -e DUFFEL_API_KEY_LIVE=xxx -e CHECKOUT_BASE_URL=http://localhost:8080 duffel-mcp
```

## Architecture

### Core Components (`duffel_mcp/server.py`)

**Single-file server** (~3100 lines) containing:
- Pydantic v2 input models for all tools
- Optimization/scoring logic in `calculate_flight_score()`
- 7 MCP tools, 4 resources, 3 prompts
- Checkout flow with Duffel Payments integration

**Key functions:**
- `_make_api_request()`: Authenticated Duffel API calls
- `_optimize_offers()`: Apply sorting/scoring strategies
- `_create_payment_intent()`: Create Duffel Payment Intent for checkout
- `_confirm_payment_intent()`: Confirm payment after card collection
- `create_combined_app()`: Combine MCP SSE with Starlette checkout routes

### Checkout Flow

The `duffel_create_checkout` tool creates a hosted checkout page:

1. **MCP Tool** creates payment intent + session → returns checkout URL
2. **Checkout Page** (`/checkout/{session_id}`) shows flight summary + Duffel card component
3. **Confirmation** (`/checkout/{session_id}/confirm`) confirms payment + creates order
4. **Success Page** (`/checkout/{session_id}/success`) shows booking reference

**Key models:**
- `CheckoutSession`: Stored session with offer, passengers, payment intent
- `CreateCheckoutInput`: Input for `duffel_create_checkout` tool
- `checkout_sessions`: In-memory session storage (dict)

**Environment variables:**
- `CHECKOUT_BASE_URL`: Full URL for checkout links (e.g., `https://your-app.railway.app`)

### Optimization Strategies

The `optimization` parameter accepts: `none`, `cheapest`, `fastest`, `least_stops`, `best`, `earliest`, `latest`

The `best` strategy uses weighted scoring (default weights):
- Price: 0.4, Duration: 0.3, Stops: 0.2, Departure Time: 0.1

### MCP Resources

- `duffel://airlines` - All available airlines
- `duffel://airlines/{iata_code}` - Single airline by code
- `duffel://places/{query}` - Airport/city search
- `duffel://instructions` - AI agent guidelines for smart travel assistance

## Configuration

**Required:** `DUFFEL_API_KEY_LIVE` environment variable

The API key needs these permissions:
- `air.offer_requests.create`
- `air.offers.read`
- `air.orders.create`
- `air.airlines.read`
- `payments.payment_intents.create` (for checkout flow)

**Optional:** `CHECKOUT_BASE_URL` - Base URL for checkout links

MCP client config is in `.mcp.json`.

## Deployment

**Railway:** Deploy via GitHub connection, uses `Dockerfile` with SSE transport on port 8080. Set `CHECKOUT_BASE_URL` to your Railway domain.

**Docker:** `Dockerfile` runs `server.py --transport sse --host 0.0.0.0 --port 8080`

## Key Patterns

- All logging goes to stderr (stdout reserved for MCP protocol)
- Tools receive `ctx: Context` for `report_progress()` calls
- `ToolError` exception sets MCP `isError` flag
- Character limit: 25000 (responses truncated with `_truncate_if_needed()`)
- SSE transport combines MCP routes with Starlette checkout routes via `Mount`
