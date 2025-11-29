# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Duffel MCP server that enables LLMs to search for flights, retrieve offers, and create bookings through the Duffel flight booking API. Single-file FastMCP server with intelligent optimization strategies.

## Commands

```bash
# Install dependencies
pip install fastmcp httpx pydantic
# Or with uv
uv pip install -e duffel_mcp/

# Run server (stdio transport - default for CLI)
python duffel_mcp/server.py

# Run server (SSE transport - for web/Railway deployment)
python duffel_mcp/server.py --transport sse --host 0.0.0.0 --port 8080

# Enable debug logging
python duffel_mcp/server.py --debug

# Test API key permissions
DUFFEL_API_KEY_LIVE="your_key" python duffel_mcp/test_api_key.py

# Docker build & run
docker build -t duffel-mcp .
docker run -p 8080:8080 -e DUFFEL_API_KEY_LIVE=xxx duffel-mcp
```

## Architecture

### Core Components (`duffel_mcp/server.py`)

**Single-file server** (~1750 lines) containing:
- Pydantic v2 input models for all tools (lines 164-443)
- Optimization/scoring logic in `calculate_flight_score()` (lines 614-651)
- 6 MCP tools, 4 resources, 3 prompts

**Key functions:**
- `_make_api_request()`: Authenticated Duffel API calls
- `_optimize_offers()`: Apply sorting/scoring strategies
- `_handle_api_error()`: Consistent error formatting with HTTP status handling

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

MCP client config is in `.mcp.json`.

## Deployment

**Railway:** Deploy via GitHub connection, uses `Dockerfile` with SSE transport on port 8080
**Docker:** `Dockerfile` runs `server.py --transport sse --host 0.0.0.0 --port 8080`

## Key Patterns

- All logging goes to stderr (stdout reserved for MCP protocol)
- Tools receive `ctx: Context` for `report_progress()` calls
- `ToolError` exception sets MCP `isError` flag
- Character limit: 25000 (responses truncated with `_truncate_if_needed()`)

## Technical Debt

### Payment/Booking Flow
- **Issue**: `duffel_create_order` requires payment details (card info) which can't be safely collected in chat
- **Duffel has no hosted checkout** - no redirect URL like Stripe Checkout
- **Options to fix**:
  1. Switch to **hold orders** - reserve flight without payment, return booking details for user to pay elsewhere
  2. Build a **separate payment page** with Stripe/Duffel Payments component, return URL to user
  3. **Remove booking capability** - make this search/advisory only, direct users to airline sites
- **Recommended**: Hold orders + clear instructions on how to complete payment
