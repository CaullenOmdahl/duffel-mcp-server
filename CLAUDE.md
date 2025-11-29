# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Duffel MCP (Model Context Protocol) server that enables LLMs to search for flights, retrieve offers, and create bookings through the Duffel flight booking API. It includes intelligent optimization strategies for finding the cheapest, fastest, or best overall flights.

## Running the Server

```bash
# Run with default stdio transport
python duffel_mcp/server.py

# Run with SSE transport for web deployments
python duffel_mcp/server.py --transport sse --host 0.0.0.0 --port 8000

# Enable debug logging
python duffel_mcp/server.py --debug

# Or run with uv
uv --directory duffel_mcp run python server.py
```

## Testing API Key Permissions

```bash
python duffel_mcp/test_api_key.py
```

This verifies the Duffel API key has required permissions: `air.offer_requests.create`, `air.offers.read`, `air.orders.create`, `air.airlines.read`.

## Architecture

### MCP Server (`duffel_mcp/server.py`)

Single-file FastMCP server built with:
- **FastMCP** framework with lifespan management for persistent HTTP client
- **Pydantic v2** models for input validation
- **httpx** async client with connection reuse
- **Context injection** for progress reporting and logging

### Tools Provided

| Tool | Purpose |
|------|---------|
| `duffel_search_flights` | Search for flights with optimization strategies |
| `duffel_analyze_offers` | Analyze and rank offers from a previous search |
| `duffel_get_offer` | Get latest details for a specific offer |
| `duffel_list_offers` | List/filter offers from a previous search |
| `duffel_create_order` | Book a flight with passenger and payment details |
| `duffel_list_airlines` | Reference data for available airlines |

### Resources

| URI | Description |
|-----|-------------|
| `duffel://airlines` | List of all available airlines |
| `duffel://airlines/{code}` | Details for a specific airline by IATA code |
| `duffel://places/{query}` | Search airports and cities by name |

### Prompts

| Prompt | Description |
|--------|-------------|
| `book_round_trip` | Guided workflow for round-trip booking |
| `find_cheapest` | Strategy for finding affordable options |
| `compare_options` | Analyze and compare multiple flight options |

### Optimization Strategies

The search tool supports these strategies via the `optimization` parameter:

| Strategy | Description |
|----------|-------------|
| `none` | Return results as-is from API |
| `cheapest` | Sort by price ascending |
| `fastest` | Sort by total duration ascending |
| `least_stops` | Sort by number of connections |
| `best` | Weighted score algorithm |
| `earliest` | Sort by departure time (morning first) |
| `latest` | Sort by departure time (evening first) |

### Weighted Scoring (`best` strategy)

Default weights for the composite score (0-100):
- **Price**: 0.4 (lower prices score higher)
- **Duration**: 0.3 (shorter flights score higher)
- **Stops**: 0.2 (fewer stops score higher)
- **Departure Time**: 0.1 (match to preferred window)

Customize via `optimization_weights` parameter with `OptimizationWeights` model.

### Key Constants

- `API_BASE_URL`: `https://api.duffel.com`
- `API_VERSION`: `v2`
- `CHARACTER_LIMIT`: 25000 (truncation limit for responses)
- `DEFAULT_TIMEOUT`: 30.0 seconds

### Response Formats

All tools support two output formats via `response_format` parameter:
- `markdown` (default): Human-readable formatted output with scores
- `json`: Complete API response for programmatic processing

## Configuration

The server requires `DUFFEL_API_KEY_LIVE` environment variable. MCP configuration is in `.mcp.json`.

## Key Implementation Patterns

### Lifespan Management
Persistent HTTP client initialized at startup, reused for all requests:
```python
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    async with httpx.AsyncClient(...) as client:
        yield {"http_client": client}
```

### Context Injection
All tools receive `ctx: Context` for progress reporting:
```python
await ctx.report_progress(0.5, "Searching for flights...")
```

### Structured Logging
All operations log to stderr (stdout reserved for MCP protocol):
```python
logger.info("Flight search: %s, %d passengers", route, count)
logger.error("API error %s: %s", status, message)
```

### Multiple Transports
Supports stdio (default) and HTTP/SSE:
```python
mcp.run()  # stdio
mcp.run(transport="sse", host="0.0.0.0", port=8000)  # HTTP/SSE
```

### Error Handling
Uses `ToolError` exception for proper `isError` flag handling. All errors are logged with context.

### Flight Scoring
Offers are scored using normalized factors combined with configurable weights. The `calculate_flight_score()` function handles all scoring logic.
