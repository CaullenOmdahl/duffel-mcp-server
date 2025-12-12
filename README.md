# Duffel MCP Server

A Model Context Protocol (MCP) server for the Duffel flight booking API. This server enables LLMs to search for flights, retrieve offers, and create bookings through a compliant MCP interface with intelligent optimization strategies.

## Features

### Tools Provided

1. **duffel_search_flights** - Search for flights with optimization
   - One-way and round-trip searches
   - Multiple passengers support
   - Cabin class preferences
   - Connection filters (non-stop, max connections)
   - **Optimization strategies**: cheapest, fastest, best, least_stops, earliest, latest
   - **Weighted scoring** for finding optimal flights

2. **duffel_analyze_offers** - Analyze and rank offers from a search
   - Apply optimization strategies post-search
   - Custom weight configuration for price, duration, stops, departure time
   - Market overview with price/duration ranges
   - Scored rankings for easy comparison

3. **duffel_get_offer** - Get detailed, up-to-date offer information
   - Latest pricing
   - Complete itinerary details
   - Booking conditions
   - Available services

4. **duffel_list_offers** - List and filter offers from a search
   - Pagination support
   - Sort by price or duration
   - Filter by connections

5. **duffel_create_order** - Create flight bookings
   - Complete passenger details
   - Payment processing
   - Booking confirmation

6. **duffel_list_airlines** - Reference data for available airlines

### Resources

- `duffel://airlines` - List of all available airlines
- `duffel://airlines/{iata_code}` - Details for a specific airline
- `duffel://places/{query}` - Search for airports and cities

### Prompts

- `book_round_trip` - Interactive workflow for booking round-trip flights
- `find_cheapest` - Strategy for finding the most affordable options
- `compare_options` - Compare and analyze multiple flight options

## Optimization Strategies

| Strategy | Description |
|----------|-------------|
| `cheapest` | Sort by price ascending |
| `fastest` | Sort by total duration ascending |
| `least_stops` | Sort by number of connections |
| `best` | Weighted score combining price, duration, stops, and departure time |
| `earliest` | Sort by departure time (morning first) |
| `latest` | Sort by departure time (evening first) |

### Weighted Scoring (Best Strategy)

The `best` strategy uses configurable weights to calculate a composite score (0-100):

| Factor | Default Weight | Description |
|--------|---------------|-------------|
| Price | 0.4 | Lower prices score higher |
| Duration | 0.3 | Shorter flights score higher |
| Stops | 0.2 | Fewer stops score higher |
| Departure Time | 0.1 | Match to preferred time window |

Customize weights with the `optimization_weights` parameter:
```json
{
  "optimization_weights": {
    "price": 0.5,
    "duration": 0.3,
    "stops": 0.15,
    "departure_time": 0.05
  }
}
```

## Installation

```bash
# Install with uv
uv pip install -e .

# Or with pip
pip install -e .
```

## Configuration

Set your Duffel API key as an environment variable:

```bash
export DUFFEL_API_KEY_LIVE="your_api_key_here"
```

## Usage

### As an MCP Server

Add to your MCP configuration file (e.g., `.mcp.json`):

```json
{
  "mcpServers": {
    "duffel": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/duffel_mcp",
        "run",
        "python",
        "server.py"
      ],
      "env": {
        "DUFFEL_API_KEY_LIVE": "your_api_key_here"
      }
    }
  }
}
```

### Direct Execution

```bash
python server.py
```

## API Key Requirements

Your Duffel API key must have the following permissions:
- `air.offer_requests.create` - For searching flights
- `air.offers.read` - For retrieving offer details
- `air.orders.create` - For creating bookings
- `air.airlines.read` - For airline reference data

To get an API key:
1. Sign up at [duffel.com](https://duffel.com)
2. Complete account verification
3. Generate an API key with the required permissions in your dashboard

## Examples

### Search for the Best Flights

```
Search for the best flights from JFK to LAX on December 15, optimizing for price and convenience
```

### Find the Cheapest Option

```
Find the cheapest flight from SGN to KUL, departing November 21, returning November 23
```

### Analyze Search Results

```
Analyze the search results and show me the top 5 options with the best overall value
```

### Search with Time Preference

```
Search for morning flights from LHR to CDG with the best optimization
```

## Response Formats

All tools support two output formats:

- **Markdown** (default): Human-readable formatted output with scores
- **JSON**: Complete API response for programmatic processing

Specify the format with the `response_format` parameter.

## Error Handling

The server provides clear, actionable error messages:
- Authentication errors (401) - Check your API key
- Permission errors (403) - Verify API key permissions
- Validation errors (422) - Check input parameters
- Not found errors (404) - Verify resource IDs
- Rate limiting (429) - Wait before retrying

## Architecture

Built with MCP SDK best practices:
- **FastMCP** framework with lifespan management
- **Pydantic v2** for input validation
- **Context injection** for progress reporting and logging
- **Persistent HTTP client** via lifespan for connection reuse
- **Resources** for reference data access
- **Prompts** for guided workflows
- **Multiple transports**: stdio (default) and HTTP/SSE
- **Structured logging** to stderr (preserves stdout for MCP protocol)
- Tool annotations for client behavior hints
- Character limits and graceful truncation
- Async/await for all I/O operations

## Command Line Options

```bash
# Run with default stdio transport
python server.py

# Run with SSE transport for web deployments
python server.py --transport sse --host 0.0.0.0 --port 8000

# Enable debug logging
python server.py --debug
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transport` | `stdio` | Transport type: `stdio` or `sse` |
| `--host` | `127.0.0.1` | Host for SSE transport |
| `--port` | `8000` | Port for SSE transport |
| `--debug` | `false` | Enable debug logging |

## Deployment

### Local Installation (.mcpb)

Package for one-click installation in Claude Desktop:
```bash
npm install -g @anthropic-ai/mcpb
mcpb pack  # Creates duffel-flights-1.0.0.mcpb
```

### Hosted Server

Deploy as a hosted SSE server:

| Platform | Command |
|----------|---------|
| Railway | Connect GitHub repo at railway.app |
| Render | Connect GitHub repo at render.com |
| Fly.io | `fly launch && fly secrets set DUFFEL_API_KEY_LIVE=xxx` |
| Docker | `docker build -t duffel-mcp . && docker run -p 8000:8000 -e DUFFEL_API_KEY_LIVE=xxx duffel-mcp` |

Once deployed, connect to your server at `https://your-app.railway.app/sse`

## License

MIT
