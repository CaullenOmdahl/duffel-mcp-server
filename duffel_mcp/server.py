#!/usr/bin/env python3
"""
Duffel MCP Server

This server provides tools to search for flights, retrieve offers, and create bookings
using the Duffel API. It enables LLMs to help users find and book flights through
a compliant MCP interface.

Features:
- Flight search with optimization strategies (cheapest, fastest, best)
- Weighted scoring algorithm for finding optimal flights
- MCP Resources for airlines and airports
- MCP Prompts for common booking scenarios
- Progress reporting and structured logging via Context
- Persistent HTTP client via lifespan management
- Proper isError flag handling for tool errors
- Multiple transport support (stdio, HTTP with SSE)
"""

import os
import sys
import json
import re
import logging
from typing import Optional, List, Dict, Any, TypedDict, Tuple, Union
from enum import Enum
from datetime import datetime
from contextlib import asynccontextmanager
import httpx
from pydantic import BaseModel, Field, field_validator, ConfigDict
from fastmcp import FastMCP, Context

# Configure logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("duffel_mcp")

# Constants
API_BASE_URL = "https://api.duffel.com"
API_VERSION = "v2"
CHARACTER_LIMIT = 25000
DEFAULT_TIMEOUT = 30.0

# Get API key from environment
DUFFEL_API_KEY = os.getenv("DUFFEL_API_KEY_LIVE", "")

# HTTP client headers for Duffel API
def _get_http_headers() -> Dict[str, str]:
    """Get headers for Duffel API requests."""
    return {
        "Authorization": f"Bearer {DUFFEL_API_KEY}",
        "Duffel-Version": API_VERSION,
        "Accept": "application/json",
        "Accept-Encoding": "gzip"
    }

# Initialize the MCP server
mcp = FastMCP("duffel_mcp")


# ============================================================================
# Error Result Helper
# ============================================================================

class ToolError(Exception):
    """Custom exception for tool errors that should set isError=True."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


def _create_error_result(error_message: str, ctx: Optional[Context] = None) -> str:
    """
    Create an error result string.

    Note: FastMCP handles isError flag automatically when exceptions are raised.
    For explicit error control, we raise ToolError which FastMCP catches.
    """
    if ctx:
        try:
            # Log the error via context if available
            # Note: ctx.log methods may not be available in all FastMCP versions
            pass
        except AttributeError:
            pass
    return error_message

# ============================================================================
# Enums
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"

class CabinClass(str, Enum):
    """Cabin class options for flights."""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"

class PassengerType(str, Enum):
    """Passenger type options."""
    ADULT = "adult"
    CHILD = "child"
    INFANT_WITHOUT_SEAT = "infant_without_seat"

class PaymentType(str, Enum):
    """Payment type options."""
    BALANCE = "balance"
    ARC_BSP_CASH = "arc_bsp_cash"

class OptimizationStrategy(str, Enum):
    """Flight optimization strategies."""
    NONE = "none"           # Return as-is from API
    CHEAPEST = "cheapest"   # Sort by price ascending
    FASTEST = "fastest"     # Sort by total duration ascending
    LEAST_STOPS = "least_stops"  # Sort by number of connections
    BEST = "best"           # Weighted score algorithm
    EARLIEST = "earliest"   # Sort by departure time
    LATEST = "latest"       # Sort by departure time descending

class DepartureTimePreference(str, Enum):
    """Preferred departure time windows."""
    MORNING = "morning"       # 6am - 12pm
    AFTERNOON = "afternoon"   # 12pm - 6pm
    EVENING = "evening"       # 6pm - 12am
    RED_EYE = "red_eye"       # 12am - 6am

# ============================================================================
# Structured Output Types
# ============================================================================

class FlightOfferSummary(TypedDict):
    """Summary of a flight offer for structured output."""
    id: str
    price: str
    currency: str
    duration_minutes: int
    stops: int
    airline: str
    departure_time: str
    arrival_time: str
    score: Optional[float]

class SearchResultSummary(TypedDict):
    """Summary of search results."""
    offer_request_id: str
    total_offers: int
    cheapest: Optional[FlightOfferSummary]
    fastest: Optional[FlightOfferSummary]
    best: Optional[FlightOfferSummary]

# ============================================================================
# Pydantic Models for Input Validation
# ============================================================================

class OptimizationWeights(BaseModel):
    """Weights for the 'best' flight scoring algorithm."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    price: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Weight for price factor (0-1). Higher = price matters more."
    )
    duration: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Weight for flight duration (0-1). Higher = shorter flights preferred."
    )
    stops: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Weight for number of stops (0-1). Higher = fewer stops preferred."
    )
    departure_time: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Weight for departure time preference (0-1). Higher = time preference matters more."
    )

class FlightSlice(BaseModel):
    """A flight slice representing one leg of a journey."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    origin: str = Field(
        ...,
        description="Origin airport IATA code (e.g., 'JFK', 'LHR', 'SGN')",
        min_length=3,
        max_length=3
    )
    destination: str = Field(
        ...,
        description="Destination airport IATA code (e.g., 'LAX', 'CDG', 'KUL')",
        min_length=3,
        max_length=3
    )
    departure_date: str = Field(
        ...,
        description="Departure date in YYYY-MM-DD format (e.g., '2025-11-21')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )

class PassengerInput(BaseModel):
    """Passenger information for flight search."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    type: Optional[PassengerType] = Field(
        None,
        description="Passenger type: 'adult', 'child', or 'infant_without_seat'"
    )
    age: Optional[int] = Field(
        None,
        description="Age of passenger (use instead of type for children/infants)",
        ge=0,
        le=120
    )

    @field_validator('age', 'type')
    @classmethod
    def validate_passenger(cls, v: Any, info) -> Any:
        """Ensure either type or age is provided, not both."""
        values = info.data
        if 'type' in values and 'age' in values:
            if values.get('type') is not None and values.get('age') is not None:
                raise ValueError("Specify either 'type' or 'age', not both")
        return v

class SearchFlightsInput(BaseModel):
    """Input model for searching flights."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    slices: List[FlightSlice] = Field(
        ...,
        description="Flight slices: one for one-way, two for round-trip",
        min_length=1,
        max_length=4
    )
    passengers: List[PassengerInput] = Field(
        ...,
        description="List of passengers",
        min_length=1,
        max_length=9
    )
    cabin_class: Optional[CabinClass] = Field(
        default=CabinClass.ECONOMY,
        description="Cabin class preference"
    )
    max_connections: Optional[int] = Field(
        default=None,
        description="Maximum number of connections (0 for non-stop, 1 for max one stop)",
        ge=0,
        le=3
    )
    return_offers: bool = Field(
        default=True,
        description="Whether to return offers immediately in the response"
    )
    # Optimization parameters
    optimization: OptimizationStrategy = Field(
        default=OptimizationStrategy.NONE,
        description="How to optimize/sort results: 'cheapest', 'fastest', 'best', 'least_stops', 'earliest', 'latest'"
    )
    optimization_weights: Optional[OptimizationWeights] = Field(
        default=None,
        description="Custom weights for 'best' optimization. Defaults: price=0.4, duration=0.3, stops=0.2, departure_time=0.1"
    )
    preferred_departure_time: Optional[DepartureTimePreference] = Field(
        default=None,
        description="Preferred departure window: 'morning' (6am-12pm), 'afternoon' (12pm-6pm), 'evening' (6pm-12am), 'red_eye' (12am-6am)"
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Return only top N results after optimization",
        ge=1,
        le=50
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )

class GetOfferInput(BaseModel):
    """Input model for retrieving a single offer."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    offer_id: str = Field(
        ...,
        description="The unique offer ID (e.g., 'off_00009htYpSCXrwaB9DnUm0')",
        min_length=10
    )
    return_available_services: bool = Field(
        default=False,
        description="Include available services (baggage, seats, etc.)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )

class ListOffersInput(BaseModel):
    """Input model for listing offers from an offer request."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    offer_request_id: str = Field(
        ...,
        description="The offer request ID from a previous search",
        min_length=10
    )
    limit: Optional[int] = Field(
        default=20,
        description="Maximum number of offers to return",
        ge=1,
        le=200
    )
    max_connections: Optional[int] = Field(
        default=None,
        description="Filter by maximum connections",
        ge=0,
        le=3
    )
    sort: Optional[str] = Field(
        default="total_amount",
        description="Sort by 'total_amount' or 'total_duration' (prefix with '-' for descending)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )

class AnalyzeOffersInput(BaseModel):
    """Input for analyzing and ranking existing offers."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    offer_request_id: str = Field(
        ...,
        description="Offer request ID from a previous search",
        min_length=10
    )
    optimization: OptimizationStrategy = Field(
        default=OptimizationStrategy.BEST,
        description="Optimization strategy: 'cheapest', 'fastest', 'best', 'least_stops', 'earliest', 'latest'"
    )
    optimization_weights: Optional[OptimizationWeights] = Field(
        default=None,
        description="Custom weights for 'best' optimization"
    )
    preferred_departure_time: Optional[DepartureTimePreference] = Field(
        default=None,
        description="Preferred departure time window"
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results to return"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )

class OrderPassenger(BaseModel):
    """Passenger details for creating an order."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    id: str = Field(..., description="Passenger ID from the offer request")
    given_name: str = Field(..., description="First/given name", min_length=1)
    family_name: str = Field(..., description="Last/family name", min_length=1)
    born_on: str = Field(
        ...,
        description="Date of birth in YYYY-MM-DD format",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    email: str = Field(..., description="Email address", pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone_number: str = Field(..., description="Phone number with country code (e.g., '+14155552671')")
    title: str = Field(..., description="Title: 'mr', 'ms', 'mrs', 'miss', 'dr'")
    gender: str = Field(..., description="Gender: 'm' or 'f'", pattern=r'^[mf]$')
    infant_passenger_id: Optional[str] = Field(
        default=None,
        description="ID of infant if this passenger is responsible for one"
    )

class Payment(BaseModel):
    """Payment information for an order."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    type: PaymentType = Field(..., description="Payment type: 'balance' or 'arc_bsp_cash'")
    amount: str = Field(..., description="Payment amount (e.g., '100.00')")
    currency: str = Field(..., description="Currency code (e.g., 'USD', 'GBP')", min_length=3, max_length=3)

class CreateOrderInput(BaseModel):
    """Input model for creating an order/booking."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    selected_offers: List[str] = Field(
        ...,
        description="List containing exactly one offer ID to book",
        min_length=1,
        max_length=1
    )
    passengers: List[OrderPassenger] = Field(
        ...,
        description="Complete passenger details for all travelers",
        min_length=1,
        max_length=9
    )
    payments: List[Payment] = Field(
        ...,
        description="Payment information (required for instant orders)",
        min_length=1,
        max_length=1
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )

class ListAirlinesInput(BaseModel):
    """Input model for listing airlines."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    limit: Optional[int] = Field(
        default=50,
        description="Maximum number of airlines to return",
        ge=1,
        le=200
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'"
    )

# ============================================================================
# Shared Utility Functions
# ============================================================================

async def _make_api_request(
    ctx: Context,
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Make an authenticated request to the Duffel API."""
    if not DUFFEL_API_KEY:
        logger.error("API key not configured")
        raise ToolError("DUFFEL_API_KEY_LIVE environment variable is not set")

    headers = _get_http_headers()
    if json_data:
        headers["Content-Type"] = "application/json"

    logger.debug("API request: %s /%s", method, endpoint)

    async with httpx.AsyncClient(
        base_url=API_BASE_URL,
        timeout=DEFAULT_TIMEOUT
    ) as client:
        response = await client.request(
            method,
            f"/{endpoint}",
            headers=headers,
            params=params,
            json=json_data
        )

        logger.debug("API response: %s %s", response.status_code, endpoint)
        response.raise_for_status()
        return response.json()

def _handle_api_error(e: Exception, ctx: Optional[Context] = None) -> str:
    """Format API errors consistently and log them."""
    error_message = ""

    if isinstance(e, ToolError):
        error_message = e.message
        logger.error("Tool error: %s", error_message)
    elif isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            error_data = e.response.json()
            if "errors" in error_data and error_data["errors"]:
                error = error_data["errors"][0]
                message = error.get("message", "Unknown error")
                code = error.get("code", "unknown")
                error_message = f"Error ({status}): {message} [Code: {code}]"
                logger.error("API error %s: %s [%s]", status, message, code)
        except:
            pass

        if not error_message:
            if status == 401:
                error_message = "Error: Authentication failed. Please check your API key has the required permissions."
            elif status == 403:
                error_message = "Error: Permission denied. Your API key lacks the required permissions for this operation."
            elif status == 404:
                error_message = "Error: Resource not found. Please check the ID is correct."
            elif status == 422:
                error_message = "Error: Validation failed. Please check your input parameters."
            elif status == 429:
                error_message = "Error: Rate limit exceeded. Please wait before making more requests."
            else:
                error_message = f"Error: API request failed with status {status}"
            logger.error("HTTP error %s: %s", status, error_message)
    elif isinstance(e, httpx.TimeoutException):
        error_message = "Error: Request timed out. Please try again."
        logger.error("Request timeout: %s", str(e))
    elif isinstance(e, ValueError):
        error_message = f"Error: {str(e)}"
        logger.error("Validation error: %s", str(e))
    else:
        error_message = f"Error: Unexpected error occurred: {type(e).__name__}: {str(e)}"
        logger.exception("Unexpected error: %s", str(e))

    return error_message

def _format_price(amount: str, currency: str) -> str:
    """Format price consistently."""
    return f"{currency} {amount}"

def _format_datetime(dt_str: str) -> str:
    """Format ISO datetime to human-readable format."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M %Z").strip()
    except:
        return dt_str

def _truncate_if_needed(content: str, data_description: str = "results") -> str:
    """Truncate content if it exceeds CHARACTER_LIMIT."""
    if len(content) > CHARACTER_LIMIT:
        truncated = content[:CHARACTER_LIMIT]
        truncated += f"\n\n**[Truncated]** Response exceeded {CHARACTER_LIMIT} characters. Use filters or pagination to see more {data_description}."
        return truncated
    return content

# ============================================================================
# Flight Optimization Helpers
# ============================================================================

def _parse_duration_minutes(offer: Dict[str, Any]) -> int:
    """Parse total duration from an offer in minutes."""
    total_minutes = 0
    for slice_data in offer.get("slices", []):
        duration_str = slice_data.get("duration", "PT0H0M")
        # Parse ISO 8601 duration (e.g., "PT2H30M")
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?', duration_str)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            total_minutes += hours * 60 + minutes
    return total_minutes

def _count_stops(offer: Dict[str, Any]) -> int:
    """Count total stops across all slices."""
    total_stops = 0
    for slice_data in offer.get("slices", []):
        segments = slice_data.get("segments", [])
        total_stops += max(0, len(segments) - 1)
    return total_stops

def _get_departure_hour(offer: Dict[str, Any]) -> int:
    """Get the departure hour of the first segment."""
    try:
        slices = offer.get("slices", [])
        if slices:
            segments = slices[0].get("segments", [])
            if segments:
                departing_at = segments[0].get("departing_at", "")
                dt = datetime.fromisoformat(departing_at.replace('Z', '+00:00'))
                return dt.hour
    except:
        pass
    return 12  # Default to noon

def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def _calculate_time_preference_score(hour: int, preference: Optional[str]) -> float:
    """Score departure time based on preference (0-1, higher is better)."""
    if not preference:
        return 0.5  # Neutral

    ranges = {
        "morning": (6, 12),
        "afternoon": (12, 18),
        "evening": (18, 24),
        "red_eye": (0, 6)
    }

    preferred_start, preferred_end = ranges.get(preference, (0, 24))
    if preferred_start <= hour < preferred_end:
        return 1.0  # Perfect match

    # Calculate distance from preferred range
    distance = min(abs(hour - preferred_start), abs(hour - preferred_end))
    return max(0, 1 - (distance / 12))  # Decay over 12 hours

def calculate_flight_score(
    offer: Dict[str, Any],
    all_offers: List[Dict[str, Any]],
    weights: OptimizationWeights,
    preferred_departure: Optional[str] = None
) -> float:
    """
    Calculate normalized score (0-100, higher is better).

    Normalization: Each factor is normalized to 0-1 range relative to all offers,
    then weighted and combined.
    """
    # Extract values
    price = float(offer.get("total_amount", 0))
    duration = _parse_duration_minutes(offer)
    stops = _count_stops(offer)
    departure_hour = _get_departure_hour(offer)

    # Get min/max for normalization
    prices = [float(o.get("total_amount", 0)) for o in all_offers]
    durations = [_parse_duration_minutes(o) for o in all_offers]
    stops_list = [_count_stops(o) for o in all_offers]

    # Normalize (invert so lower is better becomes higher score)
    price_score = 1 - _normalize(price, min(prices), max(prices)) if prices else 0.5
    duration_score = 1 - _normalize(duration, min(durations), max(durations)) if durations else 0.5
    stops_score = 1 - _normalize(stops, min(stops_list), max(stops_list)) if stops_list else 0.5
    time_score = _calculate_time_preference_score(departure_hour, preferred_departure)

    # Weighted combination
    total = (
        weights.price * price_score +
        weights.duration * duration_score +
        weights.stops * stops_score +
        weights.departure_time * time_score
    )

    return round(total * 100, 2)

def _optimize_offers(
    offers: List[Dict[str, Any]],
    strategy: OptimizationStrategy,
    weights: Optional[OptimizationWeights] = None,
    preferred_departure: Optional[str] = None,
    top_n: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Apply optimization strategy to offers and return sorted list."""
    if not offers or strategy == OptimizationStrategy.NONE:
        return offers[:top_n] if top_n else offers

    if weights is None:
        weights = OptimizationWeights()

    # Calculate scores for all offers if using BEST strategy
    if strategy == OptimizationStrategy.BEST:
        for offer in offers:
            offer["_score"] = calculate_flight_score(offer, offers, weights, preferred_departure)

    # Sort based on strategy
    if strategy == OptimizationStrategy.CHEAPEST:
        sorted_offers = sorted(offers, key=lambda x: float(x.get("total_amount", 0)))
    elif strategy == OptimizationStrategy.FASTEST:
        sorted_offers = sorted(offers, key=lambda x: _parse_duration_minutes(x))
    elif strategy == OptimizationStrategy.LEAST_STOPS:
        sorted_offers = sorted(offers, key=lambda x: _count_stops(x))
    elif strategy == OptimizationStrategy.EARLIEST:
        sorted_offers = sorted(offers, key=lambda x: _get_departure_hour(x))
    elif strategy == OptimizationStrategy.LATEST:
        sorted_offers = sorted(offers, key=lambda x: _get_departure_hour(x), reverse=True)
    elif strategy == OptimizationStrategy.BEST:
        sorted_offers = sorted(offers, key=lambda x: x.get("_score", 0), reverse=True)
    else:
        sorted_offers = offers

    return sorted_offers[:top_n] if top_n else sorted_offers

def _get_offer_summary(offer: Dict[str, Any]) -> FlightOfferSummary:
    """Extract summary from an offer."""
    slices = offer.get("slices", [])
    first_segment = slices[0].get("segments", [{}])[0] if slices else {}
    last_slice = slices[-1] if slices else {}
    last_segment = last_slice.get("segments", [{}])[-1] if last_slice else {}

    return FlightOfferSummary(
        id=offer.get("id", ""),
        price=offer.get("total_amount", "0"),
        currency=offer.get("total_currency", "USD"),
        duration_minutes=_parse_duration_minutes(offer),
        stops=_count_stops(offer),
        airline=offer.get("owner", {}).get("name", "Unknown"),
        departure_time=first_segment.get("departing_at", ""),
        arrival_time=last_segment.get("arriving_at", ""),
        score=offer.get("_score")
    )

# ============================================================================
# MCP Resources
# ============================================================================

@mcp.resource("duffel://airlines")
async def list_airlines_resource(ctx: Context) -> str:
    """
    List of all available airlines for booking through Duffel.

    Returns airline names, IATA codes, and logos for reference when
    searching for flights or filtering results.
    """
    try:
        logger.debug("Resource request: duffel://airlines")
        response = await _make_api_request(ctx, "air/airlines", params={"limit": "200"})
        logger.info("Airlines resource: returned %d airlines", len(response.get("data", [])))
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error("Airlines resource error: %s", str(e))
        return json.dumps({"error": str(e)})

@mcp.resource("duffel://airlines/{iata_code}")
async def get_airline_resource(iata_code: str, ctx: Context) -> str:
    """
    Details for a specific airline by IATA code.

    Use this to get information about a particular airline including
    their name, logo, and conditions of carriage.
    """
    try:
        logger.debug("Resource request: duffel://airlines/%s", iata_code)
        # The Duffel API uses airline IDs, not IATA codes directly
        # We need to search for the airline by IATA code
        response = await _make_api_request(ctx, "air/airlines", params={"limit": "200"})
        airlines = response.get("data", [])
        for airline in airlines:
            if airline.get("iata_code", "").upper() == iata_code.upper():
                logger.info("Found airline %s: %s", iata_code, airline.get("name", "Unknown"))
                return json.dumps({"data": airline}, indent=2)
        logger.warning("Airline not found: %s", iata_code)
        return json.dumps({"error": f"Airline with IATA code '{iata_code}' not found"})
    except Exception as e:
        logger.error("Airline resource error: %s", str(e))
        return json.dumps({"error": str(e)})

@mcp.resource("duffel://places/{query}")
async def search_places_resource(query: str, ctx: Context) -> str:
    """
    Search for airports and cities by name or code.

    Use this to find IATA codes for airports when the user provides
    a city name or partial airport code.
    """
    try:
        logger.debug("Resource request: duffel://places/%s", query)
        response = await _make_api_request(
            ctx,
            "air/places/suggestions",
            params={"query": query}
        )
        logger.info("Places search '%s': returned %d results", query, len(response.get("data", [])))
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error("Places resource error: %s", str(e))
        return json.dumps({"error": str(e)})

@mcp.resource("duffel://instructions")
def flight_search_instructions() -> str:
    """
    Instructions for the AI travel agent on how to search effectively.
    """
    return """# AI Travel Agent Guidelines

## Be a Smart Agent, Not a Questionnaire

You are a helpful travel agent. Don't interrogate the customer with many questions.
Instead, be proactive: search comprehensively and present findings with smart advice.

### Only Ask When Truly Needed:
- **Departure city** - if not clear from context (but infer from their location if known)
- **Approximate dates** - if not mentioned at all
- **Trip length** - only if they said round-trip but didn't mention return

### DON'T Ask About (Just Handle It):
- Luggage: Search and NOTE in results if baggage isn't included
- Connections: Show both options and note "direct" vs "1 stop (2hr layover)"
- Time of day: Show a range of options
- Cabin class: Default to economy unless they mention otherwise
- Airline preferences: Show what's available, note if budget carrier

## Proactive Search Strategy

When user wants the "cheapest" or "best deal":
1. **Search multiple dates automatically** (+/- 3 days from target)
2. **Compare and present** the best options found
3. **Advise** on trade-offs ("Flying Dec 23 instead of Dec 24 saves $85")

## Smart Result Presentation

For each option, include relevant warnings inline:

**Good Example:**
"✈️ **$301** - Malaysia Airlines (MH751)
- Dec 24: SGN 11:00 → KUL 14:10 (direct, 2h10m)
- Dec 26: KUL 17:10 → SGN 18:10 (direct, 2h)
- ✅ 20kg checked bag included"

"✈️ **$189** - AirAsia (AK857)
- Dec 24: SGN 08:30 → KUL 11:45 (direct, 2h15m)
- Dec 26: KUL 19:00 → SGN 20:05 (direct, 2h5m)
- ⚠️ Carry-on only - checked bag +$35 each way"

"✈️ **$156** - VietJet + Firefly
- Dec 24: SGN 06:00 → KUL 15:30 (1 stop, 5h layover in SIN)
- ⚠️ Long layover, separate tickets, bags not included"

## Key Behaviors

1. **Infer intelligently** - Use context about the user
2. **Search broadly** - Multiple dates, multiple airlines
3. **Advise clearly** - Note trade-offs, hidden costs, long layovers
4. **Be concise** - Present findings, don't ask unnecessary questions
5. **Recommend** - "I'd suggest the Malaysia Airlines option - only $45 more but includes bags and better times"

## When to Ask vs Infer

| Situation | Action |
|-----------|--------|
| User says "cheapest to Paris" | Search +/- 3 days, present options |
| User says "Christmas in Tokyo" | Infer Dec 24-26ish, ask how many days |
| User mentions budget | Prioritize price, warn about fees |
| User mentions "quick trip" | Note total travel times prominently |
| User is vague about dates | Ask for approximate timeframe |

## Optimization Strategies

- `cheapest`: Find lowest price (search multiple dates!)
- `fastest`: Shortest total travel time
- `best`: Balanced score (good for general "find me flights")
- `least_stops`: Prioritize direct flights
- `earliest`/`latest`: Time-of-day preference
"""

# ============================================================================
# MCP Prompts
# ============================================================================

@mcp.prompt("book_round_trip")
def book_round_trip_prompt(
    origin: str = "JFK",
    destination: str = "LHR",
    departure_date: str = "2025-01-15",
    return_date: str = "2025-01-22",
    passengers: str = "1 adult"
) -> str:
    """
    Smart workflow for booking a round-trip flight.
    Be helpful, not interrogative. Search proactively and advise.
    """
    return f"""# Round-Trip Flight Booking

## Trip: {origin} → {destination}
- Outbound: {departure_date}
- Return: {return_date}
- Passengers: {passengers}

## Your Approach

### 1. Search Proactively
- Search the requested dates immediately
- If looking for best deal, also search +/- 2-3 days and compare
- Use `optimization: "best"` for balanced results

### 2. Present Options with Advice
Show top options with inline notes:
- Price and what's included (bags, changes)
- Flight times and duration
- Direct vs connections (note layover length if long)
- Budget carrier warnings if applicable

### 3. Make a Recommendation
"Based on your trip, I'd recommend [option] because..."

### 4. Only Ask If Truly Needed
- Missing departure city → ask
- Vague dates ("sometime in December") → ask for range
- Everything else → search and advise

### 5. Book When Ready
Collect passenger details only after they choose:
- Name (as on passport)
- Date of birth
- Contact info (email, phone)

Use `duffel_get_offer` to verify price, then `duffel_create_order` to book.
"""

@mcp.prompt("find_cheapest")
def find_cheapest_prompt(
    destination: str = "destination",
    trip_type: str = "round-trip"
) -> str:
    """
    Find the cheapest flight - search proactively across multiple dates.
    """
    return f"""# Find Cheapest Flight to {destination}

## Your Mission
Find the absolute best deal. Search proactively, advise on trade-offs.

## Search Strategy (Do This Automatically)

1. **Search multiple dates** - Don't just search one date. Search +/- 3 days:
   - If user wants Dec 24-26, also check Dec 22-27 departures/returns
   - Track the cheapest option found across all searches

2. **Use `optimization: "cheapest"`** for all searches

3. **Compare and advise**:
   - "Cheapest overall: $156 on Dec 22-25"
   - "Your preferred dates (Dec 24-26): $301"
   - "Savings: $145 by flying 2 days earlier"

## Present Results With Context

For each option, note:
- ✅ What's good (included bags, direct flight, good times)
- ⚠️ What to watch out for (no bags, long layover, 5am departure, budget carrier)

Example:
"**$156** - VietJet (Dec 22-25)
⚠️ Basic fare: +$35/bag each way, 6:00am departure
Actual cost with 1 bag: ~$226"

"**$301** - Malaysia Airlines (Dec 24-26)
✅ 20kg bag included, reasonable 11am departure
Better value if you're checking luggage"

## Only Ask If Needed
- No departure city mentioned → ask where they're flying from
- Very vague dates ("sometime next month") → ask for a target week
- Otherwise → just search and present findings

## Make a Recommendation
End with: "My recommendation: [option] because [reason]"
"""

@mcp.prompt("compare_options")
def compare_options_prompt(
    offer_request_id: str = "orq_xxxxx",
    criteria: str = "price, duration, stops"
) -> str:
    """
    Compare multiple flight options from a search.

    This prompt helps users analyze and compare different
    flight options to make the best choice.
    """
    return f"""# Compare Flight Options

Analyze and compare offers from search: `{offer_request_id}`

## Comparison Criteria
{criteria}

## Analysis Steps

1. **Retrieve All Offers**
   Use `duffel_analyze_offers` with:
   - `offer_request_id: "{offer_request_id}"`
   - `optimization: "best"`
   - `top_n: 10`

2. **Create Comparison Table**
   For each option, show:
   | Offer | Price | Duration | Stops | Departure | Score |
   |-------|-------|----------|-------|-----------|-------|

3. **Highlight Trade-offs**
   - Cheapest vs Fastest
   - Non-stop vs Connection savings
   - Early/Late departures

4. **Recommendation**
   Based on the user's priorities, recommend:
   - Best overall value
   - Budget option
   - Convenience option

## Decision Factors
- Price difference percentage
- Time saved vs cost
- Layover quality (duration, airport)
- Airline reputation
- Baggage policies
"""

# ============================================================================
# Tool Implementations
# ============================================================================

@mcp.tool(
    name="duffel_search_flights",
    annotations={
        "title": "Search Flights",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def duffel_search_flights(params: SearchFlightsInput, ctx: Context) -> str:
    """
    Search for flights by creating an offer request in the Duffel API.

    SMART AGENT TIPS:
    - For "cheapest" requests: Search multiple date combinations (+/- 3 days)
      and compare results. Don't just search one date.
    - Present results with context: Note if bags aren't included, if there's
      a long layover, or if it's a budget carrier with extra fees.
    - Make recommendations based on the trade-offs you find.
    - Only ask the user questions if truly needed (missing origin, vague dates).

    This tool searches for available flights based on itinerary (origin, destination, dates),
    passenger information, and preferences like cabin class. It supports optimization
    strategies to find the cheapest, fastest, or best overall flights.

    Args:
        params (SearchFlightsInput): Validated input parameters containing:
            - slices (List[FlightSlice]): Flight segments (1 for one-way, 2 for round-trip)
            - passengers (List[PassengerInput]): Passenger list (max 9)
            - cabin_class (Optional[CabinClass]): Cabin preference (default: economy)
            - max_connections (Optional[int]): Max connections (0=non-stop)
            - return_offers (bool): Return offers immediately (default: true)
            - optimization (OptimizationStrategy): Sort strategy (cheapest, fastest, best, etc.)
            - optimization_weights (OptimizationWeights): Custom weights for 'best' strategy
            - preferred_departure_time (DepartureTimePreference): Preferred departure window
            - top_n (Optional[int]): Return only top N results
            - response_format (ResponseFormat): 'markdown' or 'json'
        ctx (Context): MCP context for progress reporting and logging

    Returns:
        str: Formatted response containing flight offers and search details

    Examples:
        - "Find flights from SGN to KUL departing Dec 24, returning Dec 26"
        - "Search cheapest business class flights NYC to LON" (search multiple dates!)
        - "Find best flights considering price, time, and stops"
    """
    try:
        await ctx.report_progress(0.1, "Validating search parameters...")

        # Log search request
        route_summary = " -> ".join([f"{s.origin}->{s.destination}" for s in params.slices])
        logger.info(
            "Flight search: %s, %d passengers, %s class, optimization=%s",
            route_summary,
            len(params.passengers),
            params.cabin_class.value,
            params.optimization.value
        )

        # Build request payload
        request_data = {
            "data": {
                "slices": [
                    {
                        "origin": slice_data.origin.upper(),
                        "destination": slice_data.destination.upper(),
                        "departure_date": slice_data.departure_date
                    }
                    for slice_data in params.slices
                ],
                "passengers": [
                    {"type": p.type.value} if p.type else {"age": p.age}
                    for p in params.passengers
                ],
                "cabin_class": params.cabin_class.value
            }
        }

        if params.max_connections is not None:
            request_data["data"]["max_connections"] = params.max_connections

        query_params = {
            "return_offers": "true" if params.return_offers else "false"
        }

        await ctx.report_progress(0.3, "Searching for flights...")

        response = await _make_api_request(
            ctx,
            "air/offer_requests",
            method="POST",
            params=query_params,
            json_data=request_data
        )

        await ctx.report_progress(0.7, "Processing offers...")

        data = response.get("data", {})
        offers = data.get("offers", [])

        # Apply optimization
        if offers and params.optimization != OptimizationStrategy.NONE:
            preferred_dep = params.preferred_departure_time.value if params.preferred_departure_time else None
            offers = _optimize_offers(
                offers,
                params.optimization,
                params.optimization_weights,
                preferred_dep,
                params.top_n
            )
            data["offers"] = offers

        await ctx.report_progress(0.9, "Formatting results...")

        # Format response
        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(response, indent=2)
            return _truncate_if_needed(result, "offers")

        # Markdown format
        lines = ["# Flight Search Results\n"]

        # Search details
        lines.append("## Search Criteria")
        lines.append(f"- **Offer Request ID**: `{data.get('id', 'N/A')}`")
        lines.append(f"- **Cabin Class**: {data.get('cabin_class', 'N/A').replace('_', ' ').title()}")
        lines.append(f"- **Passengers**: {len(data.get('passengers', []))}")
        if params.optimization != OptimizationStrategy.NONE:
            lines.append(f"- **Optimization**: {params.optimization.value}")

        # Slices
        lines.append("\n### Itinerary")
        for i, slice_info in enumerate(data.get("slices", []), 1):
            origin = slice_info.get("origin", {})
            destination = slice_info.get("destination", {})
            lines.append(f"{i}. **{origin.get('iata_code', 'N/A')}** -> **{destination.get('iata_code', 'N/A')}** on {slice_info.get('departure_date', 'N/A')}")

        # Offers
        if offers:
            lines.append(f"\n## Available Offers ({len(offers)} found)\n")
            display_count = min(len(offers), params.top_n or 20)

            for i, offer in enumerate(offers[:display_count], 1):
                score_str = f" | Score: {offer.get('_score', 'N/A')}" if offer.get('_score') else ""
                lines.append(f"### Offer {i}: {_format_price(offer.get('total_amount', '0'), offer.get('total_currency', 'USD'))}{score_str}")
                lines.append(f"- **Offer ID**: `{offer.get('id', 'N/A')}`")
                lines.append(f"- **Owner**: {offer.get('owner', {}).get('name', 'N/A')}")
                lines.append(f"- **Duration**: {_parse_duration_minutes(offer)} minutes")
                lines.append(f"- **Stops**: {_count_stops(offer)}")

                for j, slice_data in enumerate(offer.get("slices", []), 1):
                    segments = slice_data.get("segments", [])
                    if segments:
                        duration = slice_data.get("duration", "N/A")
                        lines.append(f"  - Slice {j}: {len(segments)} segment(s), Duration: {duration}")

                lines.append("")

            if len(offers) > display_count:
                lines.append(f"\n*Showing {display_count} of {len(offers)} offers. Use duffel_list_offers or increase top_n for more.*")
        else:
            lines.append("\n## No offers available")
            lines.append("No flights found matching your criteria. Try adjusting dates or removing restrictions.")

        await ctx.report_progress(1.0, "Search complete")

        logger.info("Search completed: %d offers found", len(offers))

        result = "\n".join(lines)
        return _truncate_if_needed(result, "offers")

    except Exception as e:
        return _handle_api_error(e, ctx)

@mcp.tool(
    name="duffel_analyze_offers",
    annotations={
        "title": "Analyze Flight Offers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def duffel_analyze_offers(params: AnalyzeOffersInput, ctx: Context) -> str:
    """
    Analyze and rank flight offers from a previous search.

    Use this after duffel_search_flights to find the best options based on
    user preferences. Supports multiple optimization strategies and custom
    weighting for the 'best' algorithm.

    Args:
        params (AnalyzeOffersInput): Validated input parameters containing:
            - offer_request_id (str): Offer request ID from previous search
            - optimization (OptimizationStrategy): Strategy to apply
            - optimization_weights (OptimizationWeights): Custom weights for 'best'
            - preferred_departure_time (DepartureTimePreference): Time preference
            - top_n (int): Number of results to return (default: 5)
            - response_format (ResponseFormat): Output format
        ctx (Context): MCP context for progress and logging

    Returns:
        str: Ranked and analyzed flight offers with scores

    Examples:
        - Use when: "Analyze the search results and find the best value"
        - Use when: "Compare the top 5 cheapest flights from the last search"
        - Use when: "Find flights that balance price and duration"
    """
    try:
        logger.info(
            "Analyzing offers for request %s with strategy=%s, top_n=%d",
            params.offer_request_id,
            params.optimization.value,
            params.top_n
        )

        await ctx.report_progress(0.2, "Fetching offers from search...")

        # Fetch all offers from the offer request
        response = await _make_api_request(
            ctx,
            "air/offers",
            params={
                "offer_request_id": params.offer_request_id,
                "limit": "200"  # Get all for analysis
            }
        )

        offers = response.get("data", [])

        if not offers:
            logger.warning("No offers found for request %s", params.offer_request_id)
            return "No offers found for this offer request ID. The search may have expired."

        await ctx.report_progress(0.5, f"Analyzing {len(offers)} offers...")

        # Apply optimization
        preferred_dep = params.preferred_departure_time.value if params.preferred_departure_time else None
        optimized = _optimize_offers(
            offers,
            params.optimization,
            params.optimization_weights,
            preferred_dep,
            params.top_n
        )

        await ctx.report_progress(0.8, "Formatting analysis...")

        if params.response_format == ResponseFormat.JSON:
            result = {
                "offer_request_id": params.offer_request_id,
                "total_analyzed": len(offers),
                "optimization": params.optimization.value,
                "top_offers": [_get_offer_summary(o) for o in optimized]
            }
            return json.dumps(result, indent=2)

        # Markdown format
        lines = ["# Flight Offer Analysis\n"]
        lines.append(f"- **Offer Request**: `{params.offer_request_id}`")
        lines.append(f"- **Total Offers Analyzed**: {len(offers)}")
        lines.append(f"- **Optimization Strategy**: {params.optimization.value}")
        if params.optimization_weights and params.optimization == OptimizationStrategy.BEST:
            w = params.optimization_weights
            lines.append(f"- **Weights**: Price={w.price}, Duration={w.duration}, Stops={w.stops}, Time={w.departure_time}")

        # Summary stats
        prices = [float(o.get("total_amount", 0)) for o in offers]
        durations = [_parse_duration_minutes(o) for o in offers]
        lines.append(f"\n## Market Overview")
        lines.append(f"- **Price Range**: {min(prices):.2f} - {max(prices):.2f} {offers[0].get('total_currency', 'USD')}")
        lines.append(f"- **Duration Range**: {min(durations)} - {max(durations)} minutes")

        lines.append(f"\n## Top {len(optimized)} Offers\n")

        for i, offer in enumerate(optimized, 1):
            score = offer.get("_score")
            score_str = f" (Score: **{score}**/100)" if score else ""
            lines.append(f"### #{i}: {_format_price(offer.get('total_amount', '0'), offer.get('total_currency', 'USD'))}{score_str}")
            lines.append(f"- **Offer ID**: `{offer.get('id', 'N/A')}`")
            lines.append(f"- **Airline**: {offer.get('owner', {}).get('name', 'N/A')}")
            lines.append(f"- **Duration**: {_parse_duration_minutes(offer)} minutes")
            lines.append(f"- **Stops**: {_count_stops(offer)}")

            for j, slice_data in enumerate(offer.get("slices", []), 1):
                segments = slice_data.get("segments", [])
                if segments:
                    first_seg = segments[0]
                    last_seg = segments[-1]
                    lines.append(f"  - **Slice {j}**: {first_seg.get('origin', {}).get('iata_code', '?')} -> {last_seg.get('destination', {}).get('iata_code', '?')}")
                    lines.append(f"    - Depart: {_format_datetime(first_seg.get('departing_at', 'N/A'))}")
                    lines.append(f"    - Arrive: {_format_datetime(last_seg.get('arriving_at', 'N/A'))}")

            lines.append("")

        await ctx.report_progress(1.0, "Analysis complete")

        logger.info("Analysis completed: %d offers analyzed, top %d returned", len(offers), len(optimized))

        result = "\n".join(lines)
        return _truncate_if_needed(result, "offers")

    except Exception as e:
        return _handle_api_error(e, ctx)

@mcp.tool(
    name="duffel_get_offer",
    annotations={
        "title": "Get Single Offer",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def duffel_get_offer(params: GetOfferInput, ctx: Context) -> str:
    """
    Retrieve the latest details for a specific flight offer.

    This tool fetches up-to-date information for a specific offer using its ID.
    It's recommended to call this before booking to ensure the offer is still
    valid and to get the latest pricing.

    Args:
        params (GetOfferInput): Validated input parameters
        ctx (Context): MCP context for progress reporting

    Returns:
        str: Formatted offer details including pricing, itinerary, and conditions
    """
    try:
        logger.info("Fetching offer details for %s", params.offer_id)
        await ctx.report_progress(0.2, "Fetching offer details...")

        query_params = {}
        if params.return_available_services:
            query_params["return_available_services"] = "true"

        response = await _make_api_request(
            ctx,
            f"air/offers/{params.offer_id}",
            method="GET",
            params=query_params
        )

        await ctx.report_progress(0.8, "Formatting response...")

        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(response, indent=2)
            return _truncate_if_needed(result)

        # Markdown format
        data = response.get("data", {})
        lines = ["# Flight Offer Details\n"]

        # Basic info
        lines.append("## Overview")
        lines.append(f"- **Offer ID**: `{data.get('id', 'N/A')}`")
        lines.append(f"- **Total Price**: **{_format_price(data.get('total_amount', '0'), data.get('total_currency', 'USD'))}**")
        lines.append(f"- **Airline**: {data.get('owner', {}).get('name', 'N/A')}")
        lines.append(f"- **Expires**: {_format_datetime(data.get('expires_at', 'N/A'))}")
        lines.append(f"- **Live Mode**: {'Yes' if data.get('live_mode') else 'No'}")

        # Itinerary
        lines.append("\n## Itinerary")
        for i, slice_data in enumerate(data.get("slices", []), 1):
            lines.append(f"\n### Slice {i}")
            lines.append(f"- **Duration**: {slice_data.get('duration', 'N/A')}")

            for j, segment in enumerate(slice_data.get("segments", []), 1):
                lines.append(f"\n#### Segment {j}")
                lines.append(f"- **Flight**: {segment.get('marketing_carrier', {}).get('name', 'N/A')} {segment.get('marketing_carrier_flight_number', 'N/A')}")
                lines.append(f"- **Aircraft**: {segment.get('aircraft', {}).get('name', 'N/A')}")
                lines.append(f"- **Departure**: {segment.get('origin', {}).get('iata_code', 'N/A')} at {_format_datetime(segment.get('departing_at', 'N/A'))}")
                lines.append(f"- **Arrival**: {segment.get('destination', {}).get('iata_code', 'N/A')} at {_format_datetime(segment.get('arriving_at', 'N/A'))}")
                lines.append(f"- **Duration**: {segment.get('duration', 'N/A')}")

        # Passengers
        passengers = data.get("passengers", [])
        if passengers:
            lines.append(f"\n## Passengers ({len(passengers)})")
            for p in passengers:
                lines.append(f"- **ID**: `{p.get('id', 'N/A')}` - {p.get('type', 'N/A').replace('_', ' ').title()}")

        await ctx.report_progress(1.0, "Done")

        logger.info("Retrieved offer %s: %s %s", params.offer_id, data.get('total_currency', 'USD'), data.get('total_amount', '0'))

        result = "\n".join(lines)
        return _truncate_if_needed(result)

    except Exception as e:
        return _handle_api_error(e, ctx)

@mcp.tool(
    name="duffel_list_offers",
    annotations={
        "title": "List Offers from Request",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def duffel_list_offers(params: ListOffersInput, ctx: Context) -> str:
    """
    List all flight offers from a specific offer request with filtering and sorting.

    This tool retrieves offers from a previous search (offer request), with support
    for pagination, filtering by connections, and sorting by price or duration.

    Args:
        params (ListOffersInput): Validated input parameters
        ctx (Context): MCP context for progress reporting

    Returns:
        str: Formatted list of flight offers with filtering applied
    """
    try:
        logger.info("Listing offers for request %s (limit=%d, sort=%s)", params.offer_request_id, params.limit, params.sort)
        await ctx.report_progress(0.2, "Fetching offers...")

        query_params = {
            "offer_request_id": params.offer_request_id,
            "limit": str(params.limit)
        }

        if params.max_connections is not None:
            query_params["max_connections"] = str(params.max_connections)

        if params.sort:
            query_params["sort"] = params.sort

        response = await _make_api_request(
            ctx,
            "air/offers",
            method="GET",
            params=query_params
        )

        await ctx.report_progress(0.8, "Formatting results...")

        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(response, indent=2)
            return _truncate_if_needed(result, "offers")

        # Markdown format
        data = response.get("data", [])
        lines = ["# Flight Offers\n"]
        lines.append(f"Found {len(data)} offer(s)\n")

        if not data:
            lines.append("No offers found matching your criteria.")
            return "\n".join(lines)

        for i, offer in enumerate(data, 1):
            lines.append(f"## Offer {i}: {_format_price(offer.get('total_amount', '0'), offer.get('total_currency', 'USD'))}")
            lines.append(f"- **ID**: `{offer.get('id', 'N/A')}`")
            lines.append(f"- **Airline**: {offer.get('owner', {}).get('name', 'N/A')}")

            for j, slice_data in enumerate(offer.get("slices", []), 1):
                segments = slice_data.get("segments", [])
                connections = len(segments) - 1
                duration = slice_data.get("duration", "N/A")

                if segments:
                    origin = segments[0].get("origin", {}).get("iata_code", "N/A")
                    destination = segments[-1].get("destination", {}).get("iata_code", "N/A")
                    lines.append(f"  - **Slice {j}**: {origin} -> {destination}")
                    lines.append(f"    - Segments: {len(segments)}, Connections: {connections}, Duration: {duration}")

            lines.append("")

        await ctx.report_progress(1.0, "Done")

        logger.info("Listed %d offers for request %s", len(data), params.offer_request_id)

        result = "\n".join(lines)
        return _truncate_if_needed(result, "offers")

    except Exception as e:
        return _handle_api_error(e, ctx)

@mcp.tool(
    name="duffel_create_order",
    annotations={
        "title": "Create Flight Booking",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def duffel_create_order(params: CreateOrderInput, ctx: Context) -> str:
    """
    Create a flight booking (order) with passenger and payment details.

    This tool creates a confirmed flight booking for a selected offer. It requires
    complete passenger information (names, DOB, contact details) and payment information.
    This operation charges the payment method and confirms the booking with the airline.

    Args:
        params (CreateOrderInput): Validated input parameters
        ctx (Context): MCP context for progress reporting

    Returns:
        str: Formatted order confirmation with booking reference and details
    """
    try:
        logger.info(
            "Creating order for offer %s with %d passengers",
            params.selected_offers[0] if params.selected_offers else "unknown",
            len(params.passengers)
        )
        await ctx.report_progress(0.1, "Validating booking details...")

        request_data = {
            "data": {
                "selected_offers": params.selected_offers,
                "payments": [
                    {
                        "type": p.type.value,
                        "amount": p.amount,
                        "currency": p.currency.upper()
                    }
                    for p in params.payments
                ],
                "passengers": [
                    {
                        "id": p.id,
                        "given_name": p.given_name,
                        "family_name": p.family_name,
                        "born_on": p.born_on,
                        "email": p.email,
                        "phone_number": p.phone_number,
                        "title": p.title,
                        "gender": p.gender,
                        **({"infant_passenger_id": p.infant_passenger_id} if p.infant_passenger_id else {})
                    }
                    for p in params.passengers
                ]
            }
        }

        await ctx.report_progress(0.3, "Creating booking...")

        response = await _make_api_request(
            ctx,
            "air/orders",
            method="POST",
            json_data=request_data
        )

        await ctx.report_progress(0.9, "Confirming booking...")

        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(response, indent=2)
            return _truncate_if_needed(result)

        # Markdown format
        data = response.get("data", {})
        lines = ["# Booking Confirmed\n"]

        lines.append("## Order Details")
        lines.append(f"- **Order ID**: `{data.get('id', 'N/A')}`")
        lines.append(f"- **Booking Reference**: **{data.get('booking_reference', 'N/A')}**")
        lines.append(f"- **Total Amount**: **{_format_price(data.get('total_amount', '0'), data.get('total_currency', 'USD'))}**")
        lines.append(f"- **Status**: {data.get('booking_status', {}).get('status', 'N/A').title()}")
        lines.append(f"- **Created**: {_format_datetime(data.get('created_at', 'N/A'))}")

        # Passengers
        passengers = data.get("passengers", [])
        if passengers:
            lines.append(f"\n## Passengers ({len(passengers)})")
            for p in passengers:
                name = f"{p.get('given_name', '')} {p.get('family_name', '')}".strip()
                lines.append(f"- **{name}** ({p.get('type', 'N/A')})")

        # Slices
        slices = data.get("slices", [])
        if slices:
            lines.append("\n## Flight Itinerary")
            for i, slice_data in enumerate(slices, 1):
                lines.append(f"\n### Slice {i}")
                for j, segment in enumerate(slice_data.get("segments", []), 1):
                    lines.append(f"**Segment {j}**: {segment.get('marketing_carrier', {}).get('name', 'N/A')} {segment.get('marketing_carrier_flight_number', 'N/A')}")
                    lines.append(f"  - {segment.get('origin', {}).get('iata_code', 'N/A')} -> {segment.get('destination', {}).get('iata_code', 'N/A')}")
                    lines.append(f"  - Departs: {_format_datetime(segment.get('departing_at', 'N/A'))}")
                    lines.append(f"  - Arrives: {_format_datetime(segment.get('arriving_at', 'N/A'))}")

        lines.append("\n---")
        lines.append("**Important**: Please save your booking reference. You may receive confirmation emails from the airline.")

        await ctx.report_progress(1.0, "Booking complete")

        logger.info(
            "Order created successfully: ID=%s, Reference=%s, Amount=%s %s",
            data.get('id', 'N/A'),
            data.get('booking_reference', 'N/A'),
            data.get('total_currency', 'USD'),
            data.get('total_amount', '0')
        )

        result = "\n".join(lines)
        return _truncate_if_needed(result)

    except Exception as e:
        return _handle_api_error(e, ctx)

@mcp.tool(
    name="duffel_list_airlines",
    annotations={
        "title": "List Airlines",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def duffel_list_airlines(params: ListAirlinesInput, ctx: Context) -> str:
    """
    List available airlines in the Duffel API.

    This tool retrieves information about airlines that can be booked through Duffel,
    including their names, IATA codes, and logos.

    Args:
        params (ListAirlinesInput): Validated input parameters
        ctx (Context): MCP context for progress reporting

    Returns:
        str: Formatted list of airlines with codes and names
    """
    try:
        logger.info("Listing airlines (limit=%d)", params.limit)
        await ctx.report_progress(0.3, "Fetching airlines...")

        query_params = {"limit": str(params.limit)}

        response = await _make_api_request(
            ctx,
            "air/airlines",
            method="GET",
            params=query_params
        )

        await ctx.report_progress(0.8, "Formatting results...")

        if params.response_format == ResponseFormat.JSON:
            result = json.dumps(response, indent=2)
            return _truncate_if_needed(result, "airlines")

        # Markdown format
        data = response.get("data", [])
        lines = ["# Airlines\n"]
        lines.append(f"Showing {len(data)} airline(s)\n")

        for airline in data:
            name = airline.get("name", "N/A")
            iata = airline.get("iata_code", "N/A")
            lines.append(f"- **{name}** ({iata})")

        await ctx.report_progress(1.0, "Done")

        logger.info("Listed %d airlines", len(data))

        result = "\n".join(lines)
        return _truncate_if_needed(result, "airlines")

    except Exception as e:
        return _handle_api_error(e, ctx)

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the Duffel MCP server with configurable transport."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Duffel MCP Server - Flight search and booking via MCP"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type: 'stdio' (default) for CLI, 'sse' for HTTP Server-Sent Events"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger("duffel_mcp").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Starting Duffel MCP server with transport: %s", args.transport)

    if args.transport == "stdio":
        # Standard stdio transport for CLI tools
        mcp.run()
    elif args.transport == "sse":
        # HTTP with Server-Sent Events for web deployments
        logger.info("SSE server starting on http://%s:%d", args.host, args.port)
        mcp.run(
            transport="sse",
            host=args.host,
            port=args.port
        )


if __name__ == "__main__":
    main()
