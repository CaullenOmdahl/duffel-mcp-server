#!/usr/bin/env python3
"""
Test script to verify Duffel API key permissions.

This script tests the API key against all required endpoints to ensure
it has the necessary permissions before using the MCP server.
"""

import asyncio
import httpx
import json
import os
from datetime import datetime, timedelta

API_KEY = os.getenv("DUFFEL_API_KEY_LIVE", "")
API_BASE_URL = "https://api.duffel.com"
API_VERSION = "v2"

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_result(test_name: str, success: bool, message: str):
    """Print a formatted test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n{status} - {test_name}")
    print(f"   {message}")

async def test_endpoint(name: str, method: str, endpoint: str,
                       json_data: dict = None, params: dict = None):
    """Test a single API endpoint."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Duffel-Version": API_VERSION,
        "Accept": "application/json",
        "Accept-Encoding": "gzip"
    }

    if json_data:
        headers["Content-Type"] = "application/json"

    url = f"{API_BASE_URL}/{endpoint}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_data,
                params=params
            )

            if response.status_code in [200, 201]:
                data = response.json()
                return True, f"Status {response.status_code} - Success!", data
            elif response.status_code == 401:
                return False, "Authentication failed - Invalid API key", None
            elif response.status_code == 403:
                try:
                    error_data = response.json()
                    if "errors" in error_data and error_data["errors"]:
                        error = error_data["errors"][0]
                        message = error.get("message", "Permission denied")
                        return False, f"Permission denied - {message}", None
                except:
                    return False, "Permission denied - API key lacks required permissions", None
            elif response.status_code == 422:
                try:
                    error_data = response.json()
                    return False, f"Validation error - {json.dumps(error_data, indent=2)}", None
                except:
                    return False, "Validation error", None
            else:
                return False, f"Unexpected status code: {response.status_code}", None

    except httpx.TimeoutException:
        return False, "Request timed out", None
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {str(e)}", None

async def main():
    """Run all API key permission tests."""
    print_header("DUFFEL API KEY PERMISSION TEST")

    if not API_KEY:
        print("\n❌ ERROR: DUFFEL_API_KEY_LIVE environment variable is not set!")
        print("\nTo run this test:")
        print("  export DUFFEL_API_KEY_LIVE='your_api_key_here'")
        print("  python test_api_key.py")
        return

    print(f"\nTesting API Key: {API_KEY[:20]}...")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Version: {API_VERSION}")

    all_passed = True
    test_results = {}

    # Test 1: List Airlines (Read-only, should work with any valid key)
    print_header("TEST 1: List Airlines (air.airlines.read)")
    success, message, data = await test_endpoint(
        "List Airlines",
        "GET",
        "air/airlines",
        params={"limit": "5"}
    )
    print_result("GET /air/airlines", success, message)
    test_results["airlines_read"] = success
    all_passed = all_passed and success

    if success and data:
        airlines = data.get("data", [])
        print(f"   Found {len(airlines)} airlines:")
        for airline in airlines[:3]:
            print(f"     - {airline.get('name')} ({airline.get('iata_code', 'N/A')})")

    # Test 2: Create Offer Request (Requires write permission)
    print_header("TEST 2: Create Offer Request (air.offer_requests.create)")

    # Use dates 30 days in the future
    departure_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    return_date = (datetime.now() + timedelta(days=32)).strftime("%Y-%m-%d")

    offer_request_data = {
        "data": {
            "slices": [
                {
                    "origin": "LHR",
                    "destination": "JFK",
                    "departure_date": departure_date
                },
                {
                    "origin": "JFK",
                    "destination": "LHR",
                    "departure_date": return_date
                }
            ],
            "passengers": [
                {"type": "adult"}
            ],
            "cabin_class": "economy"
        }
    }

    success, message, data = await test_endpoint(
        "Create Offer Request",
        "POST",
        "air/offer_requests",
        json_data=offer_request_data,
        params={"return_offers": "true"}
    )
    print_result("POST /air/offer_requests", success, message)
    test_results["offer_requests_create"] = success
    all_passed = all_passed and success

    offer_request_id = None
    offer_id = None

    if success and data:
        result_data = data.get("data", {})
        offer_request_id = result_data.get("id")
        offers = result_data.get("offers", [])

        print(f"   Offer Request ID: {offer_request_id}")
        print(f"   Found {len(offers)} offers")

        if offers:
            offer_id = offers[0].get("id")
            print(f"   First Offer ID: {offer_id}")
            print(f"   Price: {offers[0].get('total_currency')} {offers[0].get('total_amount')}")

    # Test 3: Get Single Offer (Requires read permission)
    print_header("TEST 3: Get Single Offer (air.offers.read)")

    if offer_id:
        success, message, data = await test_endpoint(
            "Get Offer",
            "GET",
            f"air/offers/{offer_id}"
        )
        print_result("GET /air/offers/{id}", success, message)
        test_results["offers_read"] = success
        all_passed = all_passed and success

        if success and data:
            result_data = data.get("data", {})
            print(f"   Offer ID: {result_data.get('id')}")
            print(f"   Price: {result_data.get('total_currency')} {result_data.get('total_amount')}")
            print(f"   Expires: {result_data.get('expires_at')}")
    else:
        print_result("GET /air/offers/{id}", False, "Skipped - no offer ID available from previous test")
        test_results["offers_read"] = False
        all_passed = False

    # Test 4: List Offers (Requires read permission)
    print_header("TEST 4: List Offers (air.offers.read)")

    if offer_request_id:
        success, message, data = await test_endpoint(
            "List Offers",
            "GET",
            "air/offers",
            params={"offer_request_id": offer_request_id, "limit": "5"}
        )
        print_result("GET /air/offers", success, message)
        test_results["offers_list"] = success
        # Don't fail overall if this doesn't work, as it's similar to get single offer

        if success and data:
            offers = data.get("data", [])
            print(f"   Found {len(offers)} offers")
    else:
        print_result("GET /air/offers", False, "Skipped - no offer request ID available")
        test_results["offers_list"] = False

    # Test 5: Create Order (Note: We won't actually create an order, just check permission error)
    print_header("TEST 5: Create Order Permission Check (air.orders.create)")
    print("   ⚠️  Note: Not actually creating an order, just checking permissions")
    print("   This test will intentionally fail validation to check permission level")

    # Send invalid data to trigger either permission error (403) or validation error (422)
    # If we get 422, it means we have permission but data is invalid (good!)
    # If we get 403, it means we don't have permission (bad!)
    order_data = {
        "data": {
            "selected_offers": ["off_invalid"],
            "passengers": [],
            "payments": []
        }
    }

    success, message, data = await test_endpoint(
        "Create Order (Permission Check)",
        "POST",
        "air/orders",
        json_data=order_data
    )

    # For this test, 422 (validation error) means we have permission
    # 403 means we don't have permission
    if "422" in message or "Validation" in message:
        print_result("POST /air/orders", True, "Has permission (got validation error as expected)")
        test_results["orders_create"] = True
    elif "403" in message or "Permission denied" in message:
        print_result("POST /air/orders", False, message)
        test_results["orders_create"] = False
        all_passed = False
    else:
        print_result("POST /air/orders", False, message)
        test_results["orders_create"] = False
        all_passed = False

    # Summary
    print_header("SUMMARY")

    print("\nPermission Status:")
    print(f"  ✓ air.airlines.read:          {'✅ YES' if test_results.get('airlines_read') else '❌ NO'}")
    print(f"  ✓ air.offer_requests.create:  {'✅ YES' if test_results.get('offer_requests_create') else '❌ NO'}")
    print(f"  ✓ air.offers.read:             {'✅ YES' if test_results.get('offers_read') else '❌ NO'}")
    print(f"  ✓ air.orders.create:           {'✅ YES' if test_results.get('orders_create') else '❌ NO'}")

    print("\n" + "=" * 70)
    if all_passed:
        print("  ✅ ALL TESTS PASSED - API key has all required permissions!")
        print("  🚀 Ready to use the Duffel MCP server")
    else:
        print("  ⚠️  SOME TESTS FAILED - API key is missing permissions")
        print("  📝 Generate a new API key with the required permissions:")
        print("     https://app.duffel.com/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
