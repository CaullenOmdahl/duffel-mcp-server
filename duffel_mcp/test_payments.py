#!/usr/bin/env python3
"""
Test script to verify Duffel Payments API access.

This script specifically tests if Payments is enabled on your Duffel account.
"""

import asyncio
import httpx
import json
import os

API_KEY = os.getenv("DUFFEL_API_KEY_LIVE", "")
API_BASE_URL = "https://api.duffel.com"
API_VERSION = "v2"


async def test_payments():
    """Test Duffel Payments API access."""
    print("\n" + "=" * 70)
    print("  DUFFEL PAYMENTS API TEST")
    print("=" * 70)

    if not API_KEY:
        print("\n❌ ERROR: DUFFEL_API_KEY_LIVE environment variable is not set!")
        print("\nTo run this test:")
        print("  export DUFFEL_API_KEY_LIVE='your_api_key_here'")
        print("  python test_payments.py")
        return

    print(f"\nAPI Key: {API_KEY[:20]}...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Duffel-Version": API_VERSION,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Test creating a payment intent with minimal amount
    payment_data = {
        "data": {
            "amount": "1.00",
            "currency": "USD"
        }
    }

    print("\n🔍 Testing POST /payments/payment_intents...")
    print(f"   Payload: {json.dumps(payment_data)}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/payments/payment_intents",
                headers=headers,
                json=payment_data
            )

            print(f"\n📡 Response Status: {response.status_code}")

            try:
                data = response.json()
                print(f"📦 Response Body:\n{json.dumps(data, indent=2)}")
            except:
                print(f"📦 Response Body: {response.text}")

            if response.status_code == 201:
                print("\n✅ SUCCESS! Duffel Payments is enabled on your account.")
                payment_intent = data.get("data", {})
                print(f"   Payment Intent ID: {payment_intent.get('id')}")
                print(f"   Client Token: {payment_intent.get('client_token', 'N/A')[:30]}...")

            elif response.status_code == 422:
                # Check if it's a "not available" error
                errors = data.get("errors", [])
                for error in errors:
                    message = error.get("message", "")
                    if "not available" in message.lower():
                        print("\n❌ FAILED: Duffel Payments is NOT enabled on your account.")
                        print(f"   Error: {message}")
                        print("\n📋 NEXT STEPS:")
                        print("   1. Log into https://app.duffel.com")
                        print("   2. Go to Settings → Payments")
                        print("   3. Enable Duffel Payments or contact help@duffel.com")
                    else:
                        print(f"\n⚠️ Validation Error: {message}")

            elif response.status_code == 403:
                print("\n❌ FAILED: Permission denied.")
                print("   Your API key does not have payments.payment_intents.create permission.")
                print("\n📋 NEXT STEPS:")
                print("   1. Generate a new API key at https://app.duffel.com")
                print("   2. Ensure 'Payments' permissions are enabled")

            elif response.status_code == 401:
                print("\n❌ FAILED: Authentication failed.")
                print("   Check that your DUFFEL_API_KEY_LIVE is correct.")

            else:
                print(f"\n⚠️ Unexpected status code: {response.status_code}")

    except httpx.TimeoutException:
        print("\n❌ Request timed out")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")

    # Also check if there's a way to list payment methods or check status
    print("\n" + "-" * 70)
    print("Additional checks:")

    # Try to get organization info
    print("\n🔍 Checking organization/account info...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/identity/user",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                user_data = data.get("data", {})
                print(f"   User ID: {user_data.get('id')}")
                print(f"   Email: {user_data.get('email')}")
            else:
                print(f"   Could not fetch user info (status {response.status_code})")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_payments())
