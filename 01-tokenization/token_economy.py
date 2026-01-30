#!/usr/bin/env python3
"""
Token Economy Calculator
Calculates the cost implications of tokenization for LLM API calls.
"""
from dataclasses import dataclass


@dataclass
class TokenPricing:
    """Pricing for a specific model."""

    model_name: str
    input_cost_per_1k: float  # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens


# Common model pricing (as of 2026)
PRICING_TABLE = {
    "gpt-4": TokenPricing("GPT-4", 0.03, 0.06),
    "gpt-4-turbo": TokenPricing("GPT-4 Turbo", 0.01, 0.03),
    "gpt-3.5-turbo": TokenPricing("GPT-3.5 Turbo", 0.0005, 0.0015),
    "claude-3-opus": TokenPricing("Claude 3 Opus", 0.015, 0.075),
    "claude-3-sonnet": TokenPricing("Claude 3 Sonnet", 0.003, 0.015),
    "llama-3-70b": TokenPricing("Llama 3 70B (hosted)", 0.0008, 0.0008),
}


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4",
) -> dict:
    """Calculate the cost of an API call."""
    if model not in PRICING_TABLE:
        raise ValueError(f"Unknown model: {model}. Available: {list(PRICING_TABLE.keys())}")

    pricing = PRICING_TABLE[model]
    input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
    output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
    total_cost = input_cost + output_cost

    return {
        "model": pricing.model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


def estimate_monthly_cost(
    avg_input_tokens: int,
    avg_output_tokens: int,
    requests_per_day: int,
    model: str = "gpt-4",
) -> dict:
    """Estimate monthly costs based on usage patterns."""
    single_call = calculate_cost(avg_input_tokens, avg_output_tokens, model)
    daily_cost = single_call["total_cost_usd"] * requests_per_day
    monthly_cost = daily_cost * 30

    return {
        "model": single_call["model"],
        "requests_per_day": requests_per_day,
        "requests_per_month": requests_per_day * 30,
        "cost_per_request": single_call["total_cost_usd"],
        "daily_cost_usd": round(daily_cost, 2),
        "monthly_cost_usd": round(monthly_cost, 2),
        "annual_cost_usd": round(monthly_cost * 12, 2),
    }


def compression_ratio(original_chars: int, num_tokens: int) -> float:
    """Calculate the compression ratio of tokenization."""
    return original_chars / num_tokens if num_tokens > 0 else 0


if __name__ == "__main__":
    print("=== Token Economy Calculator ===\n")

    # Example 1: Single API call
    print("Example 1: Single API Call")
    text = "Hello, how are you today? I'm building an LLM application."
    input_tokens = 15  # Approximate
    output_tokens = 50  # Approximate response length

    for model in ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]:
        cost = calculate_cost(input_tokens, output_tokens, model)
        print(f"{cost['model']}: ${cost['total_cost_usd']:.6f}")

    print("\n" + "=" * 60 + "\n")

    # Example 2: Monthly cost estimation
    print("Example 2: Monthly Cost Estimation")
    print("Scenario: Chatbot with 1,000 requests/day\n")

    scenario = estimate_monthly_cost(
        avg_input_tokens=500,
        avg_output_tokens=300,
        requests_per_day=1000,
        model="gpt-4",
    )

    print(f"Model: {scenario['model']}")
    print(f"Requests/day: {scenario['requests_per_day']:,}")
    print(f"Requests/month: {scenario['requests_per_month']:,}")
    print(f"Cost per request: ${scenario['cost_per_request']:.6f}")
    print(f"Daily cost: ${scenario['daily_cost_usd']:,.2f}")
    print(f"Monthly cost: ${scenario['monthly_cost_usd']:,.2f}")
    print(f"Annual cost: ${scenario['annual_cost_usd']:,.2f}")

    print("\n" + "=" * 60 + "\n")

    # Example 3: Compression analysis
    print("Example 3: Tokenization Efficiency")
    test_text = "The quick brown fox jumps over the lazy dog"
    char_count = len(test_text)
    token_count = 10  # Approximate for GPT tokenizer

    ratio = compression_ratio(char_count, token_count)
    print(f"Text: '{test_text}'")
    print(f"Characters: {char_count}")
    print(f"Tokens: {token_count}")
    print(f"Compression ratio: {ratio:.2f} chars/token")
    print(f"\nNote: English text typically has ~4 chars/token")
    print(f"      Code typically has ~2.5 chars/token")
    print(f"      Chinese text has ~1.5 chars/token")



