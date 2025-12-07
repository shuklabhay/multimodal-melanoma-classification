import asyncio
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import aiofiles
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm import tqdm


async def generate_narrative_with_nous(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    metadata: dict,
    params: dict,
    prompt_template: str,
) -> str:
    """Makes async API call to Nous (Hermes 4) to generate narrative using OpenAI client.

    Args:
        client: Initialized AsyncOpenAI client.
        limiter: Rate limiter to prevent hitting API limits.
        metadata: Patient metadata (age, sex, site).
        params: Selected narrative parameters.
        prompt_template: Prompt template string.

    Returns:
        Generated narrative text extracted from XML tags.
    """
    meta_json = json.dumps({k: v for k, v in metadata.items() if v is not None})

    symptom_pools = ", ".join(params["symptoms"]) if params["symptoms"] else "none"
    filled_prompt = prompt_template.format(
        meta_json=meta_json,
        style_token=params["style"],
        memory=params["memory"],
        symptom_pools=symptom_pools,
        temporal_quantifiers=params["temporal_quantifier"],
        structure=params["structure"],
    )

    max_attempts = 3
    last_error = None

    for attempt in range(max_attempts):
        try:
            async with limiter:
                response = await client.chat.completions.create(
                    model="Hermes-4-70B",
                    messages=[{"role": "user", "content": filled_prompt}],
                    max_tokens=400,
                    temperature=0.8,
                )

            raw_response = response.choices[0].message.content
            if not raw_response:
                raise ValueError("API response content is empty or missing.")

            raw_response = raw_response.strip()
            root = ET.fromstring(f"<root>{raw_response}</root>")
            narrative_elem = root.find("narrative")
            if narrative_elem is not None and narrative_elem.text:
                return narrative_elem.text.strip()

            return raw_response

        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(1)
            continue

    raise last_error if last_error else RuntimeError("Failed after max attempts")


def create_llm_client() -> tuple[AsyncOpenAI, AsyncLimiter]:
    """Creates and returns an async OpenAI client and rate limiter.

    Returns:
        Tuple of (AsyncOpenAI client, AsyncLimiter).

    Raises:
        ValueError: If required environment variables are not set.
    """
    api_key = os.environ.get("NOUS_API_KEY")
    api_url = os.environ.get("NOUS_BASE_URL")

    if not api_key or not api_url:
        raise ValueError("NOUS_API_KEY and NOUS_BASE_URL must be set in environment")

    client = AsyncOpenAI(api_key=api_key, base_url=api_url)
    limiter = AsyncLimiter(95, 60)

    return client, limiter


async def generate_narrative_batch(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    batch_entries: list[dict],
    prompt_template: str,
) -> list[tuple[dict, str]]:
    """Generate narrative texts for a batch of entries concurrently.

    Args:
        client: Initialized AsyncOpenAI client.
        limiter: Rate limiter to prevent hitting API limits.
        batch_entries: List of narrative manifest entries to process.
        prompt_template: Prompt template string.

    Returns:
        List of tuples containing (entry, generated_narrative_text).
    """
    tasks = []
    for entry in batch_entries:
        metadata = {
            "age_approx": entry.get("age_approx"),
            "sex": entry.get("sex"),
            "anatom_site_general": entry.get("anatom_site_general"),
        }

        params = {
            "style": entry["style"],
            "memory": entry["memory"],
            "memory_category": entry["memory_category"],
            "temporal_quantifier": entry["temporal_quantifier"],
            "symptoms": entry["symptoms"],
            "structure": entry["structure"],
        }

        tasks.append(
            (
                entry,
                generate_narrative_with_nous(
                    client,
                    limiter,
                    metadata,
                    params,
                    prompt_template,
                ),
            )
        )

    entries_list = [t[0] for t in tasks]
    coros = [t[1] for t in tasks]
    narrative_texts = await asyncio.gather(*coros)

    results = []
    for i, text in enumerate(narrative_texts):
        results.append((entries_list[i], text))
    return results


async def process_narrative_entries(
    narrative_entries: list[dict],
    prompt_template: str,
    split_name: str,
    batch_size: int = 4,
) -> int:
    """Process all narrative entries, generating text and writing to files.

    Args:
        narrative_entries: List of narrative manifest entries to process.
        prompt_template: Prompt template string.
        split_name: Name of the split being processed (for progress display).
        batch_size: Number of entries to process concurrently.

    Returns:
        Total number of entries processed.
    """
    client, limiter = create_llm_client()

    for i in tqdm(
        range(0, len(narrative_entries), batch_size),
        desc=f"Generating {split_name} narrative batches",
    ):
        batch_entries = narrative_entries[i : i + batch_size]
        results = await generate_narrative_batch(
            client, limiter, batch_entries, prompt_template
        )

        sample_entry, sample_text = results[0]
        print(f"\nSample: {sample_entry['text_path']}\n{sample_text}\n")

        for entry, narrative_text in results:
            text_path = Path(entry["text_path"])
            async with aiofiles.open(text_path, mode="w") as f:
                await f.write(narrative_text)

    return len(narrative_entries)
