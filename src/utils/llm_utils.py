import asyncio
import base64
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import aiofiles
import wandb
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from src.utils.eval_utils import NarrativeDiversityTracker


def create_client(provider: str = "nous") -> tuple[AsyncOpenAI, AsyncLimiter]:
    """Create async client and rate limiter for specified provider.

    Args:
        provider: API provider, either "nous" or "openai".

    Returns:
        Tuple of (AsyncOpenAI client, AsyncLimiter for rate limiting).

    Raises:
        ValueError: If required environment variables are not set or provider is unknown.
    """
    if provider == "nous":
        api_key = os.environ.get("NOUS_API_KEY")
        api_url = os.environ.get("NOUS_BASE_URL")
        if not api_key or not api_url:
            raise ValueError(
                "NOUS_API_KEY and NOUS_BASE_URL must be set in environment"
            )
        client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        limiter = AsyncLimiter(400, 60)
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment")
        client = AsyncOpenAI(api_key=api_key)
        limiter = AsyncLimiter(4950, 60)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return client, limiter


def encode_image_base64(image_path: str) -> str:
    """Encode image file to base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded string of the image contents.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_xml_tag(response: str, tag: str) -> str | None:
    """Parse content from XML tag in response.

    Args:
        response: Raw response string containing XML tags.
        tag: Name of the XML tag to extract content from.

    Returns:
        The text content inside the tag, or None if not found or parse error.
    """
    try:
        root = ET.fromstring(f"<root>{response}</root>")
        elem = root.find(tag)
        if elem is not None and elem.text:
            return elem.text.strip()
    except ET.ParseError:
        pass
    return None


async def call_llm_with_retry(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    content: str | list[dict],
    model: str,
    max_tokens: int = 400,
    temperature: float = 0.8,
    max_attempts: int = 3,
) -> str:
    """Make an LLM API call with retry logic and rate limiting.

    Args:
        client: AsyncOpenAI client instance.
        limiter: Rate limiter for API calls.
        content: Message content (string for text, list of dicts for multimodal).
        model: Model identifier to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        max_attempts: Number of retry attempts on failure.

    Returns:
        The response text from the model.

    Raises:
        Exception: The last error encountered after all retry attempts fail.
    """
    last_error = None
    messages = cast(
        list[ChatCompletionMessageParam],
        [{"role": "user", "content": content}],
    )

    for attempt in range(max_attempts):
        try:
            async with limiter:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            raw_response = response.choices[0].message.content
            if not raw_response:
                raise ValueError("API response content is empty or missing.")

            return raw_response.strip()

        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(1)
            continue

    raise last_error if last_error else RuntimeError("Failed after max attempts")


def build_narrative_prompt(
    metadata: dict,
    params: dict,
    prompt_template: str,
) -> str:
    """Build narrative generation prompt from metadata and parameters.

    Args:
        metadata: Patient metadata (age_approx, sex, anatom_site_general).
        params: Generation parameters (style, memory, symptoms, etc.).
        prompt_template: Template string with placeholders.

    Returns:
        Formatted prompt string ready for LLM.
    """
    meta_json = json.dumps({k: v for k, v in metadata.items() if v is not None})
    symptom_pools = ", ".join(params["symptoms"]) if params["symptoms"] else "none"
    action_cues = params.get("action_cues") or "none"
    context_slot = params.get("context_slot") or "none"
    narrative_mode = params.get("narrative_mode") or "lesion"

    return prompt_template.format(
        meta_json=meta_json,
        style_token=params["style"],
        memory=params["memory"],
        symptom_pools=symptom_pools,
        temporal_quantifiers=params["temporal_quantifier"],
        structure=params["structure"],
        action_cues=action_cues,
        context_slot=context_slot,
        narrative_mode=narrative_mode,
    )


async def generate_narrative(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    metadata: dict,
    params: dict,
    prompt_template: str,
) -> str:
    """Generate a patient narrative using the Nous/Hermes model.

    Args:
        client: AsyncOpenAI client instance.
        limiter: Rate limiter for API calls.
        metadata: Patient metadata (age_approx, sex, anatom_site_general).
        params: Generation parameters (style, memory, symptoms, etc.).
        prompt_template: Template string for prompt construction.

    Returns:
        Generated narrative text.
    """
    content = build_narrative_prompt(metadata, params, prompt_template)

    raw_response = await call_llm_with_retry(
        client=client,
        limiter=limiter,
        content=content,
        model="Hermes-4-70B",
        max_tokens=400,
        temperature=0.8,
    )

    narrative = parse_xml_tag(raw_response, "narrative")
    return narrative if narrative else raw_response


CLASSIFICATION_PROMPT = """Classify this skin lesion into one of these categories:
0: Melanoma
1: Melanocytic nevus
2: Basal cell carcinoma
3: Actinic keratosis
4: Benign keratosis
5: Dermatofibroma
6: Vascular lesion
7: Squamous cell carcinoma

{input_description}

Respond with the class number (0-7) wrapped in XML tags like <answer>3</answer>."""


def build_classification_content(
    image_base64: str | None,
    narrative_text: str | None,
) -> list[dict]:
    """Build multimodal content for GPT-5 classification request.

    Args:
        image_base64: Base64-encoded image string, or None for text-only.
        narrative_text: Patient narrative text, or None for image-only.

    Returns:
        List of content dicts suitable for OpenAI chat completion API.
    """
    content = []

    if image_base64 and narrative_text:
        input_desc = f"Patient narrative: {narrative_text}"
    elif narrative_text:
        input_desc = f"Patient narrative: {narrative_text}"
    else:
        input_desc = "Examine the image below."

    content.append(
        {
            "type": "text",
            "text": CLASSIFICATION_PROMPT.format(input_description=input_desc),
        }
    )

    if image_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )

    return content


def parse_classification_response(response: str) -> int:
    """Parse classification response to extract class number.

    Args:
        response: Raw response string from GPT-5.

    Returns:
        Predicted class number (0-7).

    Raises:
        ValueError: If no valid class number can be extracted from response.
    """
    answer = parse_xml_tag(response, "answer")
    if answer and answer.isdigit() and 0 <= int(answer) <= 7:
        return int(answer)

    match = re.search(r"[0-7]", response)
    if match:
        return int(match.group())

    raise ValueError(f"Could not parse class from response: {response}")


async def classify_lesion(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    image_base64: str | None,
    narrative_text: str | None,
) -> int:
    """Classify a skin lesion using GPT-5.

    Args:
        client: AsyncOpenAI client instance.
        limiter: Rate limiter for API calls.
        image_base64: Base64-encoded image string, or None for text-only.
        narrative_text: Patient narrative text, or None for image-only.

    Returns:
        Predicted class number (0-7).
    """
    content = build_classification_content(image_base64, narrative_text)

    raw_response = await call_llm_with_retry(
        client=client,
        limiter=limiter,
        content=content,
        model="gpt-5-chat-latest",
        max_tokens=50,
        temperature=0.0,
    )

    return parse_classification_response(raw_response)


async def process_narrative_batch(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    batch_entries: list[dict],
    prompt_template: str,
) -> list[tuple[dict, str]]:
    """Generate narrative texts for a batch of entries concurrently.

    Args:
        client: AsyncOpenAI client instance.
        limiter: Rate limiter for API calls.
        batch_entries: List of manifest entries with metadata and params.
        prompt_template: Template string for prompt construction.

    Returns:
        List of (entry, narrative_text) tuples.
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
            "action_cues": entry.get("action_cues"),
            "context_slot": entry.get("context_slot"),
            "narrative_mode": entry.get("narrative_mode", "lesion"),
        }

        tasks.append(
            generate_narrative(client, limiter, metadata, params, prompt_template)
        )

    narrative_texts = await asyncio.gather(*tasks)

    return list(zip(batch_entries, narrative_texts))


async def classify_entry(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    entry: dict,
    modality: str,
) -> int:
    """Classify a single entry, loading data lazily.

    Args:
        client: AsyncOpenAI client instance.
        limiter: Rate limiter for API calls.
        entry: Manifest entry with image_path, text_path, label.
        modality: Input modality ("image", "text", or "fusion").

    Returns:
        Predicted class number (0-7).
    """
    image_base64 = None
    narrative_text = None

    if modality in ("image", "fusion"):
        image_base64 = encode_image_base64(entry["image_path"])

    if modality in ("text", "fusion"):
        with open(entry["text_path"]) as f:
            narrative_text = f.read().strip()

    return await classify_lesion(client, limiter, image_base64, narrative_text)


async def process_classification_batch(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    batch_entries: list[dict],
    modality: str,
) -> list[tuple[int, int]]:
    """Classify a batch of entries concurrently with lazy loading.

    Args:
        client: AsyncOpenAI client instance.
        limiter: Rate limiter for API calls.
        batch_entries: List of manifest entries with image_path, text_path, label.
        modality: Input modality ("image", "text", or "fusion").

    Returns:
        List of (prediction, label) tuples.
    """
    tasks = [
        classify_entry(client, limiter, entry, modality) for entry in batch_entries
    ]

    predictions = await asyncio.gather(*tasks)
    labels = [entry["label"] for entry in batch_entries]

    return list(zip(predictions, labels))


async def process_narrative_entries(
    narrative_entries: list[dict],
    prompt_template: str,
    split_name: str,
    batch_size: int = 4,
    log_interval: int = 100,
    sample_interval: int = 100,
) -> int:
    """Process all narrative entries, generating text and writing to files.

    Tracks diversity metrics (Distinct-1, Distinct-2) incrementally and logs
    to wandb. Computes Self-BLEU at the end of generation.

    Args:
        narrative_entries: List of manifest entries to process.
        prompt_template: Template string for prompt construction.
        split_name: Name of the data split (for progress bar display).
        batch_size: Number of entries to process per batch.
        log_interval: Number of batches between wandb logs.

    Returns:
        Total number of entries processed.
    """
    client, limiter = create_client("nous")
    tracker = NarrativeDiversityTracker()

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "narrative-generation"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"narratives_{split_name}",
        config={
            "split": split_name,
            "total_entries": len(narrative_entries),
            "batch_size": batch_size,
        },
    )

    total_batches = (len(narrative_entries) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(
        tqdm(
            range(0, len(narrative_entries), batch_size),
            desc=f"Generating {split_name} narrative batches",
        )
    ):
        batch_entries = narrative_entries[i : i + batch_size]
        results = await process_narrative_batch(
            client, limiter, batch_entries, prompt_template
        )

        for entry, narrative_text in results:
            tracker.update(narrative_text)
            text_path = Path(entry["text_path"])
            async with aiofiles.open(text_path, mode="w") as f:
                await f.write(narrative_text)

        if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
            metrics = tracker.get_metrics(compute_self_bleu=False)
            step = int(metrics["num_narratives"])
            wandb.log(
                {
                    "narratives/distinct_1": metrics["distinct_1"],
                    "narratives/distinct_2": metrics["distinct_2"],
                    "narratives/unique_unigrams": metrics["unique_unigrams"],
                    "narratives/unique_bigrams": metrics["unique_bigrams"],
                },
                step=step,
            )

        if sample_interval and batch_idx % sample_interval == 0 and results:
            sample_entry, sample_text = results[0]
            print(f"\nSample: {sample_entry['text_path']}\n{sample_text}\n")

    print(f"\nComputing Self-BLEU for {split_name} split...")
    final_metrics = tracker.get_metrics(compute_self_bleu=True)

    wandb.log(
        {
            "narratives/distinct_1": final_metrics["distinct_1"],
            "narratives/distinct_2": final_metrics["distinct_2"],
            "narratives/self_bleu": final_metrics["self_bleu"],
            "narratives/unique_unigrams": final_metrics["unique_unigrams"],
            "narratives/unique_bigrams": final_metrics["unique_bigrams"],
        },
        step=int(final_metrics["num_narratives"]),
    )

    wandb.summary["narratives/distinct_1"] = final_metrics["distinct_1"]
    wandb.summary["narratives/distinct_2"] = final_metrics["distinct_2"]
    wandb.summary["narratives/self_bleu"] = final_metrics["self_bleu"]

    print(f"\n{split_name} Diversity Metrics:")
    print(f"  Distinct-1: {final_metrics['distinct_1']:.4f}")
    print(f"  Distinct-2: {final_metrics['distinct_2']:.4f}")
    print(f"  Self-BLEU:  {final_metrics['self_bleu']:.4f} (lower is better)")

    wandb.finish()

    return len(narrative_entries)


async def process_classification_entries(
    entries: list[dict],
    modality: str,
    batch_size: int = 4,
) -> tuple[list[int], list[int]]:
    """Process all entries for classification with GPT-5.

    Args:
        entries: List of manifest entries with image_path, text_path, label.
        modality: Input modality ("image", "text", or "fusion").
        batch_size: Number of entries to process per batch.

    Returns:
        Tuple of (predictions list, labels list).
    """
    client, limiter = create_client("openai")

    all_predictions = []
    all_labels = []

    for i in tqdm(
        range(0, len(entries), batch_size),
        desc=f"Classifying {modality} batches",
    ):
        batch_entries = entries[i : i + batch_size]
        results = await process_classification_batch(
            client, limiter, batch_entries, modality
        )

        for pred, label in results:
            all_predictions.append(pred)
            all_labels.append(label)

        if i == 0:
            sample_pred, sample_label = results[0]
            print(f"\nSample: pred={sample_pred}, label={sample_label}")

    return all_predictions, all_labels
