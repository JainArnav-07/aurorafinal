"""
Asynchronous Template Generator for Notification Orchestrator

Generates exactly 5 bilingual (Hindi + English) push-notification templates
per Segment x Lifecycle x Goal x Theme combination, using a local Ollama LLM.

Requirements (high level):
- Cross-join segment lifecycle/goal data with segment mapping themes.
- For each (segment_id, lifecycle_stage, primary_goal, theme, tone) row:
  - Call a local Ollama endpoint asynchronously (with concurrency limit).
  - Enforce JSON output: list of exactly 5 objects with required keys.
  - Validate and enrich each object with metadata.
  - Export all templates to `message_templates.csv`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import nest_asyncio
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEGMENT_LIFECYCLE_GOALS_JSON = Path("segment_lifecycle_goals.json")
SEGMENT_MAPPING_CSV = Path("segment_mapping.csv")
OUTPUT_CSV = Path("message_templates.csv")

# Local Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"  # Change to the model you have locally

# Concurrency and retry configuration
MAX_CONCURRENCY = 5
MAX_RETRIES = 3  # number of retries on top of the initial attempt

# Set to True to bypass the LLM call and generate deterministic mock templates
MOCK_LLM_MODE = False


REQUIRED_TEMPLATE_KEYS = [
    "hook_used",
    "feature_reference",
    "content_english",
    "content_hindi",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("template_generator")


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def load_segment_lifecycle_goals() -> pd.DataFrame:
    """
    Load segment lifecycle + goals data from JSON, with a mock fallback.

    Expected columns:
        - segment_id
        - lifecycle_stage
        - primary_goal
    """
    if SEGMENT_LIFECYCLE_GOALS_JSON.exists():
        try:
            df = pd.read_json(SEGMENT_LIFECYCLE_GOALS_JSON)
            missing = {
                c
                for c in ["segment_id", "lifecycle_stage", "primary_goal"]
                if c not in df.columns
            }
            if missing:
                raise ValueError(
                    f"segment_lifecycle_goals.json is missing columns: {missing}"
                )
            logger.info(
                "Loaded segment lifecycle goals from %s with %d rows",
                SEGMENT_LIFECYCLE_GOALS_JSON,
                len(df),
            )
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s (%s). Falling back to mock data.",
                SEGMENT_LIFECYCLE_GOALS_JSON,
                exc,
            )

    # Mock fallback
    data = [
        {
            "segment_id": "SEG-A",
            "lifecycle_stage": "onboarding",
            "primary_goal": "start_daily_practice",
        },
        {
            "segment_id": "SEG-B",
            "lifecycle_stage": "habit_building",
            "primary_goal": "increase_speaking_minutes",
        },
    ]
    df = pd.DataFrame(data)
    logger.info("Using mock segment lifecycle goals with %d rows", len(df))
    return df


def load_segment_mapping() -> pd.DataFrame:
    """
    Load segment mapping data from CSV, with a mock fallback.

    Expected columns:
        - segment_id
        - primary_theme
        - secondary_theme
        - tone
    """
    if SEGMENT_MAPPING_CSV.exists():
        try:
            df = pd.read_csv(SEGMENT_MAPPING_CSV)
            missing = {
                c
                for c in [
                    "segment_id",
                    "primary_theme",
                    "secondary_theme",
                    "tone",
                ]
                if c not in df.columns
            }
            if missing:
                raise ValueError(
                    f"segment_mapping.csv is missing columns: {missing}"
                )
            logger.info(
                "Loaded segment mapping from %s with %d rows",
                SEGMENT_MAPPING_CSV,
                len(df),
            )
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s (%s). Falling back to mock data.",
                SEGMENT_MAPPING_CSV,
                exc,
            )

    # Mock fallback aligned with mock lifecycle goals
    data = [
        {
            "segment_id": "SEG-A",
            "primary_theme": "Epic Meaning & Calling",
            "secondary_theme": "Development & Accomplishment",
            "tone": "encouraging",
        },
        {
            "segment_id": "SEG-B",
            "primary_theme": "Development & Accomplishment",
            "secondary_theme": "Scarcity & Impatience",
            "tone": "urgent",
        },
    ]
    df = pd.DataFrame(data)
    logger.info("Using mock segment mapping with %d rows", len(df))
    return df


def prepare_task_dataframe() -> pd.DataFrame:
    """
    Merge lifecycle/goal and mapping data, and explode into per-theme tasks.

    Output columns:
        - segment_id
        - lifecycle_stage
        - primary_goal
        - theme
        - tone
    """
    lifecycle_df = load_segment_lifecycle_goals()
    mapping_df = load_segment_mapping()

    merged = lifecycle_df.merge(mapping_df, on="segment_id", how="inner")
    if merged.empty:
        logger.warning("Merged DataFrame is empty. No tasks to generate.")
        return merged

    primary = merged.copy()
    primary["theme"] = primary["primary_theme"]

    secondary = merged.copy()
    secondary["theme"] = secondary["secondary_theme"]

    tasks_df = pd.concat([primary, secondary], ignore_index=True)

    # Select the minimal required columns
    tasks_df = tasks_df[
        ["segment_id", "lifecycle_stage", "primary_goal", "theme", "tone"]
    ]

    logger.info(
        "Prepared task DataFrame with %d rows (after theme explosion)", len(tasks_df)
    )
    return tasks_df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(row: "TaskRow") -> str:
    """
    Build the LLM prompt enforcing structure and JSON-only output.
    """
    lifecycle_stage = row.lifecycle_stage
    primary_goal = row.primary_goal
    theme = row.theme
    tone = row.tone

    example_json = json.dumps(
        [
            {
                "hook_used": "Curiosity about daily progress",
                "feature_reference": "daily speaking streak tracker",
                "content_english": (
                    "You’re just one practice away from boosting your English speaking streak. "
                    "Open the app now and speak for 3 minutes!"
                ),
                "content_hindi": (
                    "बस एक और प्रैक्टिस से आपकी English speaking streak और मजबूत हो सकती है। "
                    "ऐप खोलें और अभी 3 मिनट बोलने की प्रैक्टिस करें!"
                ),
            }
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are an expert copywriter for an English learning app. Our North Star metric is Daily Active Speaking Minutes.

Write exactly 5 push notification templates for a user in the "{lifecycle_stage}" lifecycle stage.
Their current goal is "{primary_goal}".
You must use the Octalysis core drive of "{theme}" and maintain a "{tone}" tone.

Each template must be bilingual with English and Hindi content.

You MUST obey all of the following formatting rules:
- Output ONLY a JSON array (no explanations, no markdown, no surrounding text).
- The array MUST contain exactly 5 objects.
- Each object MUST have exactly these keys:
  - "hook_used"
  - "feature_reference"
  - "content_english"
  - "content_hindi"

Here is a ONE-SHOT EXAMPLE with only 1 object. Your real answer MUST be the same structure but with exactly 5 objects:

{example_json}

Now output ONLY the JSON array with exactly 5 objects, following the same key names.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Async LLM generator with retries
# ---------------------------------------------------------------------------


@dataclass
class TaskRow:
    segment_id: Any
    lifecycle_stage: Any
    primary_goal: Any
    theme: Any
    tone: Any


async def _mock_generate_templates(row: TaskRow) -> List[Dict[str, Any]]:
    """
    Deterministic mock templates used when MOCK_LLM_MODE is True.
    """
    templates: List[Dict[str, Any]] = []
    for i in range(5):
        templates.append(
            {
                "hook_used": f"Mock hook {i + 1} for {row.theme}",
                "feature_reference": "mock_speaking_feature",
                "content_english": (
                    f"[MOCK] Practice speaking now to progress your goal "
                    f'"{row.primary_goal}" in the {row.lifecycle_stage} stage.'
                ),
                "content_hindi": (
                    f"[MOCK] अभी बोलने की प्रैक्टिस करें ताकि आप अपना goal "
                    f'"{row.primary_goal}" ({row.lifecycle_stage} स्टेज) आगे बढ़ा सकें।'
                ),
            }
        )
    return templates


async def generate_templates(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: TaskRow,
    retries: int = 0,
) -> List[Dict[str, Any]]:
    """
    Asynchronous call to Ollama to generate templates for a single task row.

    Retries up to MAX_RETRIES times on failure, with exponential backoff.
    """
    if MOCK_LLM_MODE:
        return await _mock_generate_templates(row)

    prompt = build_prompt(row)

    async with semaphore:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            }

            logger.debug(
                "Sending request to Ollama for segment_id=%s, theme=%s",
                row.segment_id,
                row.theme,
            )

            async with session.post(OLLAMA_URL, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Ollama returned status {resp.status}: {text}"
                    )

                resp_json = await resp.json()

        except Exception as exc:  # noqa: BLE001
            if retries < MAX_RETRIES:
                backoff = 2**retries
                logger.warning(
                    "Error during Ollama call for segment_id=%s, theme=%s "
                    "(attempt %d/%d). Retrying in %ds. Error: %s",
                    row.segment_id,
                    row.theme,
                    retries + 1,
                    MAX_RETRIES + 1,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
                return await generate_templates(session, semaphore, row, retries + 1)
            logger.error(
                "Exhausted retries for segment_id=%s, theme=%s. Error: %s",
                row.segment_id,
                row.theme,
                exc,
            )
            raise

    # Parse and validate the JSON response
    if "response" not in resp_json:
        raise ValueError("Ollama response JSON missing 'response' field")

    raw_output = resp_json["response"]
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model output: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError("Model output must be a JSON array (list)")

    if len(parsed) != 5:
        raise ValueError(
            f"Model output must contain exactly 5 templates, got {len(parsed)}"
        )

    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Template at index {idx} is not an object/dict")
        missing_keys = [k for k in REQUIRED_TEMPLATE_KEYS if k not in item]
        if missing_keys:
            raise ValueError(
                f"Template at index {idx} is missing keys: {missing_keys}"
            )

    return parsed


# ---------------------------------------------------------------------------
# Metadata injection
# ---------------------------------------------------------------------------


def annotate_templates(row: TaskRow, templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Inject metadata fields into each template dict.
    """
    enriched: List[Dict[str, Any]] = []
    for template in templates:
        # Copy to avoid mutating the original dict
        tpl = {k: template.get(k) for k in REQUIRED_TEMPLATE_KEYS}

        tpl["template_id"] = f"TPL-{uuid.uuid4()}"
        tpl["segment_id"] = row.segment_id
        tpl["lifecycle_stage"] = row.lifecycle_stage
        tpl["goal"] = row.primary_goal
        tpl["theme"] = row.theme
        tpl["tone"] = row.tone

        enriched.append(tpl)
    return enriched


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def process_row(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: TaskRow,
) -> Optional[List[Dict[str, Any]]]:
    """
    End-to-end processing for a single row:
    - call LLM
    - annotate templates
    """
    try:
        templates = await generate_templates(session, semaphore, row)
        enriched = annotate_templates(row, templates)
        return enriched
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to generate templates for segment_id=%s, theme=%s: %s",
            row.segment_id,
            row.theme,
            exc,
        )
        return None


async def main_async() -> None:
    """
    Main async entrypoint:
    - Prepare task DataFrame
    - Run generation with bounded concurrency
    - Flatten results and export to CSV
    """
    tasks_df = prepare_task_dataframe()
    if tasks_df.empty:
        logger.warning("No tasks to process. Exiting without generating templates.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=120)

    all_results: List[List[Dict[str, Any]]] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: List["asyncio.Task[Optional[List[Dict[str, Any]]]]"] = []
        for row_tuple in tasks_df.itertuples(index=False, name=None):
            row = TaskRow(*row_tuple)
            tasks.append(
                asyncio.create_task(process_row(session, semaphore, row))
            )

        logger.info("Dispatching %d async generation tasks", len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = 0
    for result in results:
        if isinstance(result, Exception):
            failures += 1
            logger.error("Task raised an unexpected exception: %s", result)
            continue
        if result is None:
            failures += 1
            continue
        all_results.append(result)

    if not all_results:
        logger.error("All template generation tasks failed. No CSV will be written.")
        return

    # Flatten list of lists into single list of dicts
    flat_templates: List[Dict[str, Any]] = [
        item for sublist in all_results for item in sublist
    ]

    df = pd.DataFrame(flat_templates)

    # Ensure final columns ordering and presence
    final_columns = [
        "template_id",
        "segment_id",
        "lifecycle_stage",
        "goal",
        "theme",
        "tone",
        "hook_used",
        "feature_reference",
        "content_english",
        "content_hindi",
    ]

    # Some defensive handling in case any column is missing
    for col in final_columns:
        if col not in df.columns:
            df[col] = None

    df = df[final_columns]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    total_templates = len(df)
    logger.info(
        "Wrote %d templates to %s (%d failed tasks).",
        total_templates,
        OUTPUT_CSV,
        failures,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Synchronous entrypoint that runs the async main.
    """
    # Helps when running inside notebooks or environments with an existing loop
    nest_asyncio.apply()
    try:
        asyncio.run(main_async())
    except RuntimeError as exc:
        # In case there's already a running loop and nest_asyncio wasn't enough
        logger.error("Failed to run async main: %s", exc)


if __name__ == "__main__":
    main()

