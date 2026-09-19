"""Resource and network boundary for a small local TIMDR environment.

The local machine evaluates frozen artifacts.  External material enters only
through a declared, checksummed source manifest; it never changes a frozen
preregistration or triggers training by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class EnvironmentPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalBudget:
    max_parallel_jobs: int = 1
    max_episode_bytes: int = 8 * 1024 * 1024
    max_download_bytes: int = 25 * 1024 * 1024


BUDGET = LocalBudget()


def assert_episode_size(path: str | Path) -> None:
    target = Path(path)
    if target.stat().st_size > BUDGET.max_episode_bytes:
        raise EnvironmentPolicyError(
            f"Episode exceeds local limit of {BUDGET.max_episode_bytes} bytes: {target.name}"
        )


def download_declared_source(url: str, expected_sha256: str, output_path: str | Path) -> Path:
    """Explicit, bounded download. No discovery, crawling, upload, or code execution."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise EnvironmentPolicyError("Only an explicit HTTPS source URL is accepted.")
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256.lower()):
        raise EnvironmentPolicyError("A lowercase SHA-256 checksum is required before download.")
    target = Path(output_path)
    if target.exists():
        raise EnvironmentPolicyError(f"Refusing to overwrite existing cache item: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, received = sha256(), 0
    try:
        with urlopen(Request(url, headers={"User-Agent": "TIMDR-AI-Core/0.1"}), timeout=30) as response:
            with target.open("xb") as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > BUDGET.max_download_bytes:
                        raise EnvironmentPolicyError("Download exceeds configured local byte budget.")
                    digest.update(chunk)
                    handle.write(chunk)
    except Exception:
        if target.exists():
            target.unlink()
        raise
    if digest.hexdigest() != expected_sha256.lower():
        target.unlink()
        raise EnvironmentPolicyError("Downloaded checksum differs from declared SHA-256.")
    return target
