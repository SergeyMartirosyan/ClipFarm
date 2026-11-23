"""Cloudflare R2 (S3-compatible) uploads + retention cleanup, and local cleanup."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class R2:
    def __init__(self, cfg):
        import boto3

        self.cfg = cfg
        self.bucket = cfg.r2_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{cfg.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=cfg.r2_access_key_id,
            aws_secret_access_key=cfg.r2_secret_access_key,
            region_name="auto",
        )

    def public_url(self, key: str) -> str | None:
        base = self.cfg.r2_public_base_url.rstrip("/")
        return f"{base}/{key}" if base else None

    def upload(self, local: Path, key: str, content_type: str) -> str | None:
        self.client.upload_file(str(local), self.bucket, key, ExtraArgs={"ContentType": content_type})
        return self.public_url(key)

    def upload_json(self, data: dict, key: str) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=json.dumps(data, indent=2).encode(), ContentType="application/json"
        )

    def cleanup(self, days: int, prefix: str = "clips/") -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            old = [{"Key": o["Key"]} for o in page.get("Contents", []) if o["LastModified"] < cutoff]
            for i in range(0, len(old), 1000):
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": old[i: i + 1000]})
                deleted += len(old[i: i + 1000])
        return deleted


def clip_key(job_id: int, idx: int, attempt: int = 1, created: float | None = None) -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime(created or time.time()))
    return f"clips/{day}/job-{job_id}/try-{attempt}/clip-{idx:02d}.mp4"


def cleanup_local(clips_dir: Path, days: int) -> int:
    """Delete local per-job clip folders older than `days`."""
    cutoff = time.time() - days * 86400
    removed = 0
    for d in Path(clips_dir).glob("job-*"):
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed
