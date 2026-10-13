"""R2 presigned-URL artifact exchange.

Cloud workers never hold Garage or long-lived R2 credentials — they receive
presigned GET (input) + PUT (output) URLs valid for up to 6 hours. SigV4 is
mandatory for Cloudflare R2.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any

import aioboto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]

from vo_rag.core.storage.garage import GarageClient

log = logging.getLogger("vo_rag.cloud.r2_staging")

DEFAULT_TTL_SECONDS = 6 * 60 * 60
STAGING_IN_PREFIX = "staging/in"
STAGING_OUT_PREFIX = "staging/out"


def _resolve_artifacts_bucket() -> str:
    try:
        from vo_rag.core.config import get_settings

        return get_settings().garage.buckets.artifacts
    except Exception:
        return "vorag-artifacts"


@dataclass(frozen=True)
class StagedInput:
    r2_key: str
    presigned_get_url: str


@dataclass(frozen=True)
class StagedOutputSlot:
    r2_key: str
    presigned_put_url: str


class R2Staging:
    """R2 staging helper for one-shot artifact exchange via presigned URLs."""

    def __init__(
        self,
        r2_endpoint: str,
        r2_access_key: str,
        r2_secret_key: str,
        r2_bucket: str,
        garage: GarageClient,
    ) -> None:
        if not (r2_endpoint and r2_access_key and r2_secret_key and r2_bucket):
            raise ValueError("all four R2 parameters + bucket are required")
        self.bucket = r2_bucket
        self.garage = garage
        self._garage_bucket = _resolve_artifacts_bucket()
        self._session = aioboto3.Session()
        self._config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            region_name="auto",
        )
        self._client_kwargs: dict[str, Any] = {
            "endpoint_url": r2_endpoint,
            "aws_access_key_id": r2_access_key,
            "aws_secret_access_key": r2_secret_key,
            "region_name": "auto",
            "config": self._config,
        }

    def _client(self) -> Any:
        return self._session.client("s3", **self._client_kwargs)

    async def stage_input(
        self,
        garage_key: str,
        *,
        ttl: int = DEFAULT_TTL_SECONDS,
    ) -> StagedInput:
        """Copy Garage object → R2 staging/in/<key>, return presigned GET URL."""
        data = await self.garage.get_bytes(self._garage_bucket, garage_key)
        r2_key = f"{STAGING_IN_PREFIX}/{garage_key}"
        async with self._client() as s3:
            await s3.put_object(Bucket=self.bucket, Key=r2_key, Body=data)
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": r2_key},
                ExpiresIn=ttl,
            )
        return StagedInput(r2_key=r2_key, presigned_get_url=url)

    async def stage_bytes(
        self,
        data: bytes,
        *,
        key: str,
        ttl: int = DEFAULT_TTL_SECONDS,
        content_type: str = "application/json",
    ) -> StagedInput:
        """Write a homelab-built bundle to R2 and return a presigned GET URL."""
        r2_key = f"{STAGING_IN_PREFIX}/{key}"
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=r2_key,
                Body=data,
                ContentType=content_type,
            )
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": r2_key},
                ExpiresIn=ttl,
            )
        return StagedInput(r2_key=r2_key, presigned_get_url=url)

    async def get_output_url(
        self,
        output_key: str,
        *,
        ttl: int = DEFAULT_TTL_SECONDS,
    ) -> StagedOutputSlot:
        r2_key = f"{STAGING_OUT_PREFIX}/{output_key}"
        async with self._client() as s3:
            url = await s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket, "Key": r2_key},
                ExpiresIn=ttl,
            )
        return StagedOutputSlot(r2_key=r2_key, presigned_put_url=url)

    async def pull_output(self, r2_key: str, garage_dest_key: str) -> None:
        """Pull completed output from R2 → Garage; delete the R2 staging object."""
        async with self._client() as s3:
            obj = await s3.get_object(Bucket=self.bucket, Key=r2_key)
            async with obj["Body"] as body:
                data = await body.read()
            await s3.delete_object(Bucket=self.bucket, Key=r2_key)
        await self.garage.put_bytes(self._garage_bucket, garage_dest_key, data)

    async def read_output_bytes(self, r2_key: str, *, delete: bool = True) -> bytes:
        """Read a completed cloud output directly from R2 for local finalization."""
        async with self._client() as s3:
            obj = await s3.get_object(Bucket=self.bucket, Key=r2_key)
            async with obj["Body"] as body:
                data = await body.read()
            if delete:
                await s3.delete_object(Bucket=self.bucket, Key=r2_key)
        return bytes(data)

    async def cleanup_expired(self, *, max_age_hours: int = 24) -> int:
        """Delete staging/in/* and staging/out/* older than max_age_hours."""
        cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=max_age_hours)  # noqa: UP017
        removed = 0
        async with self._client() as s3:
            for prefix in (STAGING_IN_PREFIX, STAGING_OUT_PREFIX):
                paginator = s3.get_paginator("list_objects_v2")
                async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix + "/"):
                    for obj in page.get("Contents", []) or []:
                        if obj["LastModified"] < cutoff:
                            await s3.delete_object(Bucket=self.bucket, Key=obj["Key"])
                            removed += 1
        return removed
