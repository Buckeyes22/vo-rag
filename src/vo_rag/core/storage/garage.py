"""Async S3-compatible blob client for Garage.

Garage rejects SigV2; `signature_version` must be `s3v4`.

The env names `GARAGE_ENDPOINT` / `GARAGE_ACCESS_KEY` / `GARAGE_SECRET_KEY` /
`GARAGE_REGION` are vendor-namespaced rather than vo-rag-namespaced, so they
carry no `VORAG_` prefix.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aioboto3  # type: ignore[import-untyped]
from botocore.client import Config  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class GarageObjectMetadata:
    """Authoritative metadata returned by Garage for one stored object."""

    size: int
    content_type: str | None
    etag: str | None


@dataclass(frozen=True, slots=True)
class VerifiedGarageObject:
    """Metadata plus the SHA-256 calculated from the stored object bytes."""

    size: int
    content_type: str | None
    etag: str | None
    sha256: str


class GarageObjectTooLarge(ValueError):
    """Raised before or during hashing when an object exceeds the byte limit."""


class GarageClient:
    """Async S3-compatible wrapper around Garage.

    Garage rejects SigV2; ``signature_version='s3v4'`` is mandatory.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = "garage",
    ) -> None:
        self._session = aioboto3.Session()
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._config = Config(signature_version="s3v4", s3={"addressing_style": "path"})

    def _client(self) -> Any:
        return self._session.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name=self._region,
            config=self._config,
        )

    async def generate_presigned_upload(
        self,
        bucket: str,
        key: str,
        ttl: int = 21600,
        *,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> str:
        params: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        if content_length is not None:
            params["ContentLength"] = content_length
        async with self._client() as s3:
            url: str = await s3.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=ttl,
            )
            return url

    async def generate_presigned_download(self, bucket: str, key: str, ttl: int = 21600) -> str:
        async with self._client() as s3:
            url: str = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl,
            )
            return url

    async def upload_file(self, bucket: str, key: str, path: Path) -> None:
        async with self._client() as s3:
            await s3.upload_file(str(path), bucket, key)

    async def download_file(self, bucket: str, key: str, dest: Path) -> None:
        async with self._client() as s3:
            await s3.download_file(bucket, key, str(dest))

    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}
        async with self._client() as s3:
            await s3.put_object(Bucket=bucket, Key=key, Body=data, **extra)

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        async with self._client() as s3:
            resp = await s3.get_object(Bucket=bucket, Key=key)
            async with resp["Body"] as body:
                data: bytes = await body.read()
                return data

    async def list_objects(self, bucket: str, prefix: str = "") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    results.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                        }
                    )
        return results

    async def delete_object(self, bucket: str, key: str) -> None:
        async with self._client() as s3:
            await s3.delete_object(Bucket=bucket, Key=key)

    async def head_object(self, bucket: str, key: str) -> dict[str, Any] | None:
        """Compatibility mapping for existing callers; prefer :meth:`object_metadata`."""
        metadata = await self.object_metadata(bucket, key)
        if metadata is None:
            return None
        return {
            "size": metadata.size,
            "content_type": metadata.content_type,
            "etag": metadata.etag,
        }

    async def object_metadata(self, bucket: str, key: str) -> GarageObjectMetadata | None:
        """Return actual object size, media type and ETag without reading its body."""
        async with self._client() as s3:
            try:
                resp = await s3.head_object(Bucket=bucket, Key=key)
                raw_etag = resp.get("ETag")
                return GarageObjectMetadata(
                    size=int(resp["ContentLength"]),
                    content_type=_normalize_content_type(resp.get("ContentType")),
                    etag=str(raw_etag).strip('"') if raw_etag is not None else None,
                )
            except s3.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                    return None
                raise

    async def verify_object_sha256(
        self,
        bucket: str,
        key: str,
        *,
        max_bytes: int,
        chunk_size: int = 1024 * 1024,
    ) -> VerifiedGarageObject | None:
        """Hash the bytes Garage actually stored, with a hard streaming size bound.

        The HEAD size is checked first to avoid downloading an obviously oversized
        object.  The running byte count is checked again because metadata must not
        be the only enforcement boundary.
        """
        metadata = await self.object_metadata(bucket, key)
        if metadata is None:
            return None
        if metadata.size > max_bytes:
            raise GarageObjectTooLarge(
                f"garage object exceeds limit: actual={metadata.size} max={max_bytes}"
            )

        digest = hashlib.sha256()
        total = 0
        async with self._client() as s3:
            get_params: dict[str, Any] = {"Bucket": bucket, "Key": key}
            if metadata.etag is not None:
                get_params["IfMatch"] = f'"{metadata.etag}"'
            response = await s3.get_object(**get_params)
            body = response["Body"]
            # Read through the StreamingBody itself: aiobotocore's __aenter__
            # yields the raw aiohttp response, whose read() takes no size.
            async with body:
                while True:
                    chunk = await body.read(chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise GarageObjectTooLarge(
                            f"garage object exceeds limit while hashing: actual>{max_bytes}"
                        )
                    digest.update(chunk)

        if total != metadata.size:
            raise OSError(
                f"garage object size changed during verification: head={metadata.size} read={total}"
            )
        return VerifiedGarageObject(
            size=total,
            content_type=metadata.content_type,
            etag=metadata.etag,
            sha256=digest.hexdigest(),
        )

    async def promote_verified_object(
        self,
        bucket: str,
        staged_key: str,
        *,
        sha256: str,
        etag: str | None,
    ) -> str:
        """Copy verified bytes to a server-owned content-addressed key.

        The staged key is intentionally retained. The intake caller deletes it
        only after the envelope transaction commits, so a database failure is
        retryable using the original upload key.
        """
        suffix = Path(staged_key).suffix.lower()
        if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) is None:
            suffix = ""
        authoritative_key = f"verified/sha256/{sha256}{suffix}"
        if staged_key == authoritative_key:
            return authoritative_key
        copy_params: dict[str, Any] = {
            "Bucket": bucket,
            "Key": authoritative_key,
            "CopySource": {"Bucket": bucket, "Key": staged_key},
            "MetadataDirective": "COPY",
        }
        if etag is not None:
            copy_params["CopySourceIfMatch"] = f'"{etag}"'
        async with self._client() as s3:
            await s3.copy_object(**copy_params)
        return authoritative_key


def _normalize_content_type(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).split(";", 1)[0].strip().lower()
    return normalized or None


def garage_from_env() -> GarageClient:
    return GarageClient(
        endpoint_url=os.environ["GARAGE_ENDPOINT"],
        access_key=os.environ["GARAGE_ACCESS_KEY"],
        secret_key=os.environ["GARAGE_SECRET_KEY"],
        region=os.environ.get("GARAGE_REGION", "garage"),
    )
