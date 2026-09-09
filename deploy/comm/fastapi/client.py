#!/usr/bin/env python3
"""
FastAPI-based Policy Client for Remote Inference (HTTP/JSON).

This is an HTTP/JSON alternative to the TCP+pickle client in `deploy/comm/client.py`.

Key requirement:
- Client send function accepts `MetaObs`, converts to dict, sends via FastAPI POST
- Server returns actions, client converts back into the same structure used by existing `PolicyClient`
  (a list of numpy object arrays where each element is a dict compatible with MetaAction).

Note:
- SSL certificate verification is disabled by default to support self-signed certificates.
- Each client instance has a unique `client_id` for server-side request deduplication.
- For older remote servers (no state_ee remap / no JPEG codec), this client can
  letterbox-resize images and copy state_ee → state before sending.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import requests
import urllib3
from loguru import logger

from benchmark.base import MetaObs

from ..base import BaseClient
from .codec import (
    JPEG_TYPE_NAME,
    NDARRAY_TYPE_TAG,
    NDARRAY_TYPE_NAME,
    decode_images_jpeg,
    decode_ndarray,
    encode_images_jpeg,
    encode_ndarray,
    payload_nbytes_estimate,
    prepare_meta_obs_for_remote,
)

# Suppress InsecureRequestWarning globally for this module (self-signed certs are common)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        # object arrays can contain Python dicts; do not base64-encode raw pointers
        if x.dtype == object:
            return [_to_jsonable(v) for v in x.tolist()]
        return encode_ndarray(x)
    if isinstance(x, (np.generic,)):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _meta_obs_to_jsonable(
    meta_obs: MetaObs,
    *,
    image_encoding: str,
    jpeg_quality: int,
) -> Dict[str, Any]:
    d = asdict(meta_obs)
    image = d.get("image")
    if image is not None and isinstance(image, np.ndarray) and image.dtype != object:
        if image_encoding == "jpeg":
            d["image"] = encode_images_jpeg(image, quality=jpeg_quality)
        else:
            d["image"] = encode_ndarray(np.ascontiguousarray(image))
    return {str(k): (v if k == "image" and isinstance(v, dict) else _to_jsonable(v)) for k, v in d.items()}


def _from_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, dict):
        tag = x.get(NDARRAY_TYPE_TAG)
        if tag == NDARRAY_TYPE_NAME:
            return decode_ndarray(x)
        if tag == JPEG_TYPE_NAME:
            return decode_images_jpeg(x)
        return {k: _from_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_from_jsonable(v) for v in x]
    return x


def _normalize_mact_list(decoded: Any) -> List[np.ndarray]:
    """
    Convert decoded JSON response into the same structure returned by MetaPolicy.inference:
      List[np.ndarray(dtype=object)] where each element holds per-batch dict(s) describing MetaAction.
    """
    if decoded is None:
        return []
    if not isinstance(decoded, list):
        raise ValueError(f"Expected mact_list to be a list, got {type(decoded)}")

    out: List[np.ndarray] = []
    for step in decoded:
        # Each step is typically a list of dicts (batch actions) or already a numpy object array.
        if isinstance(step, np.ndarray):
            out.append(step.astype(object))
        elif isinstance(step, list):
            out.append(np.array(step, dtype=object))
        else:
            # Fallback: single action dict
            out.append(np.array([step], dtype=object))
    return out


class FastAPIPolicyClient(BaseClient):
    """
    HTTP/JSON policy client using FastAPI server.

    Provides `send_meta_obs` / `inference` interface for sending observations
    and receiving the full action chunk list from the server.
    Action queue management is delegated to the action_manager.

    Legacy remote adaptation (default on):
    - Copy ``state_ee`` → ``state`` (old MetaPolicy ignores state_ee)
    - Letterbox-resize images to ``image_size`` (default 256x256, pad=0)
    - Optionally JPEG-compress images when the server advertises support
      (``image_encoding='auto'|'jpeg'``). Old servers fall back to raw uint8.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 30.0,
        ctrl_space: str = "ee",
        ctrl_type: str = "delta",
        client_id: Optional[str] = None,
        *,
        adapt_legacy_server: bool = True,
        prefer_state_ee: bool = True,
        image_size: Optional[Union[int, Sequence[int]]] = (256, 256),
        image_pad_value: int = 0,
        image_encoding: str = "auto",
        jpeg_quality: int = 85,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.adapt_legacy_server = bool(adapt_legacy_server)
        self.prefer_state_ee = bool(prefer_state_ee)
        self.image_size = image_size
        self.image_pad_value = int(image_pad_value)
        self.image_encoding = str(image_encoding).lower()
        self.jpeg_quality = int(jpeg_quality)
        self._resolved_image_encoding: Optional[str] = None
        self._logged_adapt = False

        # Generate unique client_id for request deduplication
        if client_id is None:
            client_id = f"fastapi_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.client_id = client_id

        self._session = requests.Session()
        # Disable SSL verification to support self-signed certificates
        self._session.verify = False
        logger.info(f"✓ Connected to FastAPI policy server at {self.base_url}")
        logger.info(f"   Client ID: {self.client_id}")
        if self.adapt_legacy_server:
            logger.info(
                "   Legacy adapt ON: state_ee→state, letterbox={}, image_encoding={}",
                self.image_size,
                self.image_encoding,
            )

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=self.timeout_s)
            if r.status_code != 200 or not (r.content or b"").strip():
                return False
            return True
        except Exception:
            return False

    def _server_image_codecs(self) -> List[str]:
        try:
            r = self._session.get(
                f"{self.base_url}/health",
                timeout=min(10.0, self.timeout_s),
            )
            if r.status_code != 200:
                logger.warning(
                    "Remote /health HTTP {} — is SSH -L tunnel up? "
                    "(Cursor Port Forward on the same local port often returns empty)",
                    r.status_code,
                )
                return []
            if not (r.content or b"").strip():
                logger.error(
                    "Remote /health empty reply from {} — local port is almost certainly "
                    "a dead Cursor Port Forward, not ssh -L. Fix:\n"
                    "  1) In Cursor: stop Port Forward for this port\n"
                    "  2) ssh -N -L 11002:127.0.0.1:10002 -p 9022 root@127.0.0.1\n"
                    "  3) --ckpt 'http://127.0.0.1:11002'\n"
                    "  4) On GPU host: curl -s http://127.0.0.1:10002/health | grep jpeg",
                    self.base_url,
                )
                return []
            data = r.json()
            codecs = data.get("image_codecs") or data.get("codecs") or []
            if isinstance(codecs, str):
                return [codecs]
            return [str(c) for c in codecs]
        except Exception as e:
            logger.warning("Remote /health failed ({}): {}", self.base_url, e)
            return []

    def _resolve_image_encoding(self) -> str:
        if self._resolved_image_encoding is not None:
            return self._resolved_image_encoding

        enc = self.image_encoding
        if enc == "auto":
            codecs = self._server_image_codecs()
            if JPEG_TYPE_NAME in codecs or "jpeg" in codecs:
                enc = "jpeg"
                logger.info(
                    "Remote /health advertises JPEG ({}) — using jpeg",
                    codecs,
                )
            else:
                # Prefer JPEG anyway (ACT/DP/AsiMeta/… identical wire format).
                # If the server cannot decode, send_meta_obs falls back to raw.
                enc = "jpeg"
                logger.warning(
                    "Remote /health has no image_codecs={} — still trying JPEG first "
                    "(~0.02MB). If decode fails we fall back to raw. "
                    "Prefer: python start_policy_server_jpeg.py … and "
                    "curl /health | grep jpeg_nchw. Or set ILSTUDIO_IMAGE_ENCODING=raw.",
                    codecs,
                )
        elif enc not in ("raw", "jpeg"):
            raise ValueError(f"Unsupported image_encoding={enc!r} (use auto|raw|jpeg)")

        self._resolved_image_encoding = enc
        logger.info("FastAPI client image_encoding={}", enc)
        return enc

    def send_meta_obs(
        self,
        meta_obs: MetaObs,
        *,
        episode_id: Optional[int] = None,
        index: Optional[int] = None,
        reasoning: Any = None,
        timestamp: Optional[float] = None,
    ) -> List[np.ndarray]:
        mobs = meta_obs
        if self.adapt_legacy_server:
            mobs = prepare_meta_obs_for_remote(
                meta_obs,
                prefer_state_ee=self.prefer_state_ee,
                image_size=self.image_size,
                pad_value=self.image_pad_value,
            )
            if not self._logged_adapt:
                st = None if mobs.state is None else np.asarray(mobs.state).reshape(-1)[:7]
                im = None if mobs.image is None else tuple(mobs.image.shape)
                logger.info(
                    "Legacy MetaObs prep: state[:7]={} image_shape={} "
                    "(state_ee preferred={})",
                    st,
                    im,
                    self.prefer_state_ee,
                )
                self._logged_adapt = True

        # Allow env override without changing create_client signature callers
        env_enc = os.environ.get("ILSTUDIO_IMAGE_ENCODING", "").strip().lower()
        if env_enc in ("raw", "jpeg") and self.image_encoding == "auto":
            self._resolved_image_encoding = env_enc
        image_encoding = self._resolve_image_encoding()

        meta_json = _meta_obs_to_jsonable(
            mobs,
            image_encoding=image_encoding,
            jpeg_quality=self.jpeg_quality,
        )
        nbytes, nstr = payload_nbytes_estimate(meta_json)
        if nbytes > 1_500_000:
            logger.warning("Large /inference payload: {} (encoding={})", nstr, image_encoding)
        else:
            logger.debug("/inference payload: {} (encoding={})", nstr, image_encoding)

        payload: Dict[str, Any] = {
            "meta_obs": meta_json,
            "client_id": self.client_id,  # For server-side request deduplication
        }
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if index is not None:
            payload["__index__"] = index
        if reasoning is not None:
            payload["reasoning"] = _to_jsonable(reasoning)
        if timestamp is not None:
            payload["timestamp"] = timestamp

        def _post_once() -> requests.Response:
            return self._session.post(
                f"{self.base_url}/inference",
                json=payload,
                timeout=self.timeout_s,
            )

        def _reset_session() -> None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = requests.Session()
            self._session.verify = False

        # SSH tunnels / proxies often drop keep-alive sockets; retry once.
        try:
            r = _post_once()
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            logger.warning("Inference connection dropped ({}), reconnecting once…", e)
            _reset_session()
            try:
                r = _post_once()
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
                # Large raw payloads often reset the tunnel; if we were on jpeg already,
                # re-raise. If somehow raw, nothing else to try here.
                raise

        # Auto-fallback: jpeg against old server → retry once as raw
        if r.status_code >= 400 and image_encoding == "jpeg":
            logger.warning(
                "JPEG inference failed (HTTP {}), retrying with raw letterboxed images",
                r.status_code,
            )
            self._resolved_image_encoding = "raw"
            payload["meta_obs"] = _meta_obs_to_jsonable(
                mobs,
                image_encoding="raw",
                jpeg_quality=self.jpeg_quality,
            )
            nbytes, nstr = payload_nbytes_estimate(payload["meta_obs"])
            logger.warning("Raw fallback payload: {} (encoding=raw)", nstr)
            _reset_session()
            r = _post_once()

        if r.status_code >= 400:
            detail = (r.text or "")[:500]
            logger.error(
                "Inference HTTP {} from {}: {}",
                r.status_code,
                self.base_url,
                detail,
            )
        r.raise_for_status()
        data = r.json()
        decoded = _from_jsonable(data.get("mact_list"))
        return _normalize_mact_list(decoded)

    def reset(self):
        """Reset internal state (no-op for FastAPI client)."""
        pass

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
