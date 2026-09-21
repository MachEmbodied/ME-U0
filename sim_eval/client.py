"""WebSocket policy client for remote model inference.

No torch/CUDA dependency — only numpy, websockets, msgpack.

Usage::

    from sim_eval.client import PolicyClient
    client = PolicyClient("localhost", 8765)
    actions = client.predict_action(pixel_values=img_array, instruction="pick up the cup")
    client.close()
"""

from __future__ import annotations

import logging
import time

import numpy as np
from websockets.sync.client import connect

from sim_eval.transport import decode, encode

logger = logging.getLogger(__name__)


class PolicyClient:
    """Synchronous WebSocket client for remote action prediction."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        timeout: float = 300.0,
        max_retries: int = 5,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries
        self._ws = None

    def _connect(self) -> None:
        """Establish WebSocket connection with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                self._ws = connect(
                    self.uri,
                    max_size=100 * 1024 * 1024,  # 100 MB
                    open_timeout=self.timeout,
                    close_timeout=self.timeout,
                )
                logger.info("Connected to %s", self.uri)
                return
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Connection attempt %d/%d failed (%s), retrying in %ds...",
                    attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)
        raise ConnectionError(
            f"Failed to connect to {self.uri} after {self.max_retries} attempts"
        )

    def predict_action(self, **kwargs) -> np.ndarray:
        """Send observation payload and receive predicted actions.

        Raises:
            :class:`websockets.exceptions.ConnectionClosed` and friends are
            re-raised unchanged if the server has gone away mid-session —
            callers (``sim_eval.base_env._run_episode``) catch these and
            abort the worker loop cleanly rather than spamming the same
            error on every episode.

        Args:
            **kwargs: Observation dict (np.ndarray values and/or strings).

        Returns:
            Action array from the remote model.
        """
        from websockets.exceptions import ConnectionClosed  # local import — websockets lib is optional until used

        if self._ws is None:
            self._connect()

        request = encode({"payload": kwargs})
        try:
            self._ws.send(request)
            raw_response = self._ws.recv(timeout=self.timeout)
        except ConnectionClosed:
            # Server went away. Drop the stale socket so future calls don't
            # keep spamming the same error via the dead handle.
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            raise
        response = decode(raw_response)

        if response.get("status") != "ok":
            raise RuntimeError(
                f"Server error: {response.get('message', 'unknown error')}"
            )

        return response["actions"]

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
