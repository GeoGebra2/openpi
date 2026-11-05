import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        return self._request(obs)

    def infer_with_noise(self, observation: Dict, noise) -> Dict:  # noqa: UP006
        """Run the ordinary policy query with an explicit initial noise sample."""
        return self._request(
            {
                "_openpi_method": "infer_action",
                "arguments": {"obs": observation, "noise": noise},
            }
        )

    def infer_drift(
        self,
        observation: Dict,
        reference_observation: Dict,
        noise,
        *,
        check_consistency: bool = False,
        consistency_atol: float = 1e-4,
        consistency_rtol: float = 1e-4,
    ) -> Dict:
        """Paired diagnostics with shared initial noise; requires a schema-v2 diagnostics server."""
        if self._server_metadata.get("activation_drift", {}).get("schema_version") != 2:
            raise RuntimeError("Update the policy server and restart it with --diagnostics (schema v2 required)")
        return self._request(
            {
                "_openpi_method": "infer_drift_v2",
                "arguments": {
                    "observation": observation,
                    "reference_observation": reference_observation,
                    "noise": noise,
                    "check_consistency": check_consistency,
                    "consistency_atol": consistency_atol,
                    "consistency_rtol": consistency_rtol,
                },
            }
        )

    def infer_attention(self, observation: Dict, noise=None) -> Dict:  # noqa: UP006
        """Run one diagnostic query and return grouped action-token attention."""
        if self._server_metadata.get("attention", {}).get("schema_version") != 1:
            raise RuntimeError(
                "Update the policy server and restart it with --diagnostics "
                "(attention schema v1 required)"
            )
        return self._request(
            {
                "_openpi_method": "infer_attention",
                "arguments": {"observation": observation, "noise": noise},
            }
        )

    def infer_token_drift(
        self, observation: Dict, reference_observation: Dict, noise, *, camera: str = "left_wrist_0_rgb"
    ) -> Dict:
        """Return compact spatial SigLIP token drift for one camera."""
        if self._server_metadata.get("token_drift", {}).get("schema_version") != 1:
            raise RuntimeError("Update the policy server/client for token_drift schema v1 and restart the server")
        return self._request({
            "_openpi_method": "infer_token_drift",
            "arguments": {
                "observation": observation,
                "reference_observation": reference_observation,
                "noise": noise,
                "camera": camera,
            },
        })

    def infer_wrist_patch(
        self, observation: Dict, current_wrist_image, noise, mode: str, layers, site: str = "wrist_kv"
    ) -> Dict:
        """Only the returned recipient action is intended for execution."""
        if self._server_metadata.get("wrist_kv_patch", {}).get("schema_version") != 1:
            raise RuntimeError("Update the policy server/client for wrist KV patch schema v1 and restart the server")
        return self._request({
            "_openpi_method": "infer_wrist_patch",
            "arguments": {
                "observation": observation, "current_wrist_image": current_wrist_image,
                "noise": noise, "mode": mode, "layers": list(layers), "site": site,
            },
        })

    def infer_token_patch(
        self, observation: Dict, donor_image, noise, *, mode: str, layers,
        site: str = "siglip", camera: str = "left_wrist_0_rgb", token_mask=None,
    ) -> Dict:
        """Replace only selected image tokens with donor-camera activations."""
        if self._server_metadata.get("token_patch", {}).get("schema_version") != 1:
            raise RuntimeError("Update the policy server/client for token_patch schema v1 and restart the server")
        if self._server_metadata.get("wrist_kv_patch", {}).get("schema_version") != 1:
            raise RuntimeError("Update server/client for wrist KV patch schema v1 and restart the server")
        return self._request({
            "_openpi_method": "infer_token_patch",
            "arguments": {
                "observation": observation, "donor_image": donor_image,
                "noise": noise, "mode": mode, "layers": list(layers), "site": site,
                "camera": camera, "token_mask": token_mask,
            },
        })

    def _request(self, request: Dict) -> Dict:
        data = self._packer.pack(request)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
