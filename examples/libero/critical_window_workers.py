"""Bounded spawn workers: one simulator/client per process, coordinator owns records.

Workers create their own simulator/websocket. A separate pipe per worker
and process sentinels detect native crashes (including fatal GL exits).
Python 3.8 compatible; no shared random-number state or inherited GL contexts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing as mp
from multiprocessing.connection import wait
import signal
import traceback
import urllib.parse


class WorkerError(RuntimeError):
    def __init__(self, key, detail):
        self.key = key
        super().__init__(f"Simulation worker failed for {key}: {detail}")


def server_address(host, port):
    host = host if host.startswith(("ws://", "wss://")) else "ws://" + host
    return f"{host}:{port}"


def parse_servers(servers, host="localhost", port=8000):
    """Return normalized (scheme://host, port) endpoints, with no credentials/path."""
    if servers.strip():
        entries = servers.split(",")
    else:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Policy server port must be an integer from 1 to 65535")
        entries = [server_address(host, port)]
    endpoints = []
    for entry in entries:
        value = entry.strip()
        try:
            parsed = urllib.parse.urlsplit(value if "://" in value else "ws://" + value)
            number, name = parsed.port, parsed.hostname
            if (
                not value
                or any(char.isspace() for char in value)
                or parsed.scheme not in ("ws", "wss")
                or not name
                or number is None
                or number < 1
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Invalid policy server address")
        except ValueError as error:
            raise ValueError(
                "Each --servers entry must be host:port or ws[s]://host:port, without credentials/path/query"
            ) from error
        authority = f"[{name}]" if ":" in name else name
        endpoints.append((f"{parsed.scheme}://{authority}", number))
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Duplicate policy server addresses; list each server once and use --workers for parallelism")
    return endpoints


def create_client(host, port):
    from openpi_client import websocket_client_policy  # noqa: PLC0415 - keep offline planning dependency-free.

    return websocket_client_policy.WebsocketClientPolicy(host, port)


def run_rollout_job(job, client, episode_runner):
    kwargs = dict(job["episode_kwargs"])
    windows, delay = job["windows"], job["delay_steps"]
    return episode_runner(
        **kwargs,
        client=client,
        delay_schedule=lambda step: delay if any(a <= step < b for a, b in windows) else 0,
    )


def _worker_loop(connection, host, port, metadata, episode_runner, client_factory):
    # Ctrl+C is handled by the coordinator, which terminates and joins the pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO)
    client, key = None, None
    try:
        client = client_factory(host, port)
        if json.loads(json.dumps(client.get_server_metadata())) != metadata:
            raise ValueError("Worker policy-server metadata differs from the coordinator")
        if not callable(getattr(client, "infer_with_noise", None)):
            raise ValueError("Worker requires an explicit-noise policy client")
        connection.send(("ready", None, None))
        while True:
            message = connection.recv()
            if message is None:
                break
            key, job = message
            result = run_rollout_job(job, client, episode_runner)
            result["policy_server"] = server_address(host, port)
            connection.send(("ok", key, result))
            key = None
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        with contextlib.suppress(EOFError, BrokenPipeError, OSError):
            connection.send(("error", key, f"Server {server_address(host, port)}:\n{traceback.format_exc()}"))
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client._ws.close()  # noqa: SLF001 - current client has no public close().
        connection.close()


class RolloutPool:
    """Persistent processes, at most one in-flight episode per worker.

    iter_results() is consumed on the coordinator thread. Do not let workers
    mutate manifests/registries or pass an existing environment/client here.
    """

    def __init__(
        self, workers, host, port, metadata, episode_runner, client_factory=create_client, *, server_endpoints=None
    ):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        endpoints = parse_servers("", host, port) if server_endpoints is None else list(server_endpoints)
        if not endpoints or workers < len(endpoints):
            raise ValueError("--workers must be at least the number of --servers addresses")
        self.workers = []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for index in range(workers):
                worker_host, worker_port = endpoints[index % len(endpoints)]
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker_loop,
                    args=(
                        child,
                        worker_host,
                        worker_port,
                        json.loads(json.dumps(metadata)),
                        episode_runner,
                        client_factory,
                    ),
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.workers.append(
                    {
                        "process": process,
                        "connection": parent,
                        "key": None,
                        "server": server_address(worker_host, worker_port),
                    }
                )
            # Complete every handshake before the first inference can block the
            # server event loop (especially the initial JIT compilation).
            self._wait_until_ready()
        except BaseException:
            self.close(terminate=True)
            raise

    def _wait_until_ready(self):
        pending = list(self.workers)
        while pending:
            ready = wait([row["connection"] for row in pending] + [row["process"].sentinel for row in self.workers])
            for row in list(pending):
                if row["connection"] not in ready:
                    continue
                try:
                    status, key, detail = row["connection"].recv()
                except (EOFError, BrokenPipeError, OSError) as error:
                    raise WorkerError(None, f"Worker for {row['server']} exited during startup") from error
                if status != "ready":
                    raise WorkerError(key, detail)
                pending.remove(row)
            for row in self.workers:
                if row["process"].sentinel in ready:
                    row["process"].join(timeout=0)
                    raise WorkerError(
                        None, f"Worker for {row['server']} exited during startup: exitcode={row['process'].exitcode}"
                    )

    @property
    def pids(self):
        return [row["process"].pid for row in self.workers]

    @property
    def assignments(self):
        return [
            {"worker": index + 1, "pid": row["process"].pid, "server": row["server"]}
            for index, row in enumerate(self.workers)
        ]

    def iter_results(self, jobs):
        """Yield (key, raw_episode_record); bound queued jobs by worker count."""
        if self.closed:
            raise RuntimeError("Worker pool is closed")
        if any(row["key"] is not None for row in self.workers):
            raise RuntimeError("Previous worker stage has not finished")
        jobs, exhausted = iter(jobs), False
        try:
            while True:
                for row in self.workers:
                    if row["key"] is not None or exhausted:
                        continue
                    try:
                        key, job = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    row["key"] = key
                    try:
                        row["connection"].send((key, job))
                    except (EOFError, BrokenPipeError, OSError) as error:
                        raise WorkerError(key, f"{row['server']}: {error!r}") from error
                busy = [row for row in self.workers if row["key"] is not None]
                if not busy:
                    break
                ready = wait([row["connection"] for row in busy] + [row["process"].sentinel for row in self.workers])
                # Read completed messages before reacting to their process sentinel.
                for row in busy:
                    if row["connection"] not in ready:
                        continue
                    try:
                        status, key, result = row["connection"].recv()
                    except (EOFError, BrokenPipeError, OSError) as error:
                        raise WorkerError(
                            row["key"], f"Worker for {row['server']} exited without a result (possible GL/native crash)"
                        ) from error
                    if key != row["key"]:
                        raise WorkerError(row["key"], "Worker response belongs to a different trial")
                    if status != "ok":
                        raise WorkerError(key, result)
                    row["key"] = None
                    yield key, result
                for row in self.workers:
                    if row["process"].sentinel in ready:
                        row["process"].join(timeout=0)
                        raise WorkerError(
                            row["key"],
                            f"Process for {row['server']} exited unexpectedly: exitcode={row['process'].exitcode}",
                        )
        except BaseException:
            self.close(terminate=True)
            raise

    def close(self, *, terminate=False):
        if self.closed:
            return
        self.closed = True
        for row in self.workers:
            process = row["process"]
            if terminate or row["key"] is not None:
                if process.is_alive():
                    process.terminate()
            else:
                with contextlib.suppress(EOFError, BrokenPipeError, OSError):
                    row["connection"].send(None)
        for row in self.workers:
            process = row["process"]
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            row["connection"].close()
            if not process.is_alive():
                process.close()
