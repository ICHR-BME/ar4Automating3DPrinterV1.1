#!/usr/bin/env python3
"""Shared link between the website computer and the robot computer.

Keep the two copies identical -- copy, don't edit twice:
    ar4_ws/src/ar4Automating3DPrinter/robot_link.py   (robot computer)
    UCDavisGUI/robot_link.py                          (website computer)
robot_agent.py imports the config below and serves the API; the website
imports the functions and drives it. If the copies do drift, PROTOCOL_VERSION
catches it on the next ping() instead of letting a command do the wrong thing.

    from robot_link import send_command, wait_for_result, ping, RobotError

    ping()                                   # is the robot computer up?
    receipt = send_command("scan_marker", {"marker_id": 1})
    result = wait_for_result(receipt["id"])  # blocks until the move ends
    result["state"]                          # "done" | "failed"

    result = send_command("echo", {"message": "hi"}, wait=True)   # short cmds

Link failures raise RobotError rather than returning an error dict, so they
can't be mistaken for success.

The wire protocol -- JSON bodies, X-Auth-Token on every request:

    POST /command   {"name": "scan_marker", "params": {"marker_id": 1}}
      202  {"received": true, "id": 4, "name": ..., "params": ...}
      400  bad body                  404  unknown command
      401  bad token                 409  the arm is already busy

    GET /command/<id>
      200  {"id", "name", "state": "running"|"done"|"failed",
            "result", "error", "duration_s", ...}
      404  unknown id

    GET /ping
      200  {"ok", "protocol", "robot", "dry_run", "busy", "current",
            "history": [...last 10 commands...],
            "commands": [{"name", "doc", "moves",
                          "params": [{"name", "default", "type"}]}]}
      dry_run means the agent prints commands instead of moving the arm, worth
      showing in the UI. commands carries each handler's real signature, so the
      forms are built from the robot itself.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

# Bump on any wire-format change; ping() refuses a mismatched agent.
PROTOCOL_VERSION = 3

# ---- config (edit these; no CLI args) ------------------------------------
# Where the client looks for the robot computer. "127.0.0.1" while both halves
# run on one machine; otherwise the robot's hostname (a name, not 100.x.y.z or
# a DHCP address -- it survives changing networks).
ROBOT_HOST = "127.0.0.1"
ROBOT_PORT = 8420

# Interfaces the agent listens on, which is a separate question from the one
# above: 0.0.0.0 is every interface, and is not an address anything connects
# to. "127.0.0.1" here would refuse every device except this computer.
BIND_HOST = "0.0.0.0"

# Shared secret. With BIND_HOST open to the network this is the only thing
# between another device on that wifi and the arm, so change it on both
# machines rather than leaving the default.
AUTH_TOKEN = "change-me"

# Per-request timeout, not the length of a move -- results are polled.
REQUEST_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.5      # how often wait_for_result re-checks


class RobotError(Exception):
    """Agent unreachable, command refused, or the protocol versions disagree."""


# ==========================================================================
# Client
# ==========================================================================

def _request(path: str, payload: dict | None = None) -> dict:
    """One HTTP round trip. POST when a payload is given, else GET."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"http://{ROBOT_HOST}:{ROBOT_PORT}{path}",
        data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json",
                 "X-Auth-Token": AUTH_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # the agent refused; its own reason beats "HTTP 409"
        raise RobotError(json.loads(e.read()).get("error") or f"HTTP {e.code}") \
            from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise RobotError(
            f"cannot reach the robot agent at {ROBOT_HOST}:{ROBOT_PORT} ({e}). "
            f"Is robot_agent.py running on the robot computer?") from None


def ping() -> dict:
    """Liveness check, and the cheapest way to see which commands the robot
    accepts."""
    status = _request("/ping")
    remote = status.get("protocol")
    if remote != PROTOCOL_VERSION:
        raise RobotError(
            f"protocol mismatch: this machine speaks v{PROTOCOL_VERSION}, the "
            f"agent v{remote}. The two copies of robot_link.py have drifted -- "
            f"copy the newer one across and restart both.")
    return status


def send_command(name: str, params: dict | None = None, *,
                 wait: bool = False, timeout: float | None = None) -> dict:
    """Run `name` on the robot, with `params` as keyword arguments to the
    matching handler in robot_agent.py. Returns the agent's own receipt as soon
    as it accepts the command:

        {"received": True, "id": 4, "name": "scan_marker", "params": {...}}

    Raises RobotError if the agent is unreachable, the command is unknown, or
    the arm is busy. wait=True instead blocks for the result; `timeout` bounds
    that wait but cancels nothing -- the robot keeps going either way.
    """
    receipt = _request("/command", {"name": name, "params": params or {}})
    if not wait:
        return receipt
    return wait_for_result(receipt["id"], timeout=timeout)


def command_status(command_id: int) -> dict:
    """One non-blocking look at a command, in whatever state it is. This is
    what the web UI polls: it renders progress rather than blocking."""
    return _request(f"/command/{command_id}")


def wait_for_result(command_id: int, *, timeout: float | None = None) -> dict:
    """Poll until the command finishes, then return its record:

        {"id": 4, "name": "scan_marker", "state": "done"|"failed",
         "result": <handler return value>, "error": "", "duration_s": 12.4}

    A move that failed on the robot returns normally with state="failed"; it is
    an outcome to display, not an exception.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = command_status(command_id)
        if record.get("state") in ("done", "failed"):
            return record
        if deadline is not None and time.monotonic() > deadline:
            raise RobotError(
                f"command {command_id} ({record.get('name')}) has not finished "
                f"after {timeout}s -- it is still running on the robot")
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    # End-to-end check against a running robot_agent.py (start it first, in
    # another terminal). Safe under DRY_RUN: the motion commands are printed by
    # the agent rather than executed.
    status = ping()
    mode = "TESTING (dry run)" if status.get("dry_run") else "LIVE -- WILL MOVE"
    print(f"agent up: robot={status['robot']}  mode={mode}  busy={status['busy']}")
    for spec in status["commands"]:
        args = ", ".join(f"{p['name']}={p['default']}" for p in spec["params"])
        print(f"  {spec['name']}({args})")
    print()

    print("-- round trip, no hardware --")
    print("  ", send_command("echo", {"message": "hello"}, wait=True)["result"])

    print("-- receipt now, result later --")
    receipt = send_command("slow_task", {"seconds": 2})
    print("   receipt:", {k: receipt[k] for k in ("received", "id", "name")})
    print("   result :", wait_for_result(receipt["id"])["result"])

    print("-- structured return value --")
    print("  ", send_command("list_markers", wait=True)["result"])

    print("-- motion commands --")
    for name, params in (("go_home", {"velocity_scaling": 0.2}),
                         ("scan_marker", {"marker_id": 1}),
                         ("scrape_plate", {"source_id": 2, "scrape_id": 1})):
        outcome = send_command(name, params, wait=True)
        print(f"   {name:12s} {outcome['state']:6s} {outcome['result']}")

    print("-- refusals --")
    for label, call in (
            ("unknown command", lambda: send_command("launch_missiles")),
            ("bad params     ", lambda: send_command("echo", {"nope": 1}, wait=True)),
    ):
        try:
            result = call()
            print(f"   {label}: {result.get('state')} / {result.get('error')}")
        except RobotError as e:
            print(f"   {label}: refused -- {e}")

    print("\nlink OK.")
