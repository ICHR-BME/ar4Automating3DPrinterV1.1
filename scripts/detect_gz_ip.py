#!/usr/bin/env python3
"""Print the IP gz-transport should bind on this host, or nothing when the
default is fine. Used by the launchVirtual*.sh scripts:

    GZ_IP_AUTO=$(python3 "$(dirname "$0")/detect_gz_ip.py")
    [ -n "$GZ_IP_AUTO" ] && export GZ_IP="$GZ_IP_AUTO"

gz binds its sockets to the first non-loopback interface; on some networks
host-local traffic on that interface is dropped (seen with wifi handing out
CGNAT 100.64/10 addresses while Tailscale's firewall claims the same range)
and then every gz service call hangs — the sim spins on "Requesting list of
world names" and the robot never spawns. Instead of hardcoding loopback (or
hunting for that one culprit), empirically test the interface gz would pick:
bind a listening socket to it and try to connect to ourselves. Only when that
self-connection fails do we print 127.0.0.1; on hosts where the default
interface is healthy we print nothing and gz behaves exactly as stock.

Standalone on purpose (no ROS/ar4_automation imports) so the launch scripts
can run it before anything is sourced. ar4_automation.runner_common's
ensure_gz_ip() applies the SAME logic for the python entry scripts — keep the
two in step if this changes.
"""
import socket

# The IP gz-transport would choose: source address of the default route.
# Connecting a UDP socket sends no packets; it just resolves routing.
try:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.connect(('8.8.8.8', 53))
    candidate = probe.getsockname()[0]
    probe.close()
except OSError:
    raise SystemExit(0)  # no route at all; gz falls back to loopback itself

try:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((candidate, 0))
    srv.listen(1)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.settimeout(1.0)
    cli.connect((candidate, srv.getsockname()[1]))
    cli.close()
    srv.close()
except OSError:
    print('127.0.0.1')
