#!/usr/bin/env bash
set -euo pipefail

# Start a 2.4 GHz hotspot on one adapter while leaving another adapter for upstream Wi-Fi.
# Defaults match this machine's current setup, but can be overridden with flags.

SSID="Kevin-Hotspot2"
PASSWORD="1133557799"
HOTSPOT_IFACE="wlx9cefd5f6cfd0"
UPLINK_IFACE="wlp2s0"
BAND="bg"

usage() {
  cat <<'EOF'
Usage:
  start_hotspot.sh [options]

Options:
  -s, --ssid <name>          Hotspot SSID (default: AR4-Hotspot24)
  -p, --password <pass>      Hotspot password (8-63 chars)
  -i, --iface <iface>        Hotspot adapter interface (default: wlx9cefd5f6cfd0)
  -u, --uplink <iface>       Upstream Wi-Fi interface to keep connected (default: wlp2s0)
  -b, --band <band>          Band for hotspot (default: bg for 2.4 GHz)
      --show-password        Show the generated hotspot credentials after start
  -h, --help                 Show this help

Examples:
  ./scripts/start_hotspot.sh
  ./scripts/start_hotspot.sh --ssid MyPrinter --password SuperSecret123
  ./scripts/start_hotspot.sh -i wlx9cefd5f6cfd0 -u wlp2s0
EOF
}

SHOW_PASSWORD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--ssid)
      SSID="${2:-}"
      shift 2
      ;;
    -p|--password)
      PASSWORD="${2:-}"
      shift 2
      ;;
    -i|--iface)
      HOTSPOT_IFACE="${2:-}"
      shift 2
      ;;
    -u|--uplink)
      UPLINK_IFACE="${2:-}"
      shift 2
      ;;
    -b|--band)
      BAND="${2:-}"
      shift 2
      ;;
    --show-password)
      SHOW_PASSWORD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v nmcli >/dev/null 2>&1; then
  echo "Error: nmcli is required but not installed." >&2
  exit 1
fi

if [[ ${#PASSWORD} -lt 8 || ${#PASSWORD} -gt 63 ]]; then
  echo "Error: password must be 8-63 characters." >&2
  exit 1
fi

if ! ip link show "$HOTSPOT_IFACE" >/dev/null 2>&1; then
  echo "Error: hotspot interface '$HOTSPOT_IFACE' not found." >&2
  exit 1
fi

if ! ip link show "$UPLINK_IFACE" >/dev/null 2>&1; then
  echo "Error: uplink interface '$UPLINK_IFACE' not found." >&2
  exit 1
fi

UPLINK_STATE=$(nmcli -t -f DEVICE,TYPE,STATE device status | awk -F: -v i="$UPLINK_IFACE" '$1==i {print $3}')
if [[ "$UPLINK_STATE" != "connected" ]]; then
  echo "Warning: uplink interface '$UPLINK_IFACE' is not currently connected." >&2
fi

echo "Starting 2.4 GHz hotspot on $HOTSPOT_IFACE (SSID: $SSID)..."
nmcli device wifi hotspot ifname "$HOTSPOT_IFACE" ssid "$SSID" band "$BAND" password "$PASSWORD"

# Pin the hotspot to plain WPA2 (PMF/802.11w disabled). Without this,
# NetworkManager runs the AP in WPA2/WPA3-SAE transition mode, which Bambu
# printers associate to and then drop within seconds (repeating flap).
HOTSPOT_CON=$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v i="$HOTSPOT_IFACE" '$2==i{print $1}')
if [[ -n "$HOTSPOT_CON" ]]; then
  CURRENT_PMF=$(nmcli -g 802-11-wireless-security.pmf connection show "$HOTSPOT_CON")
  if [[ "$CURRENT_PMF" != "1" ]]; then
    echo "Disabling PMF/WPA3 on '$HOTSPOT_CON' (Bambu compatibility)..."
    nmcli connection modify "$HOTSPOT_CON" 802-11-wireless-security.pmf disable
    nmcli connection up "$HOTSPOT_CON"
  fi
fi

echo
echo "Active connections:"
nmcli -t -f NAME,TYPE,DEVICE connection show --active

echo
echo "Device status:"
nmcli device status

echo
echo "Hotspot started."
if [[ "$SHOW_PASSWORD" -eq 1 ]]; then
  nmcli dev wifi show-password ifname "$HOTSPOT_IFACE"
else
  echo "Use '--show-password' to print credentials from NetworkManager."
fi
