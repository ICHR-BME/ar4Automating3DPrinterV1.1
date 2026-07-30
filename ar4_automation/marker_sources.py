"""Read marker poses in base_link from whichever file a script wants to trust.

Two files hold the same kind of data — an ArUco ID plus its pose in base_link —
but they are produced by different procedures:

  * data/printer_state.json           written by printerAutomation.save_state(),
    i.e. the end of scanFor2Markers.py / scanFor3Markers.py. Entries carry an
    'estimated' flag: True means the pose is only a seed (hand-taught or
    geometric) that the camera never confirmed at a scan pose.
  * data/manual_marker_estimates.json written by teachMarkersByHand.py, where
    the operator drag-teaches the arm until the camera sees each marker. Every
    entry is a real measurement, so there is no flag to filter on.

load_marker_poses() returns both in the same shape, so a script picks its source
with one constant instead of its own JSON parsing.
"""

import json
import os

# repo data/ dir, one level above this package
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

SCAN = 'scan'        # data/printer_state.json      (scanFor*Markers.py)
MANUAL = 'manual'    # data/manual_marker_estimates.json (teachMarkersByHand.py)
NONE = 'none'        # don't read any file

SOURCE_PATHS = {
    SCAN: os.path.join(DATA_DIR, 'printer_state.json'),
    MANUAL: os.path.join(DATA_DIR, 'manual_marker_estimates.json'),
}


def source_path(source):
    """File a source name reads, or the name itself when it's already a path."""
    if source in SOURCE_PATHS:
        return SOURCE_PATHS[source]
    return source


def load_marker_poses(source=SCAN, require_detected=True, log=None):
    """{marker_id: (positionInBase, eulerInBase)} from a marker source.

    source: SCAN, MANUAL, or a path to a file in either format.
    require_detected: drop entries flagged 'estimated' — geometric or
        hand-taught seeds a scan never confirmed. Only the SCAN file carries
        that flag; teachMarkersByHand's entries are measurements and are always
        kept.
    log: optional callable for the one-line summary (e.g. node.get_logger().info).

    Returns {} (never raises) when the source is NONE, missing, or unreadable —
    callers fall back to their own poses.
    """
    def _say(msg):
        if log:
            log(msg)

    if not source or source == NONE:
        return {}

    path = source_path(source)
    if not os.path.isfile(path):
        _say(f"marker source {path} does not exist")
        return {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        _say(f"could not read marker source {path}: {e}")
        return {}

    out, skipped = {}, []
    for m in data.get('markers', []):
        if require_detected and m.get('estimated', False):
            skipped.append(int(m['id']))
            continue
        out[int(m['id'])] = (m['positionInBase'], m['eulerInBase'])

    _say(f"marker source {os.path.basename(path)}: ids {sorted(out)}"
         + (f" (skipped {sorted(skipped)} — estimated, never scanned)" if skipped else ""))
    return out
