"""Read named group_states out of the LIVE SRDF, so poses live in the robot
description instead of in this repo.

Home is defined once, as a `home` group_state in the robot's SRDF:
  * annin_ar4_moveit_config/srdf/ar_macro.srdf.xacro        (ar4)
  * xarm_moveit_config/srdf/_xarm6_macro.srdf.xacro         (xarm6)
  * xarm_moveit_config/srdf/_lite6_macro.srdf.xacro         (lite6)
RViz offers exactly those states in its goal-state dropdown, and the matching
initial_value params in each description's ros2_control xacro make the sim spawn
there — so go_home, RViz and the spawn pose cannot drift apart.

The SRDF is not a topic: move_group holds it in its `robot_description_semantic`
parameter. That parameter is fetched over rcl_interfaces/GetParameters, from
whichever node ends in 'move_group' (xarm launches namespace it, e.g.
/ufactory/move_group, so the node is discovered rather than hardcoded).
"""

import time
import xml.etree.ElementTree as ET

from rcl_interfaces.srv import GetParameters

SEMANTIC_PARAMETER = 'robot_description_semantic'


def group_state_positions(node, group, state_name, joint_names, timeout=10.0):
    """Joint values of `state_name` for `group`, ordered like joint_names.

    Returns None (having logged why) if the SRDF cannot be read, the state is
    missing, or the state does not cover every joint in joint_names — callers
    must treat that as "do not move" rather than fall back to a guess, since the
    pose being read is the one that keeps the tool off the floor.
    """
    srdf = fetch_srdf(node, timeout=timeout)
    if srdf is None:
        return None
    return parse_group_state(node, srdf, group, state_name, joint_names)


def fetch_srdf(node, timeout=10.0):
    """The live SRDF XML string from move_group's parameter, or None."""
    deadline = time.time() + timeout
    service_name = None
    while time.time() < deadline and service_name is None:
        service_name = _find_move_group_parameter_service(node)
        if service_name is None:
            time.sleep(0.25)
    if service_name is None:
        node.get_logger().error(
            "SRDF: no move_group node found — cannot read the named home state. "
            "Is the MoveIt launch up?")
        return None

    client = node.create_client(GetParameters, service_name,
                                callback_group=getattr(node, '_cb_group', None))
    try:
        remaining = max(0.5, deadline - time.time())
        if not client.wait_for_service(timeout_sec=remaining):
            node.get_logger().error(f"SRDF: {service_name} never became available.")
            return None
        # call_async + poll, never a blocking .call(): a blocking service call
        # from a procedure thread deadlocks against the background executor
        future = client.call_async(
            GetParameters.Request(names=[SEMANTIC_PARAMETER]))
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done() or future.result() is None:
            node.get_logger().error(f"SRDF: {service_name} call timed out.")
            return None
        values = future.result().values
        if not values or not values[0].string_value:
            node.get_logger().error(
                f"SRDF: move_group reported an empty '{SEMANTIC_PARAMETER}'.")
            return None
        return values[0].string_value
    finally:
        node.destroy_client(client)


def parse_group_state(node, srdf_xml, group, state_name, joint_names):
    """Pull one group_state out of an SRDF string. See group_state_positions."""
    try:
        root = ET.fromstring(srdf_xml)
    except ET.ParseError as exc:
        node.get_logger().error(f"SRDF: unparseable XML ({exc}).")
        return None

    for state in root.findall('group_state'):
        if state.get('name') != state_name or state.get('group') != group:
            continue
        values = {j.get('name'): float(j.get('value'))
                  for j in state.findall('joint') if j.get('value') is not None}
        missing = [n for n in joint_names if n not in values]
        if missing:
            node.get_logger().error(
                f"SRDF: group_state '{state_name}' of group '{group}' does not set "
                f"{missing}; refusing to guess those joints.")
            return None
        return [values[n] for n in joint_names]

    available = sorted({s.get('name') for s in root.findall('group_state')
                        if s.get('group') == group})
    node.get_logger().error(
        f"SRDF: no group_state '{state_name}' for group '{group}'. "
        f"Available: {available}")
    return None


def _find_move_group_parameter_service(node):
    """'<ns>/move_group/get_parameters' for the running move_group, or None."""
    for name, namespace in node.get_node_names_and_namespaces():
        if name != 'move_group':
            continue
        prefix = namespace.rstrip('/')
        return f"{prefix}/{name}/get_parameters"
    return None
