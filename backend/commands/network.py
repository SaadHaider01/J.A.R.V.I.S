import subprocess
import socket
import logging

logger = logging.getLogger("JARVIS.Network")


def _run_netsh(args: list[str]) -> str:
    """
    Runs a netsh command and returns stdout.

    Why netsh? Windows exposes all network configuration through netsh
    (Network Shell). It's the built-in CLI that every Windows version ships
    with, so we don't need any third-party library — just subprocess calls.
    """
    try:
        result = subprocess.run(
            ["netsh"] + args,
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"netsh command failed: {e}")
        return ""


def get_wifi_status() -> dict:
    """
    Returns the current WiFi connection status including:
    - connected (bool)
    - ssid (network name)
    - signal strength
    """
    output = _run_netsh(["wlan", "show", "interfaces"])
    if not output:
        return {"connected": False, "ssid": "", "signal": ""}

    info = {}
    for line in output.splitlines():
        line = line.strip()
        if "SSID" in line and "BSSID" not in line:
            info["ssid"] = line.split(":", 1)[1].strip()
        elif "Signal" in line:
            info["signal"] = line.split(":", 1)[1].strip()
        elif "State" in line:
            info["connected"] = "connected" in line.lower()

    if "connected" not in info:
        info["connected"] = False
    if "ssid" not in info:
        info["ssid"] = ""
    if "signal" not in info:
        info["signal"] = ""

    logger.info(f"WiFi status: {info}")
    return info


def list_wifi_networks() -> list[str]:
    """Scans and returns a list of available WiFi network names (SSIDs)."""
    output = _run_netsh(["wlan", "show", "networks"])
    networks = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("SSID") and "BSSID" not in line:
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                networks.append(parts[1].strip())
    logger.info(f"Found {len(networks)} WiFi networks")
    return networks


def connect_wifi(ssid: str, password: str = "") -> bool:
    """
    Connects to a WiFi network by name.

    This creates a temporary XML profile and feeds it to netsh.
    If the network was previously connected, Windows remembers the password
    and we can connect with just the profile name.
    """
    try:
        # Try connecting with an existing saved profile first
        result = subprocess.run(
            ["netsh", "wlan", "connect", f"name={ssid}"],
            capture_output=True, text=True, timeout=15
        )
        if "successfully" in result.stdout.lower():
            logger.info(f"Connected to WiFi: {ssid}")
            return True

        # If that fails and a password was provided, create a fresh profile
        if password:
            import tempfile
            import os
            profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM><security>
        <authEncryption><authentication>WPA2PSK</authentication>
        <encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
        <sharedKey><keyType>passPhrase</keyType><protected>false</protected>
        <keyMaterial>{password}</keyMaterial></sharedKey>
    </security></MSM>
</WLANProfile>"""
            # Write profile to temp file, add it, connect, clean up
            profile_path = os.path.join(tempfile.gettempdir(), f"jarvis_wifi_{ssid}.xml")
            with open(profile_path, "w") as f:
                f.write(profile_xml)

            subprocess.run(
                ["netsh", "wlan", "add", "profile", f"filename={profile_path}"],
                capture_output=True, timeout=10
            )
            result = subprocess.run(
                ["netsh", "wlan", "connect", f"name={ssid}"],
                capture_output=True, text=True, timeout=15
            )
            os.remove(profile_path)

            if "successfully" in result.stdout.lower():
                logger.info(f"Connected to WiFi: {ssid}")
                return True

        logger.warning(f"Failed to connect to WiFi: {ssid}")
        return False
    except Exception as e:
        logger.error(f"WiFi connection error: {e}")
        return False


def get_ip_address() -> dict:
    """
    Returns both the local (LAN) IP and the public (WAN) IP.

    Local IP: We open a dummy UDP socket to Google's DNS (8.8.8.8) — this
    doesn't actually send data but tells us which local interface would be
    used, revealing our LAN address.

    Public IP: We hit a lightweight API that just echoes back our IP.
    """
    info = {"local_ip": "", "public_ip": ""}

    # Local IP via dummy socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["local_ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["local_ip"] = "unknown"

    # Public IP via external API
    try:
        import urllib.request
        response = urllib.request.urlopen("https://api.ipify.org", timeout=5)
        info["public_ip"] = response.read().decode("utf-8")
    except Exception:
        info["public_ip"] = "unknown"

    logger.info(f"IP addresses: {info}")
    return info
