import logging

logger = logging.getLogger("JARVIS.SystemInfo")


def get_battery_status() -> dict:
    """
    Returns battery percentage and charging state using psutil.

    psutil reads from the Windows kernel's SYSTEM_POWER_STATUS struct,
    which is the same data the taskbar battery icon uses.
    """
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is None:
            return {"percent": -1, "charging": False, "plugged_in": False,
                    "message": "No battery detected (desktop PC?)"}
        info = {
            "percent": round(battery.percent),
            "plugged_in": battery.power_plugged,
            "charging": battery.power_plugged and battery.percent < 100,
        }
        logger.info(f"Battery: {info}")
        return info
    except Exception as e:
        logger.error(f"Failed to read battery: {e}")
        return {"percent": -1, "charging": False, "plugged_in": False,
                "message": str(e)}


def get_cpu_usage() -> float:
    """
    Returns the current CPU usage as a percentage.
    The interval=1 means it samples over 1 second for accuracy.
    """
    try:
        import psutil
        usage = psutil.cpu_percent(interval=1)
        logger.info(f"CPU usage: {usage}%")
        return usage
    except Exception as e:
        logger.error(f"Failed to read CPU usage: {e}")
        return -1.0


def get_ram_usage() -> dict:
    """Returns RAM usage statistics: total, used, available, and percentage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        info = {
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "used_gb": round(mem.used / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
            "percent": mem.percent,
        }
        logger.info(f"RAM: {info}")
        return info
    except Exception as e:
        logger.error(f"Failed to read RAM usage: {e}")
        return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0}


def list_processes(top_n: int = 10) -> list[dict]:
    """
    Returns the top N most resource-hungry processes sorted by CPU usage.
    Each entry includes the process name, PID, CPU%, and memory in MB.
    """
    try:
        import psutil
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                info = proc.info
                procs.append({
                    "name": info['name'],
                    "pid": info['pid'],
                    "cpu_percent": info['cpu_percent'] or 0,
                    "memory_mb": round(info['memory_info'].rss / (1024 ** 2), 1)
                        if info['memory_info'] else 0,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage descending, then by memory
        procs.sort(key=lambda p: (p['cpu_percent'], p['memory_mb']), reverse=True)
        top = procs[:top_n]
        logger.info(f"Top {top_n} processes retrieved")
        return top
    except Exception as e:
        logger.error(f"Failed to list processes: {e}")
        return []


def kill_process_by_name(name: str) -> bool:
    """
    Kills all processes matching the given name.
    More reliable than taskkill because psutil handles elevated processes better.
    """
    try:
        import psutil
        killed = 0
        name_lower = name.lower()
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] and name_lower in proc.info['name'].lower():
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if killed > 0:
            logger.info(f"Killed {killed} process(es) matching '{name}'")
            return True
        logger.warning(f"No process found matching '{name}'")
        return False
    except Exception as e:
        logger.error(f"Failed to kill process '{name}': {e}")
        return False
