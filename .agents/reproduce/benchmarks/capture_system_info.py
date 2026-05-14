#!/usr/bin/env python3
"""System Information Capture for Method of Moderation Benchmarking."""

import argparse
import json
import logging
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logger = logging.getLogger("capture_system_info")


def run_command(cmd, fallback="unknown", cwd=None):
    """Run a shell command and return stdout, or fallback on error.

    Stderr from failed commands is logged at WARNING level rather than swallowed
    silently, so a non-zero exit or unexpected exception leaves a trace in the
    benchmark log.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.warning(
                "Command failed (exit %s): %s\n  stderr: %s",
                result.returncode,
                cmd,
                result.stderr.strip(),
            )
            return fallback
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", cmd)
        return fallback
    except OSError as exc:
        logger.warning("Command raised OSError (%s): %s", exc, cmd)
        return fallback


def get_cpu_info():
    """Get CPU information cross-platform."""
    system = platform.system()

    cpu_info = {
        "model": "unknown",
        "architecture": platform.machine(),
        "cores_physical": None,
        "cores_logical": os.cpu_count(),
        "frequency_mhz": None,
    }

    if system == "Darwin":  # macOS
        cpu_info["model"] = run_command("sysctl -n machdep.cpu.brand_string")
        physical = run_command("sysctl -n hw.physicalcpu")
        cpu_info["cores_physical"] = int(physical) if physical.isdigit() else None
        freq = run_command("sysctl -n hw.cpufrequency_max")
        if freq.isdigit():
            cpu_info["frequency_mhz"] = int(freq) / 1_000_000

    elif system == "Linux":
        model = run_command("lscpu | grep 'Model name' | cut -d ':' -f 2")
        if model != "unknown":
            cpu_info["model"] = model.strip()
        else:
            cpu_info["model"] = run_command(
                "cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d ':' -f 2"
            ).strip()

        physical = run_command(r"lscpu | grep 'Core(s) per socket' | awk '{print $NF}'")
        sockets = run_command(r"lscpu | grep 'Socket(s)' | awk '{print $NF}'")
        if physical.isdigit() and sockets.isdigit():
            cpu_info["cores_physical"] = int(physical) * int(sockets)

        freq = run_command(r"lscpu | grep 'CPU max MHz' | awk '{print $NF}'")
        if freq.replace(".", "").isdigit():
            cpu_info["frequency_mhz"] = float(freq)

    return cpu_info


def get_memory_info():
    """Get memory information in GB."""
    system = platform.system()

    memory_info: dict[str, float | None] = {"total_gb": None, "available_gb": None}

    if system == "Darwin":
        total = run_command("sysctl -n hw.memsize")
        if total.isdigit():
            memory_info["total_gb"] = round(int(total) / (1024**3), 2)
    elif system == "Linux":
        total = run_command(r"grep MemTotal /proc/meminfo | awk '{print $2}'")
        if total.isdigit():
            memory_info["total_gb"] = round(int(total) / (1024**2), 2)
        available = run_command(r"grep MemAvailable /proc/meminfo | awk '{print $2}'")
        if available.isdigit():
            memory_info["available_gb"] = round(int(available) / (1024**2), 2)

    return memory_info


def get_disk_info(path="/"):
    """Get disk information for the given path."""
    try:
        stat = os.statvfs(path)
    except OSError as exc:
        logger.warning("os.statvfs(%s) failed: %s", path, exc)
        return {"type": "unknown", "free_gb": None}

    free_gb = round((stat.f_bavail * stat.f_frsize) / (1024**3), 2)
    disk_type = "unknown"

    if platform.system() == "Darwin":
        disk_type_cmd = run_command(
            r"diskutil info / | grep 'Solid State' | awk '{print $3}'"
        )
        disk_type = "SSD" if disk_type_cmd == "Yes" else "HDD"
    elif platform.system() == "Linux":
        rotational = run_command(
            r"cat /sys/block/$(df / | tail -1 | awk '{print $1}' | sed 's|/dev/||' | sed 's/[0-9]//g')/queue/rotational 2>/dev/null"
        )
        disk_type = (
            "SSD" if rotational == "0" else "HDD" if rotational == "1" else "unknown"
        )

    return {"type": disk_type, "free_gb": free_gb}


def get_python_packages():
    """Get versions of key Python packages.

    Uses `importlib.metadata`. If a package is genuinely not installed the
    entry is `"not installed"`; if metadata lookup fails for some other reason
    a warning is logged so the operator can investigate rather than silently
    receive misleading version strings.
    """
    packages = {}
    key_packages = [
        "econ-ark",
        "mystmd",
        "numpy",
        "scipy",
        "matplotlib",
        "numba",
        "jupyter",
        "ipywidgets",
        "pytest",
    ]

    for pkg in key_packages:
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = "not installed"
        except (OSError, ValueError) as exc:
            logger.warning("Version lookup failed for %s: %s", pkg, exc)
            packages[pkg] = "lookup failed"

    return packages


def get_git_info(repo_path=None):
    """Get git repository information without changing the process CWD."""
    if repo_path is None:
        repo_path = Path(__file__).parent.parent.parent

    commit = run_command("git rev-parse HEAD", "unknown", cwd=repo_path)
    branch = run_command("git rev-parse --abbrev-ref HEAD", "unknown", cwd=repo_path)
    dirty = run_command(
        "git diff --quiet && echo 'false' || echo 'true'", "unknown", cwd=repo_path
    )
    return {"commit": commit, "branch": branch, "dirty": dirty == "true"}


def capture_system_info():
    """Capture complete system information."""
    system = platform.system()
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    env_type = "unknown"
    if venv_path:
        env_type = (
            "uv"
            if ".venv" in venv_path or "UV_PROJECT_ENVIRONMENT" in os.environ
            else "venv"
        )
    elif os.environ.get("CONDA_DEFAULT_ENV"):
        env_type = "conda"

    return {
        "system": {
            "os": system,
            "os_version": platform.release(),
            "kernel": platform.version(),
            "hostname": platform.node(),
            "cpu": get_cpu_info(),
            "memory": get_memory_info(),
            "disk": get_disk_info(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "environment_type": env_type,
            "virtual_env": venv_path if venv_path else None,
            "key_packages": get_python_packages(),
        },
        "git": get_git_info(),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Capture system information for benchmarking"
    )
    parser.add_argument(
        "--output", "-o", help="Output file (default: stdout)", type=str
    )
    parser.add_argument("--pretty", "-p", help="Pretty-print JSON", action="store_true")
    args = parser.parse_args()

    info = capture_system_info()
    output = json.dumps(info, indent=2) if args.pretty else json.dumps(info)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info("System information saved to: %s", args.output)
    else:
        # JSON to stdout is the script's data interface; keep print() here.
        print(output)


if __name__ == "__main__":
    main()
