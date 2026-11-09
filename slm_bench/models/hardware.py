"""
Hardware and runtime environment detection.
"""

import platform
import psutil
import subprocess
import re
import json
from pathlib import Path
from typing import Optional
from .schema import HardwareInfo


def get_cpu_brand_name() -> str:
    """Retrieve human-friendly CPU model/brand string across OS platforms."""
    # 1. Windows Registry (exact CPU model name)
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            pass
    # 2. Linux /proc/cpuinfo (works on bare metal, VMs, WSL)
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name") or "Hardware" in line:
                        val = line.split(":", 1)[1].strip()
                        if val:
                            return val
        except Exception:
            pass
    # 3. macOS sysctl
    elif platform.system() == "Darwin":
        try:
            val = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if val:
                return val
        except Exception:
            pass

    return platform.processor() or platform.machine()


def is_wsl_environment() -> bool:
    """Check if the current process is running inside WSL."""
    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                ver = f.read().lower()
                return "microsoft" in ver or "wsl" in ver
        except Exception:
            pass
    return False


def is_endpoint_wsl_relayed(endpoint_url: str = "") -> bool:
    """Check if a local endpoint is hosted inside WSL2 relayed by Windows."""
    if platform.system() == "Windows" and any(h in endpoint_url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        try:
            port_match = re.search(r":(\d+)", endpoint_url)
            target_port = int(port_match.group(1)) if port_match else 8000
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == target_port and conn.status == psutil.CONN_LISTEN:
                    if conn.pid:
                        proc = psutil.Process(conn.pid)
                        if "wsl" in proc.name().lower():
                            return True
            wsl_procs = [
                p.info.get("name") or ""
                for p in psutil.process_iter(["name"])
                if any(k in (p.info.get("name") or "").lower() for k in ("wslrelay", "vmmem"))
            ]
            if wsl_procs:
                return True
        except Exception:
            pass
    return False


def generate_auto_system_notes(
    hw: HardwareInfo,
    backend_name: str = "",
    endpoint_url: str = "",
    server_runtime: Optional[str] = None,
    user_notes: Optional[str] = None,
) -> str:
    """
    Automatically compose comprehensive system & execution notes based on detected hardware,
    GPU specifications, OS/WSL environment, and backend server telemetry.
    If the user has provided explicit notes, those are respected.
    """
    if user_notes and user_notes.strip():
        return user_notes.strip()

    parts = []

    # 1. Server runtime and environment
    if server_runtime:
        parts.append(server_runtime)
    elif backend_name:
        b_label = backend_name.upper() if backend_name.lower() in ("vllm", "hf") else backend_name.capitalize()
        if is_endpoint_wsl_relayed(endpoint_url) or is_wsl_environment():
            parts.append(f"{b_label} in WSL2")
        else:
            parts.append(f"{b_label}")

    # 2. GPU Model and VRAM
    if hw.gpu_name:
        gpu_str = hw.gpu_name.replace("NVIDIA GeForce ", "GeForce ").replace("NVIDIA ", "")
        if hw.gpu_vram_gb:
            vram_round = round(hw.gpu_vram_gb)
            gpu_str = f"{gpu_str} ({vram_round}GB VRAM)"
        parts.append(f"on {gpu_str}")
    elif hw.cpu:
        parts.append(f"on CPU ({hw.cpu})")

    # 3. Host OS if not WSL
    if not (is_endpoint_wsl_relayed(endpoint_url) or is_wsl_environment()):
        os_short = "Windows" if "Windows" in hw.os else ("Linux" if "Linux" in hw.os else "macOS")
        parts.append(f"({os_short})")

    return " ".join(parts).strip()


def _parse_vulkaninfo_devices(text: str) -> list:
    """Extract device name, vendor, type, and driver from ``vulkaninfo --summary``."""
    devices = []
    current = None

    def new_device():
        return {"name": None, "driver": None, "vendor_id": None, "device_type": None}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.match(r"^GPU\d+:$", line):
            if current and current["name"]:
                devices.append(current)
            current = new_device()
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if key not in ("deviceName", "driverInfo", "vendorID", "deviceType"):
            continue
        if current is not None and current["name"] and key in ("deviceName", "vendorID", "deviceType"):
            devices.append(current)
            current = new_device()
        if current is None:
            current = new_device()
        if key == "deviceName":
            current["name"] = value
        elif key == "driverInfo":
            current["driver"] = value or None
        elif key == "vendorID":
            current["vendor_id"] = value or None
        elif key == "deviceType":
            current["device_type"] = value or None
    if current and current["name"]:
        devices.append(current)
    return devices


def _select_vulkan_device(devices: list) -> Optional[dict]:
    """Pick a hardware GPU: AMD only when vendor IDs are known, discrete first."""
    candidates = devices
    if any(device.get("vendor_id") for device in devices):
        candidates = [d for d in devices if (d.get("vendor_id") or "").lower() == "0x1002"]
        if not candidates:
            return None
    for device in candidates:
        if (device.get("device_type") or "").upper() == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU":
            return device
    ignored = ("ryzen", "raphael", "cpu", "llvmpipe", "lavapipe", "phoenix", "renoir", "vega")
    for device in candidates:
        name = (device.get("name") or "").lower()
        if name and not any(token in name for token in ignored):
            return device
    return candidates[0] if candidates else None


def _parse_sysfs_amd_cards(root: Path) -> list:
    """Return AMD cards under a sysfs DRM root, largest VRAM first."""
    cards = []
    for device_path in root.glob("card*/device"):
        try:
            vendor = (device_path / "vendor").read_text(encoding="utf-8").strip().lower()
            if vendor != "0x1002":
                continue
            try:
                vram_bytes = int((device_path / "mem_info_vram_total").read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                vram_bytes = 0
            cards.append({
                "pci_slot": device_path.resolve().name,
                "vram_gb": round(vram_bytes / (1024 ** 3), 2) if vram_bytes > 0 else None,
            })
        except OSError:
            continue
    return sorted(
        cards,
        key=lambda card: (card["vram_gb"] is not None, card["vram_gb"] or 0.0),
        reverse=True,
    )


def _parse_lspci_names(text: str) -> dict:
    """Map PCI slots to cleaned device names from ``lspci -D -mm -nn`` output."""
    names = {}
    id_suffix = re.compile(r"\s*\[[0-9a-fA-F]{4}(?::[0-9a-fA-F]{4})?\]\s*$")
    for line in text.splitlines():
        quoted = re.findall(r'"([^"]*)"', line)
        if len(quoted) < 3:
            continue
        slot = line.split(None, 1)[0]
        names[slot] = id_suffix.sub("", quoted[2]).strip()
    return names


def _detect_amd_linux() -> Optional[dict]:
    """Detect an AMD GPU on Linux via sysfs, with Vulkan/lspci name fallbacks."""
    cards = _parse_sysfs_amd_cards(Path("/sys/class/drm"))
    if not cards:
        return None

    selected = None
    try:
        proc = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            selected = _select_vulkan_device(_parse_vulkaninfo_devices(proc.stdout))
    except Exception:
        selected = None

    name = selected["name"] if selected else None
    if not name:
        try:
            proc = subprocess.run(
                ["lspci", "-D", "-mm", "-nn"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                name = _parse_lspci_names(proc.stdout).get(cards[0]["pci_slot"])
        except Exception:
            name = None

    return {
        "name": name or "AMD Radeon GPU",
        "count": len(cards),
        "vram_gb": cards[0]["vram_gb"],
        "driver": selected["driver"] if selected else None,
    }


def _parse_windows_video_controllers(text: str) -> list:
    """Extract AMD adapters from ``Win32_VideoController`` JSON."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    devices = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if any(token in lowered for token in ("microsoft basic", "remote", "virtual")):
            continue
        pnp_id = str(item.get("PNPDeviceID") or "").upper()
        if "VEN_1002" not in pnp_id and "radeon" not in lowered and "amd" not in lowered:
            continue
        adapter_ram = item.get("AdapterRAM")
        try:
            adapter_ram = int(adapter_ram) if adapter_ram is not None else None
        except (TypeError, ValueError):
            adapter_ram = None
        driver = item.get("DriverVersion")
        devices.append({
            "name": name,
            "driver": str(driver) if driver else None,
            "adapter_ram": adapter_ram,
        })
    return devices


def _choose_windows_vram(qw_memory_size: Optional[int], adapter_ram: Optional[int]) -> Optional[float]:
    """Prefer the 64-bit registry value; ignore AdapterRAM's 4 GiB wrap."""
    if isinstance(qw_memory_size, int) and qw_memory_size > 0:
        return round(qw_memory_size / (1024 ** 3), 2)
    wrapped_threshold = 4 * 1024 ** 3 - 16 * 1024 ** 2
    if isinstance(adapter_ram, int) and 0 < adapter_ram < wrapped_threshold:
        return round(adapter_ram / (1024 ** 3), 2)
    return None


def _windows_registry_vram(device_desc: str) -> Optional[int]:
    """Read ``HardwareInformation.qwMemorySize`` for a matching display adapter."""
    try:
        import winreg

        class_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_path) as class_key:
            index = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(class_key, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(class_key, subkey_name) as subkey:
                        try:
                            desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                        except OSError:
                            continue
                        if str(desc).strip().lower() != device_desc.strip().lower():
                            continue
                        value, _ = winreg.QueryValueEx(subkey, "HardwareInformation.qwMemorySize")
                        return int(value)
                except OSError:
                    continue
    except Exception:
        return None
    return None


def _detect_amd_windows() -> Optional[dict]:
    """Detect an AMD GPU on Windows via CIM and the display-driver registry."""
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM,PNPDeviceID | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None

    devices = _parse_windows_video_controllers(proc.stdout)
    if not devices:
        return None

    best_device = None
    best_vram = None
    for device in devices:
        vram = _choose_windows_vram(_windows_registry_vram(device["name"]), device["adapter_ram"])
        if best_device is None or (vram is not None and (best_vram is None or vram > best_vram)):
            best_device = device
            best_vram = vram

    return {
        "name": best_device["name"],
        "count": len(devices),
        "vram_gb": best_vram,
        "driver": best_device["driver"],
    }


def _detect_amd_gpu() -> Optional[dict]:
    """Detect an AMD GPU on supported platforms; None when nothing is found."""
    system = platform.system()
    if system == "Linux":
        return _detect_amd_linux()
    if system == "Windows":
        return _detect_amd_windows()
    return None


def get_hardware_info(model_ram_gb: Optional[float] = None) -> HardwareInfo:
    """Collect host hardware and system specifications."""
    # OS & CPU
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    cpu_name = get_cpu_brand_name()
    cpu_cores = psutil.cpu_count(logical=True) or 1
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    gpu_name: Optional[str] = None
    gpu_count = 0
    gpu_vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None

    # First attempt detection using nvidia-smi CSV query (exact, untruncated name and total memory)
    try:
        csv_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if csv_proc.returncode == 0 and csv_proc.stdout.strip():
            lines = [l.strip() for l in csv_proc.stdout.strip().splitlines() if l.strip()]
            if lines:
                gpu_count = len(lines)
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 1 and parts[0]:
                    gpu_name = parts[0]
                if len(parts) >= 2 and parts[1]:
                    try:
                        gpu_vram_gb = round(float(parts[1]) / 1024, 2)
                    except ValueError:
                        pass
                if len(parts) >= 3 and parts[2]:
                    driver_version = parts[2]
    except Exception:
        pass

    # Extract CUDA version via nvidia-smi if not set
    if not cuda_version:
        try:
            smi_proc = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False
            )
            if smi_proc.returncode == 0:
                cuda_match = re.search(r"CUDA (?:Version|UMD Version):\s*([0-9.]+)", smi_proc.stdout, re.IGNORECASE)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
        except Exception:
            pass

    # Attempt detection using torch if available
    try:
        import torch
        if torch.cuda.is_available():
            if not gpu_count:
                gpu_count = torch.cuda.device_count()
            if not gpu_name:
                gpu_name = torch.cuda.get_device_name(0)
            if gpu_vram_gb is None:
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_vram_gb = round(total_bytes / (1024 ** 3), 2)
            if not cuda_version:
                cuda_version = torch.version.cuda
    except Exception:
        pass

    # If torch is not initialized or torch.cuda has no driver, fallback to nvidia-smi
    if not gpu_name or not driver_version:
        try:
            smi_proc = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False
            )
            if smi_proc.returncode == 0:
                output = smi_proc.stdout
                # Extract Driver Version & CUDA Version
                driver_match = re.search(r"Driver-Model.*?Driver Version:\s*([0-9.]+)|Driver Version:\s*([0-9.]+)|KMD Version:\s*([0-9.]+)", output, re.IGNORECASE)
                if driver_match:
                    driver_version = next(g for g in driver_match.groups() if g is not None)
                
                cuda_match = re.search(r"CUDA (?:Version|UMD Version):\s*([0-9.]+)", output, re.IGNORECASE)
                if cuda_match and not cuda_version:
                    cuda_version = cuda_match.group(1)

                # Extract GPU Name if not yet found
                if not gpu_name:
                    name_match = re.search(r"\|\s*\d+\s+([A-Za-z0-9\s]+(?:RTX|GeForce|Tesla|A100|H100|GTX)[^|]*?)\s+(?:WDDM|Off|On)", output)
                    if not name_match:
                        name_match = re.search(r"(?:NVIDIA\s+GeForce\s+RTX\s+[0-9A-Za-z\s]+|NVIDIA\s+[A-Za-z0-9\s]+)", output)
                    if name_match:
                        gpu_name = name_match.group(1 if name_match.lastindex else 0).strip()
                        gpu_count = max(gpu_count, 1)

                # Extract Memory if not yet found
                if gpu_vram_gb is None:
                    mem_match = re.search(r"([0-9]+)\s*MiB\s*/\s*([0-9]+)\s*MiB", output)
                    if mem_match:
                        total_mib = float(mem_match.group(2))
                        gpu_vram_gb = round(total_mib / 1024, 2)
        except Exception:
            pass

    if not gpu_name:
        amd = _detect_amd_gpu()
        if amd:
            gpu_name = amd["name"]
            gpu_count = amd["count"]
            gpu_vram_gb = amd["vram_gb"]
            driver_version = amd["driver"]

    return HardwareInfo(
        os=os_name,
        cpu=cpu_name,
        cpu_cores=cpu_cores,
        total_ram_gb=total_ram_gb,
        model_ram_gb=model_ram_gb,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_gb=gpu_vram_gb,
        cuda_version=cuda_version,
        driver_version=driver_version,
    )
