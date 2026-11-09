"""
Tests for hardware detection, model metadata, and parameter extraction.
"""

import platform

import pytest
from slm_bench.models.hardware import get_hardware_info
from slm_bench.models.metadata import (
    determine_param_tier,
    extract_params_from_name,
    generate_model_slug,
    fetch_hf_model_metadata,
)


def test_determine_param_tier():
    assert determine_param_tier(1.5) == "<3B"
    assert determine_param_tier(3.0) == "3B-8B"
    assert determine_param_tier(7.0) == "3B-8B"
    assert determine_param_tier(14.0) == "8B-14B"
    assert determine_param_tier(32.0) == "14B-35B"
    assert determine_param_tier(45.0) == "35B-50B"
    assert determine_param_tier(70.0) == ">50B"
    assert determine_param_tier(None) == "Unknown"


def test_extract_params_from_name():
    assert extract_params_from_name("Qwen2.5-7B-Instruct") == 7.0
    assert extract_params_from_name("Phi-3.5-mini-instruct-3.8b") == 3.8
    assert extract_params_from_name("Llama-3.2-1B") == 1.0
    assert extract_params_from_name("Qwen2.5-0.5B-Chat") == 0.5
    assert extract_params_from_name("SmolLM-360M") == 0.36
    assert extract_params_from_name("NoParamName") is None


def test_generate_model_slug():
    slug = generate_model_slug("meta-llama/Llama-3.2-3B-Instruct", "BF16")
    assert slug == "Llama-3.2-3B-Instruct-BF16"
    assert "/" not in slug
    assert " " not in slug

    slug2 = generate_model_slug("Qwen/Qwen2.5-Coder-7B-Instruct", "GGUF_Q4_K_M")
    assert slug2 == "Qwen2.5-Coder-7B-Instruct-GGUF_Q4_K_M"


def test_get_hardware_info():
    hw = get_hardware_info()
    assert hw.os is not None
    assert hw.cpu is not None
    assert hw.cpu_cores > 0
    assert hw.total_ram_gb > 0


def test_get_cpu_brand_name():
    from slm_bench.models.hardware import get_cpu_brand_name
    cpu = get_cpu_brand_name()
    assert cpu is not None
    assert len(cpu) > 0


def test_generate_auto_system_notes():
    from slm_bench.models.hardware import generate_auto_system_notes, HardwareInfo
    hw = HardwareInfo(
        os="Windows 11",
        cpu="Intel Ultra 9",
        cpu_cores=16,
        total_ram_gb=32.0,
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        gpu_vram_gb=12.0,
    )
    # Auto-generated notes without user notes
    notes = generate_auto_system_notes(hw, backend_name="vllm", server_runtime="vLLM in WSL2")
    assert "vLLM in WSL2" in notes
    assert "GeForce RTX 5070 Ti" in notes
    assert "12GB VRAM" in notes

    # Preserves user notes when provided
    user_notes = generate_auto_system_notes(hw, backend_name="vllm", server_runtime="vLLM in WSL2", user_notes="Needs custom code execution.")
    assert user_notes == "Needs custom code execution."


def test_fetch_hf_model_metadata_fallback():
    # Test fallback heuristics when offline or with custom name
    meta = fetch_hf_model_metadata("local/test-model-7B", quantization="AWQ_4BIT", backend="vllm")
    assert meta.model_id == "local/test-model-7B"
    assert meta.param_count_b == 7.0
    assert meta.param_tier == "3B-8B"
    assert meta.quantization == "AWQ_4BIT"
    assert meta.backend == "vllm"
    assert meta.model_ram_gb is not None


def test_estimate_model_ram_gb():
    from slm_bench.models.metadata import estimate_model_ram_gb
    assert estimate_model_ram_gb(7.0, "GGUF_Q4_K_M") == 4.55
    assert estimate_model_ram_gb(8.0, "GGUF_Q8_0") == 9.20
    assert estimate_model_ram_gb(3.0, "BF16") == 6.45
    assert estimate_model_ram_gb(None, "BF16") is None


def test_auto_detect_model_ram_gb(tmp_path):
    from slm_bench.models.memory_detector import auto_detect_model_ram_gb, detect_local_file_ram_gb

    # Test file-based detection on dummy GGUF file
    dummy_gguf = tmp_path / "model.gguf"
    # Write 10 MB file
    dummy_gguf.write_bytes(b"\x00" * (10 * 1024 * 1024))
    file_ram = detect_local_file_ram_gb(str(dummy_gguf))
    assert file_ram is not None
    assert round(file_ram, 2) == 0.01

    # Test fallback analytical detection
    ram = auto_detect_model_ram_gb(param_count_b=7.0, quantization="GGUF_Q4_K_M")
    assert ram == 4.55


def test_parse_vulkaninfo_devices_reads_name_and_driver():
    from slm_bench.models.hardware import _parse_vulkaninfo_devices

    text = """deviceName = AMD Radeon RX 9060 XT (RADV GFX1200)
driverName = radv
driverInfo = Mesa 26.2.4-arch1.1
deviceName = AMD Ryzen 7 7700X 8-Core Processor (RADV RAPHAEL_MENDOCINO)
driverName = radv
driverInfo = Mesa 26.2.4-arch1.1
deviceName = llvmpipe (LLVM 20.1.8, 256 bits)
driverName = llvmpipe
"""
    devices = _parse_vulkaninfo_devices(text)
    assert devices[0] == {
        "name": "AMD Radeon RX 9060 XT (RADV GFX1200)",
        "driver": "Mesa 26.2.4-arch1.1",
        "vendor_id": None,
        "device_type": None,
    }
    assert len(devices) == 3


def test_select_vulkan_device_skips_igpu_and_software():
    from slm_bench.models.hardware import _select_vulkan_device

    devices = [
        {"name": "AMD Ryzen 7 7700X 8-Core Processor (RADV RAPHAEL_MENDOCINO)", "driver": "Mesa"},
        {"name": "llvmpipe (LLVM 20.1.8, 256 bits)", "driver": "llvmpipe"},
        {"name": "AMD Radeon RX 9060 XT (RADV GFX1200)", "driver": "Mesa 26.2.4"},
    ]
    assert _select_vulkan_device(devices)["name"] == "AMD Radeon RX 9060 XT (RADV GFX1200)"


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Fake sysfs PCI slots contain ':' which is not a valid Windows directory name",
)
def test_parse_sysfs_amd_cards_reads_amd_only_and_sorts_by_vram(tmp_path):
    from slm_bench.models.hardware import _parse_sysfs_amd_cards

    def card(name, vendor, vram, slot):
        target = tmp_path / slot
        target.mkdir()
        card_dir = tmp_path / name
        card_dir.mkdir()
        (card_dir / "device").symlink_to(target, target_is_directory=True)
        (target / "vendor").write_text(vendor)
        (target / "mem_info_vram_total").write_text(str(vram))

    card("card0", "0x1002", 512 * 1024**2, "0000:0f:00.0")       # AMD iGPU
    card("card1", "0x1002", 17095983104, "0000:03:00.0")          # AMD dGPU
    card("card2", "0x10de", 8 * 1024**3, "0000:01:00.0")          # NVIDIA

    cards = _parse_sysfs_amd_cards(tmp_path)
    assert cards == [
        {"pci_slot": "0000:03:00.0", "vram_gb": 15.92},
        {"pci_slot": "0000:0f:00.0", "vram_gb": 0.5},
    ]


def test_vulkaninfo_parser_ignores_empty_output():
    from slm_bench.models.hardware import _parse_vulkaninfo_devices

    assert _parse_vulkaninfo_devices("") == []
    assert _parse_vulkaninfo_devices("driverInfo = Mesa\n") == []


def test_parse_vulkaninfo_devices_accepts_padded_output():
    from slm_bench.models.hardware import _parse_vulkaninfo_devices

    text = (
        "\tdeviceName         = AMD Radeon RX 9060 XT (RADV GFX1200)\n"
        "\tdriverName         = radv\n"
        "\tdriverInfo         = Mesa 26.2.4-arch1.1\n"
    )
    assert _parse_vulkaninfo_devices(text) == [
        {
            "name": "AMD Radeon RX 9060 XT (RADV GFX1200)",
            "driver": "Mesa 26.2.4-arch1.1",
            "vendor_id": None,
            "device_type": None,
        }
    ]


def test_parse_lspci_names_cleans_device_name():
    from slm_bench.models.hardware import _parse_lspci_names

    text = (
        '0000:03:00.0 "VGA compatible controller [0300]" '
        '"Advanced Micro Devices, Inc. [AMD/ATI] [1002:7590]" -r00 '
        '"Navi 44 [Radeon RX 9050 / 9060 XT] [1002:7590]"\n'
    )
    assert _parse_lspci_names(text) == {
        "0000:03:00.0": "Navi 44 [Radeon RX 9050 / 9060 XT]"
    }


def test_get_hardware_info_uses_amd_when_nvidia_absent(monkeypatch):
    import types
    import slm_bench.models.hardware as hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        hardware,
        "_detect_amd_gpu",
        lambda: {"name": "AMD Radeon RX 9060 XT (RADV GFX1200)", "count": 1, "vram_gb": 15.92, "driver": "Mesa 26.2.4"},
    )
    hw = hardware.get_hardware_info()
    assert hw.gpu_name == "AMD Radeon RX 9060 XT (RADV GFX1200)"
    assert hw.gpu_count == 1
    assert hw.gpu_vram_gb == 15.92
    assert hw.driver_version == "Mesa 26.2.4"


def test_get_hardware_info_does_not_probe_amd_when_nvidia_found(monkeypatch):
    import types
    import slm_bench.models.hardware as hardware

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["nvidia-smi", "--query-gpu=name,memory.total,driver_version"]:
            return types.SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 5070 Ti, 12288, 555.42\n", stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(hardware, "_detect_amd_gpu", lambda: pytest.fail("AMD probe must not run"))
    hw = hardware.get_hardware_info()
    assert hw.gpu_name == "NVIDIA GeForce RTX 5070 Ti"


def test_parse_windows_video_controllers_keeps_amd_only():
    import json
    from slm_bench.models.hardware import _parse_windows_video_controllers

    payload = json.dumps([
        {"Name": "AMD Radeon RX 9060 XT", "DriverVersion": "31.0.12027.9001",
         "AdapterRAM": 4293918720, "PNPDeviceID": r"PCI\VEN_1002&DEV_7590&SUBSYS_00000000"},
        {"Name": "Microsoft Basic Display Adapter", "DriverVersion": "10.0.19041.1",
         "AdapterRAM": 0, "PNPDeviceID": r"PCI\VEN_1002&DEV_0000"},
        {"Name": "Intel(R) UHD Graphics 770", "DriverVersion": "31.0.101.4502",
         "AdapterRAM": 1073741824, "PNPDeviceID": r"PCI\VEN_8086&DEV_4680"},
    ])
    devices = _parse_windows_video_controllers(payload)
    assert devices == [{"name": "AMD Radeon RX 9060 XT", "driver": "31.0.12027.9001", "adapter_ram": 4293918720}]


def test_parse_windows_video_controllers_accepts_single_object():
    import json
    from slm_bench.models.hardware import _parse_windows_video_controllers

    payload = json.dumps({"Name": "AMD Radeon RX 7900 XTX", "DriverVersion": "31.0.1",
                          "AdapterRAM": 0, "PNPDeviceID": r"PCI\VEN_1002&DEV_744C"})
    assert [d["name"] for d in _parse_windows_video_controllers(payload)] == ["AMD Radeon RX 7900 XTX"]


def test_choose_windows_vram_prefers_registry_and_rejects_wrapped_adapter_ram():
    from slm_bench.models.hardware import _choose_windows_vram

    assert _choose_windows_vram(17095983104, 4293918720) == 15.92
    assert _choose_windows_vram(None, 4293918720) is None
    assert _choose_windows_vram(None, 2147483648) == 2.0
    assert _choose_windows_vram(None, None) is None


def test_windows_parser_ignores_malformed_output():
    from slm_bench.models.hardware import _parse_windows_video_controllers

    assert _parse_windows_video_controllers("not json") == []
    assert _parse_windows_video_controllers("") == []


def test_detect_amd_gpu_dispatches_windows(monkeypatch):
    import slm_bench.models.hardware as hardware

    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_detect_amd_windows", lambda: {"name": "AMD Radeon RX 9060 XT", "count": 1, "vram_gb": 15.92, "driver": "31.0.12027.9001"})
    assert hardware._detect_amd_gpu()["name"] == "AMD Radeon RX 9060 XT"


def test_parse_vulkaninfo_devices_reads_vendor_and_device_type():
    from slm_bench.models.hardware import _parse_vulkaninfo_devices

    text = (
        "Devices:\n========\nGPU0:\n"
        "\tapiVersion         = 1.4.354\n"
        "\tvendorID           = 0x1002\n"
        "\tdeviceID           = 0x7590\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        "\tdeviceName         = AMD Radeon RX 9060 XT (RADV GFX1200)\n"
        "\tdriverInfo         = Mesa 26.2.4-arch1.1\n"
        "GPU1:\n"
        "\tvendorID           = 0x1002\n"
        "\tdeviceID           = 0x164e\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
        "\tdeviceName         = AMD Ryzen 7 7700X 8-Core Processor (RADV RAPHAEL_MENDOCINO)\n"
        "\tdriverInfo         = Mesa 26.2.4-arch1.1\n"
    )
    devices = _parse_vulkaninfo_devices(text)
    assert devices[0] == {
        "name": "AMD Radeon RX 9060 XT (RADV GFX1200)",
        "driver": "Mesa 26.2.4-arch1.1",
        "vendor_id": "0x1002",
        "device_type": "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
    }
    assert devices[1]["device_type"] == "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"


def test_select_vulkan_device_prefers_discrete_amd_over_integrated_and_other_vendors():
    from slm_bench.models.hardware import _select_vulkan_device

    devices = [
        {"name": "AMD Radeon 780M (RADV PHOENIX)", "driver": "Mesa",
         "vendor_id": "0x1002", "device_type": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"},
        {"name": "Intel(R) Arc(TM) A770 Graphics (DG2)", "driver": "Intel",
         "vendor_id": "0x8086", "device_type": "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"},
        {"name": "AMD Radeon RX 7600M XT (RADV NAVI33)", "driver": "Mesa",
         "vendor_id": "0x1002", "device_type": "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"},
    ]
    assert _select_vulkan_device(devices)["name"] == "AMD Radeon RX 7600M XT (RADV NAVI33)"


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Fake sysfs PCI slots contain ':' which is not a valid Windows directory name",
)
def test_parse_sysfs_amd_cards_keeps_unknown_vram_last(tmp_path):
    from slm_bench.models.hardware import _parse_sysfs_amd_cards

    def card(name, vendor, slot, vram):
        target = tmp_path / slot
        target.mkdir()
        card_dir = tmp_path / name
        card_dir.mkdir()
        (card_dir / "device").symlink_to(target, target_is_directory=True)
        (target / "vendor").write_text(vendor)
        if vram is not None:
            (target / "mem_info_vram_total").write_text(str(vram))

    card("card0", "0x1002", "0000:0f:00.0", 512 * 1024**2)
    card("card1", "0x1002", "0000:03:00.0", None)

    cards = _parse_sysfs_amd_cards(tmp_path)
    assert cards[0]["vram_gb"] == 0.5
    assert cards[1]["vram_gb"] is None


def test_detect_amd_windows_prefers_largest_vram(monkeypatch):
    import json
    import types
    import slm_bench.models.hardware as hardware

    payload = json.dumps([
        {"Name": "AMD Radeon(TM) Graphics", "DriverVersion": "31.0.1",
         "AdapterRAM": 536870912, "PNPDeviceID": r"PCI\VEN_1002&DEV_164e"},
        {"Name": "AMD Radeon RX 7600M XT", "DriverVersion": "31.0.1",
         "AdapterRAM": 4293918720, "PNPDeviceID": r"PCI\VEN_1002&DEV_7480"},
    ])
    registry = {
        "AMD Radeon(TM) Graphics": 536870912,
        "AMD Radeon RX 7600M XT": 8589934592,
    }
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=payload, stderr=""),
    )
    monkeypatch.setattr(hardware, "_windows_registry_vram", lambda name: registry.get(name))
    result = hardware._detect_amd_windows()
    assert result["name"] == "AMD Radeon RX 7600M XT"
    assert result["vram_gb"] == 8.0
    assert result["count"] == 2
