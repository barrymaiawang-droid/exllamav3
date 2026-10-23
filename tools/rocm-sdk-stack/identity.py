#!/usr/bin/env python3
"""Print the identity of the ROCm-10 experimental stack, including the shared libraries the
running process actually mapped (proves which runtime, not just which package is installed)."""
import hashlib
import json
import os
import re
import subprocess
import sys


def sha256(path):
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def pkg_root(name):
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    return os.path.dirname(os.path.realpath(spec.origin))


def main():
    import torch

    info = {}
    info["python"] = sys.version.split()[0]
    info["python_exe"] = sys.executable
    info["prefix"] = sys.prefix
    info["torch"] = torch.__version__
    info["torch_file"] = torch.__file__
    info["torch_version_hip"] = torch.version.hip
    info["torch_version_cuda"] = torch.version.cuda
    props = torch.cuda.get_device_properties(0)
    info["device"] = dict(name=props.name, gcnArchName=props.gcnArchName,
                          warp_size=getattr(props, "warp_size", None),
                          total_mem_mb=props.total_memory // (1 << 20))
    info["hip_runtime_version"] = torch.version.hip
    try:
        info["torch_cxx11_abi"] = torch._C._GLIBCXX_USE_CXX11_ABI
    except Exception as e:  # pragma: no cover
        info["torch_cxx11_abi"] = f"unavailable: {e}"

    for mod in ("_rocm_sdk_core", "_rocm_sdk_devel", "rocm_sdk_devel", "_rocm_sdk_libraries",
                "_rocm_sdk_libraries_gfx1151", "_rocm_sdk_device_gfx1151", "rocm_sdk"):
        info[f"pkg_{mod}"] = pkg_root(mod)

    import exllamav3
    from exllamav3.version import __version__ as exl3_version
    info["exllamav3_version"] = exl3_version
    info["exllamav3_path"] = os.path.realpath(exllamav3.__file__)
    from exllamav3 import ext
    ext_path = os.path.realpath(ext.exllamav3_ext.__file__)
    info["ext_path"] = ext_path
    info["ext_sha256"] = sha256(ext_path)
    info["ext_mtime"] = os.path.getmtime(ext_path)
    info["dispatch_counter_binding"] = hasattr(ext.exllamav3_ext, "exl3_dispatch_stats")
    info["gemv_supported_symbol"] = hasattr(ext.exllamav3_ext, "exl3_gemv_supported")
    if info["dispatch_counter_binding"]:
        info["dispatch_snapshot"] = ext.exllamav3_ext.exl3_dispatch_stats()
    info["HAS_exl3_gemv_supported"] = getattr(ext.exllamav3_ext, "HAS_exl3_gemv_supported", None)

    # libraries actually mapped by this process, not the ones on disk somewhere
    maps = []
    with open("/proc/self/maps") as f:
        for line in f:
            m = re.search(r"(\S+\.so(?:\.\d+)*)$", line.strip())
            if m:
                p = m.group(1)
                if any(k in p for k in ("libhsa", "libamdhip", "hipblas", "hipblaslt", "libtorch",
                                        "librocm", "librocprofiler", "libroctx", "libamd_comgr")):
                    if p not in maps:
                        maps.append(p)
    info["loaded_runtime_libs"] = maps
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
