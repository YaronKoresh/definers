from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

from definers.constants import FFMPEG_URL


def install_ffmpeg_windows():
    import requests

    from definers import system as system_module

    print("[INFO] Running FFmpeg installer for Windows...")
    if not system_module.is_admin_windows():
        print(
            "[ERROR] This script requires Administrator privileges to run on Windows."
        )
        print(
            "[INFO] Please re-run this script from a terminal with Administrator rights."
        )
        sys.exit(1)
    print(
        "\n[INFO] Attempting to install using Winget (Windows Package Manager)..."
    )
    try:
        subprocess.run(
            [
                "winget",
                "install",
                "--id=Gyan.FFmpeg.Essentials",
                "-e",
                "--accept-source-agreements",
                "--accept-package-agreements",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[SUCCESS] FFmpeg has been installed via Winget.")
        print(
            "[INFO] You may need to restart your terminal for the PATH changes to take effect."
        )
        return
    except FileNotFoundError:
        print(
            "[WARN] Winget command not found. It might not be installed or in the PATH."
        )
    except subprocess.CalledProcessError as error:
        print(
            f"[WARN] Winget installation failed with exit code {error.returncode}."
        )
        print(f"[DEBUG] Winget stderr: {error.stderr}")
    print(
        "\n[INFO] Winget installation failed or was not available. Attempting manual download..."
    )
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "ffmpeg.zip")
    extract_path = os.path.join(temp_dir, "ffmpeg_extracted")
    if "/" in temp_dir:
        zip_path = zip_path.replace("\\", "/")
        extract_path = extract_path.replace("\\", "/")
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    ffmpeg_install_dir = os.path.join(program_files, "ffmpeg")
    try:
        print(
            f"[INFO] Downloading latest FFmpeg essentials build from {FFMPEG_URL}..."
        )
        os.makedirs(temp_dir, exist_ok=True)
        with requests.get(FFMPEG_URL, stream=True) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
        print("[SUCCESS] Download complete.")
        print(f"[INFO] Extracting FFmpeg to {extract_path}...")
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print("[SUCCESS] Extraction complete.")
        extracted_files = os.listdir(extract_path)
        if not extracted_files:
            raise OSError(
                "Extraction failed, no files found in temporary directory."
            )
        ffmpeg_build_dir = os.path.join(extract_path, extracted_files[0])
        ffmpeg_bin_dir = os.path.join(ffmpeg_build_dir, "bin")
        print(f"[INFO] Moving FFmpeg binaries to {ffmpeg_install_dir}...")
        if os.path.exists(ffmpeg_install_dir):
            shutil.rmtree(ffmpeg_install_dir)
        shutil.move(ffmpeg_bin_dir, ffmpeg_install_dir)
        print("[SUCCESS] Binaries moved.")
        print("[INFO] Adding FFmpeg to the system PATH...")
        subprocess.run(
            ["setx", "/M", "PATH", f"%PATH%;{ffmpeg_install_dir}"],
            check=True,
        )
        print("[SUCCESS] FFmpeg added to system PATH.")
        print(
            "[INFO] IMPORTANT: You must restart your terminal or PC for the new PATH to be recognized."
        )
    except Exception as error:
        system_module.logger.error(
            f"An error occurred during manual installation: {error}"
        )
        sys.exit(1)
    finally:
        print("[INFO] Cleaning up temporary files...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        print("[INFO] Cleanup complete.")


def install_ffmpeg_linux():
    from definers import system as system_module

    print("[INFO] Running FFmpeg installer for Linux...")
    if os.geteuid() != 0:
        print("[WARN] This script needs sudo privileges to install packages.")
        print("[INFO] It will likely prompt you for your password.")
    package_managers = {
        "apt": {
            "update_cmd": ["apt-get", "update"],
            "install_cmd": ["apt-get", "install", "ffmpeg", "-y"],
        },
        "dnf": {"install_cmd": ["dnf", "install", "ffmpeg", "-y"]},
        "pacman": {"install_cmd": ["pacman", "-S", "ffmpeg", "--noconfirm"]},
    }
    selected_pm = None
    for package_manager in package_managers:
        if shutil.which(package_manager):
            selected_pm = package_manager
            break
    if not selected_pm:
        print(
            "[ERROR] Could not detect a supported package manager (apt, dnf, pacman)."
        )
        print("[INFO] Please install FFmpeg manually.")
        sys.exit(1)
    print(f"[INFO] Detected package manager: {selected_pm}")
    try:
        pm_cmds = package_managers[selected_pm]
        if "update_cmd" in pm_cmds:
            print(f"[INFO] Running package list update ({selected_pm})...")
            subprocess.run(pm_cmds["update_cmd"], check=True)
        print(f"[INFO] Installing FFmpeg using {selected_pm}...")
        subprocess.run(pm_cmds["install_cmd"], check=True)
        print("\n[SUCCESS] FFmpeg installed successfully.")
    except subprocess.CalledProcessError as error:
        print(
            f"\n[ERROR] The installation command failed with exit code {error.returncode}."
        )
        print(
            "[INFO] Please check the output above for errors from the package manager."
        )
        sys.exit(1)
    except Exception as error:
        system_module.logger.error(f"An unexpected error occurred: {error}")
        sys.exit(1)


def install_ffmpeg():
    from definers import system as system_module

    if system_module.runnable("ffmpeg"):
        return True
    if system_module.installed("ffmpeg"):
        return True
    system_name = system_module.get_os_name()
    if system_name == "windows":
        system_module.install_ffmpeg_windows()
        return True
    if system_name == "linux":
        system_module.install_ffmpeg_linux()
        return True
    print(f"[ERROR] Unsupported operating system: {system_name}.")
    print("[INFO] This script only supports Windows and Linux.")
    sys.exit(1)


def install_audio_effects():
    from definers import system as system_module
    from definers.media.web_transfer import (
        add_to_path_windows,
        download_and_unzip,
        download_file,
    )

    os_name = system_module.get_os_name()
    if os_name == "linux":
        print("Detected Linux. Installing system dependencies with apt-get...")
        dependencies_apt = [
            "rubberband-cli",
            "fluidsynth",
            "fluid-soundfont-gm",
            "build-essential",
        ]
        system_module.run(["apt-get", "update", "-y"])
        system_module.run(["apt-get", "install", "-y"] + dependencies_apt)
    elif os_name == "windows":
        install_dir = os.path.join(os.path.expanduser("~"), "app_dependencies")
        os.makedirs(install_dir, exist_ok=True)
        print("Detected Windows. Automating dependency installation...")
        print(f"Dependencies will be installed in: {install_dir}")
        rubberband_url = "https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip"
        fluidsynth_url = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.5.2/fluidsynth-v2.5.2-win10-x64-glib.zip"
        soundfont_url = "https://raw.githubusercontent.com/FluidSynth/fluidsynth/master/sf2/VintageDreamsWaves-v2.sf3"
        soundfont_path = os.path.join(
            install_dir,
            "soundfonts",
            "VintageDreamsWaves-v2.sf3",
        )
        rubberband_extract_path = os.path.join(install_dir, "rubberband")
        if "rubberband" not in os.environ.get(
            "PATH", ""
        ) or not system_module.exist(rubberband_extract_path):
            if download_and_unzip(rubberband_url, rubberband_extract_path):
                extracted_dirs = os.listdir(rubberband_extract_path)
                if extracted_dirs:
                    rubberband_bin_path = os.path.join(
                        rubberband_extract_path,
                        extracted_dirs[0],
                    )
                    add_to_path_windows(rubberband_bin_path)
        fluidsynth_extract_path = os.path.join(install_dir, "fluidsynth")
        if "fluidsynth" not in os.environ.get(
            "PATH", ""
        ) or not system_module.exist(fluidsynth_extract_path):
            if download_and_unzip(fluidsynth_url, fluidsynth_extract_path):
                fluidsynth_bin_path = os.path.join(
                    fluidsynth_extract_path,
                    "bin",
                )
                add_to_path_windows(fluidsynth_bin_path)
        if not system_module.exist(soundfont_path):
            os.makedirs(os.path.dirname(soundfont_path), exist_ok=True)
            print("Downloading SoundFont for MIDI playback...")
            download_file(soundfont_url, soundfont_path)
    else:
        print(
            f"Unsupported OS: {os_name}. Manual installation of system dependencies may be required."
        )
    print("\nInstalling Python packages with pip...")


def pip_install(packs):
    from definers import system as system_module
    from definers.media.web_transfer import download_file

    packs_arr = packs.strip().split()
    for index, pack in enumerate(packs_arr):
        if (
            pack.startswith("https://") or pack.startswith("http://")
        ) and pack.endswith(".whl"):
            temp_path = system_module.tmp("whl", keep=False)
            download_file(pack, temp_path)
            packs_arr[index] = temp_path
    pack_list = (
        packs if isinstance(packs, list) else " ".join(packs_arr).split()
    )
    system_module.run(
        ["pip", "install", "--upgrade", "--force-reinstall", "--no-cache-dir"]
        + pack_list
    )
    for pack in packs_arr:
        package_name = pack
        if package_name.endswith(".whl"):
            package_name = package_name.split("-py3")[0].split("-py2")[0]
        package_paths = system_module.find_package_paths(package_name)
        system_module.log("Package paths", package_paths)
        for package_path in package_paths:
            system_module.add_path(package_path)


def modify_wheel_requirements(wheel_path: str, requirements_map: dict):
    from definers import system as system_module

    print(f"Modifying metadata for wheel: {wheel_path}")
    if not os.path.exists(wheel_path):
        raise FileNotFoundError(f"Wheel file not found at {wheel_path}")
    temp_dir = system_module.tmp(dir=True)
    output_dir = os.path.dirname(wheel_path) or "."
    wheel_filename = os.path.basename(wheel_path)
    try:
        with zipfile.ZipFile(wheel_path, "r") as wheel_zip:
            wheel_zip.extractall(temp_dir)
        metadata_files = system_module.paths(
            os.path.join(temp_dir, "*.dist-info", "METADATA")
        )
        if not metadata_files:
            raise FileNotFoundError("Could not find METADATA file in wheel.")
        metadata_path = metadata_files[0]
        with open(metadata_path, encoding="utf-8") as handle:
            metadata_content = handle.read()

        lines = metadata_content.splitlines()
        for package_name, version_specifier in requirements_map.items():
            found = False
            new_lines: list[str] = []
            lower_name = package_name.lower()
            for line in lines:
                if line.lower().startswith("requires-dist:"):
                    rest = line[len("Requires-Dist:") :].strip()
                    pkg = rest.split()[0]
                    if pkg.lower() == lower_name:
                        found = True
                        if version_specifier:
                            new_lines.append(
                                f"Requires-Dist: {package_name} ({version_specifier})"
                            )
                        else:
                            print(f"Removed dependency: {package_name}")
                            continue
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            if not found and version_specifier:
                new_lines.append(
                    f"Requires-Dist: {package_name} ({version_specifier})"
                )
                print(
                    f"Added new dependency: {package_name} ({version_specifier})"
                )
            elif found and version_specifier:
                print(
                    f"Modified dependency: {package_name} -> {version_specifier}"
                )
            lines = new_lines

        metadata_content = "\n".join(line for line in lines if line.strip())
        with open(metadata_path, "w", encoding="utf-8") as handle:
            handle.write(metadata_content)
        new_wheel_path = os.path.join(output_dir, wheel_filename)
        if os.path.abspath(wheel_path) == os.path.abspath(new_wheel_path):
            os.remove(wheel_path)
        with zipfile.ZipFile(
            new_wheel_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as new_wheel_zip:
            for root, _, files in os.walk(temp_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_wheel_zip.write(file_path, arcname)
        print(f"Repacked wheel to: {new_wheel_path}")
        return new_wheel_path
    finally:
        system_module.delete(temp_dir)


def build_faiss():
    from definers import system as system_module
    from definers.cuda import free, set_cuda_env
    from definers.ml import git

    with system_module.cwd():
        git("YaronKoresh", "faiss", parent="./xfaiss")
    set_cuda_env()
    cmake = "/usr/local/cmake/bin/cmake"
    try:
        with system_module.cwd("./xfaiss"):
            print("faiss - stage 1")
            system_module.run(
                f"{cmake} -B build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_MKL=OFF -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DPython_EXECUTABLE={sys.executable} -DPython_INCLUDE_DIR={sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor} -DPython_LIBRARY={sys.prefix}/lib/libpython{sys.version_info.major}.{sys.version_info.minor}.so -DPython_NumPy_INCLUDE_DIRS={sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/numpy/core/include ."
            )
            print("faiss - stage 2")
            system_module.run(
                f"{cmake} --build build -j {system_module.cores()} --target faiss"
            )
            print("faiss - stage 3")
            system_module.run(
                f"{cmake} --build build -j {system_module.cores()} --target swigfaiss"
            )
        temp_dir = system_module.tmp(dir=True)
        with system_module.cwd("./xfaiss/build/faiss/python"):
            print(
                "faiss - stage 4: Building wheel with numpy==1.26.4 constraint"
            )
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as reqs:
                reqs.write("numpy==1.26.4\n")
                constraints_path = reqs.name
            try:
                system_module.run(
                    f"{sys.executable} -m pip wheel . -w {temp_dir} -c {constraints_path}"
                )
            finally:
                os.remove(constraints_path)
        with system_module.cwd():
            system_module.delete("./xfaiss")
        free()
        any_wheel_path = system_module.paths(f"{temp_dir}/faiss-*.whl")[0]
        repaired_wheel_dir = system_module.tmp(dir=True)
        print("faiss - stage 5: Repairing wheel")
        system_module.run(
            f"{sys.executable} -m auditwheel repair {any_wheel_path} -w {repaired_wheel_dir}"
        )
        repaired_wheel_path = system_module.paths(
            f"{repaired_wheel_dir}/faiss-*.whl"
        )[0]
        print(
            "faiss - stage 6: Modifying final wheel metadata for runtime constraints"
        )
        dependency_constraints = {"numpy": "==1.26.4"}
        return modify_wheel_requirements(
            repaired_wheel_path,
            dependency_constraints,
        )
    except subprocess.CalledProcessError as error:
        system_module.catch(f"Error during installation: {error}")
    except FileNotFoundError as error:
        system_module.catch(f"File not found error: {error}")
    except Exception as error:
        system_module.catch(f"An unexpected error occurred: {error}")


def pre_install():
    try:
        home = str(pathlib.Path.home())
    except Exception:
        home = os.path.expanduser("~")
        if not home or home == "~":
            home = os.getcwd()
    os.environ.setdefault("HOME", home)
    os.environ["TRANSFORMERS_CACHE"] = "/opt/ml/checkpoints/"
    os.environ["HF_DATASETS_CACHE"] = "/opt/ml/checkpoints/"
    os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
    os.environ["OMP_NUM_THREADS"] = "4"
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["DISPLAY"] = ":0.0"
    os.environ["NUMBA_CACHE_DIR"] = f"{os.environ['HOME']}/.tmp"
    os.environ["DISABLE_FLASH_ATTENTION"] = "True"


def post_install():
    import numpy as _np

    from definers import cuda as _cuda

    try:
        import torch.fx.experimental.proxy_tensor as proxy_mod

        if not hasattr(proxy_mod, "get_proxy_mode"):
            proxy_mod.get_proxy_mode = lambda: None
    except Exception:
        pass
    if not hasattr(_np, "_no_nep50_warning"):
        _np._no_nep50_warning = lambda *a, **_kw: None
    _cuda.free()


def install_faiss():
    from definers import system as system_module

    if system_module.importable("faiss"):
        return False
    try:
        faiss_dir = "_faiss_"
        with system_module.cwd() as current_directory:
            faiss_dir = current_directory
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/faiss.git",
                faiss_dir,
            ],
            check=True,
        )
        py = sys.executable
        prefix = sys.prefix
        major = sys.version_info.major
        minor = sys.version_info.minor
        subprocess.run(
            [
                "cmake",
                "-B",
                f"{faiss_dir}/build",
                "-DBUILD_TESTING=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DFAISS_ENABLE_C_API=ON",
                "-DFAISS_ENABLE_GPU=ON",
                "-DFAISS_ENABLE_PYTHON=ON",
                f"-DPython_EXECUTABLE={py}",
                f"-DPython_INCLUDE_DIR={prefix}/include/python{major}.{minor}",
                f"-DPython_LIBRARY={prefix}/lib/libpython{major}.{minor}.so",
                f"-DPython_NumPy_INCLUDE_DIRS={prefix}/lib/python{major}.{minor}/site-packages/numpy/core/include",
                ".",
            ],
            check=True,
        )
        subprocess.run(
            ["make", "-C", f"{faiss_dir}/build", "-j16", "faiss"],
            check=True,
        )
        subprocess.run(
            ["make", "-C", f"{faiss_dir}/build", "-j16", "swigfaiss"],
            check=True,
        )
        subprocess.run(
            [py, "-m", "pip", "install", "."],
            cwd=f"{faiss_dir}/build/faiss/python",
            check=True,
        )
    except FileNotFoundError as error:
        print(f"File not found error: {error}")
    except Exception as error:
        print(f"An unexpected error occurred: {error}")


def apt_install(upgrade=False):
    from definers import system as system_module

    system_module.pre_install()
    basic_apt = "build-essential gcc cmake swig gdebi git git-lfs wget curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev initramfs-tools libgirepository1.0-dev libdbus-1-dev libdbus-glib-1-dev libsecret-1-0 libmanette-0.2-0 libharfbuzz0b libharfbuzz-icu0 libenchant-2-2 libhyphen0 libwoff1 libgraphene-1.0-0 libxml2-dev libxmlsec1-dev"
    audio_apt = "libportaudio2 libasound2-dev sox libsox-fmt-all praat ffmpeg libavcodec-extra libavif-dev"
    visual_apt = "libopenblas-dev libgflags-dev libgles2 libgtk-3-0 libgtk-4-1 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libatspi2.0-0 libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-gl"
    system_module.run(["apt-get", "update"])
    pkgstr = f"{basic_apt} {audio_apt} {visual_apt}".strip()
    system_module.run(["apt-get", "install", "-y"] + pkgstr.split())
    if upgrade:
        system_module.run("apt-get upgrade -y")
    system_module.post_install()
