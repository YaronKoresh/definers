import ctypes
import errno
import importlib
import logging
import os
import platform
import re
import select
import shlex
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pathlib import Path

from definers._constants import FFMPEG_URL, ai_model_extensions


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger


logger = _init_logger()


def get_os_name():
    return platform.system().lower()


def is_admin_windows():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def _install_ffmpeg_windows():
    import requests

    import definers as _d

    print("[INFO] Running FFmpeg installer for Windows...")
    if not _d.is_admin_windows():
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
    except subprocess.CalledProcessError as e:
        print(
            f"[WARN] Winget installation failed with exit code {e.returncode}."
        )
        print(f"[DEBUG] Winget stderr: {e.stderr}")
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
        with requests.get(FFMPEG_URL, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
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
            ["setx", "/M", "PATH", f"%PATH%;{ffmpeg_install_dir}"], check=True
        )
        print("[SUCCESS] FFmpeg added to system PATH.")
        print(
            "[INFO] IMPORTANT: You must restart your terminal or PC for the new PATH to be recognized."
        )
    except Exception as e:
        print(f"\n[ERROR] An error occurred during manual installation: {e}")
        sys.exit(1)
    finally:
        print("[INFO] Cleaning up temporary files...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        print("[INFO] Cleanup complete.")


def _install_ffmpeg_linux():
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
    for pm in package_managers:
        if shutil.which(pm):
            selected_pm = pm
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
    except subprocess.CalledProcessError as e:
        print(
            f"\n[ERROR] The installation command failed with exit code {e.returncode}."
        )
        print(
            "[INFO] Please check the output above for errors from the package manager."
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


def install_ffmpeg():
    import definers as _d

    if _d.installed("ffmpeg"):
        return True
    system = _d.get_os_name()
    if system == "windows":
        _d._install_ffmpeg_windows()
        return True
    elif system == "linux":
        _d._install_ffmpeg_linux()
        return True
    else:
        print(f"[ERROR] Unsupported operating system: {system}.")
        print("[INFO] This script only supports Windows and Linux.")
        sys.exit(1)


def install_audio_effects():
    import definers as _d

    os_name = _d.get_os_name()
    if os_name == "linux":
        print("Detected Linux. Installing system dependencies with apt-get...")
        dependencies_apt = [
            "rubberband-cli",
            "fluidsynth",
            "fluid-soundfont-gm",
            "build-essential",
        ]
        _d.run("apt-get update -y")
        _d.run(f"apt-get install -y {' '.join(dependencies_apt)}")
    elif os_name == "windows":
        install_dir = os.path.join(os.path.expanduser("~"), "app_dependencies")
        os.makedirs(install_dir, exist_ok=True)
        print("Detected Windows. Automating dependency installation...")
        print(f"Dependencies will be installed in: {install_dir}")
        rubberband_url = "https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip"
        fluidsynth_url = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.5.2/fluidsynth-v2.5.2-win10-x64-glib.zip"
        soundfont_url = "https://raw.githubusercontent.com/FluidSynth/fluidsynth/master/sf2/VintageDreamsWaves-v2.sf3"
        soundfont_path = os.path.join(
            install_dir, "soundfonts", "VintageDreamsWaves-v2.sf3"
        )
        rubberband_extract_path = os.path.join(install_dir, "rubberband")
        if "rubberband" not in os.environ.get("PATH", "") or not exist(
            rubberband_extract_path
        ):
            if _d.download_and_unzip(rubberband_url, rubberband_extract_path):
                extracted_dirs = os.listdir(rubberband_extract_path)
                if extracted_dirs:
                    rubberband_bin_path = os.path.join(
                        rubberband_extract_path, extracted_dirs[0]
                    )
                    _d.add_to_path_windows(rubberband_bin_path)
        fluidsynth_extract_path = os.path.join(install_dir, "fluidsynth")
        if "fluidsynth" not in os.environ.get("PATH", "") or not exist(
            fluidsynth_extract_path
        ):
            if _d.download_and_unzip(fluidsynth_url, fluidsynth_extract_path):
                fluidsynth_bin_path = os.path.join(
                    fluidsynth_extract_path, "bin"
                )
                _d.add_to_path_windows(fluidsynth_bin_path)
        if not exist(soundfont_path):
            os.makedirs(os.path.dirname(soundfont_path), exist_ok=True)
            print("Downloading SoundFont for MIDI playback...")
            _d.download_file(soundfont_url, soundfont_path)
    else:
        print(
            f"Unsupported OS: {os_name}. Manual installation of system dependencies may be required."
        )
    print("\nInstalling Python packages with pip...")


def pip_install(packs):
    from definers import download_file

    packs_arr = packs.strip().split()
    for idx, pack in enumerate(packs_arr):
        if (
            pack.startswith("https://") or pack.startswith("http://")
        ) and pack.endswith(".whl"):
            temp_path = tmp("whl", keep=False)
            download_file(pack, temp_path)
            packs_arr[idx] = temp_path
    packs = " ".join(packs_arr)
    run(f"pip install --upgrade --force-reinstall --no-cache-dir {packs}")
    for idx, pack in enumerate(packs_arr):
        if pack.endswith(".whl"):
            pack = pack.split("-py3")[0].split("-py2")[0]
        ps = find_package_paths(pack)
        log("Package paths", ps)
        for p in ps:
            add_path(p)


def modify_wheel_requirements(wheel_path: str, requirements_map: dict):
    print(f"Modifying metadata for wheel: {wheel_path}")
    if not os.path.exists(wheel_path):
        raise FileNotFoundError(f"Wheel file not found at {wheel_path}")
    temp_dir = tmp(dir=True)
    output_dir = os.path.dirname(wheel_path) or "."
    wheel_filename = os.path.basename(wheel_path)
    try:
        with zipfile.ZipFile(wheel_path, "r") as wheel_zip:
            wheel_zip.extractall(temp_dir)
        metadata_files = paths(
            os.path.join(temp_dir, "*.dist-info", "METADATA")
        )
        if not metadata_files:
            raise FileNotFoundError("Could not find METADATA file in wheel.")
        metadata_path = metadata_files[0]
        with open(metadata_path, encoding="utf-8") as f:
            metadata_content = f.read()
        for package_name, version_specifier in requirements_map.items():
            pattern = re.compile(
                f"^(Requires-Dist:\\s*{re.escape(package_name)}(\\s|\\[|;|$).*)$",
                re.IGNORECASE | re.MULTILINE,
            )
            if version_specifier:
                replacement = (
                    f"Requires-Dist: {package_name} ({version_specifier})"
                )
                found = pattern.search(metadata_content)
                if found:
                    metadata_content = pattern.sub(
                        replacement, metadata_content
                    )
                    print(
                        f"Modified dependency: {package_name} -> {version_specifier}"
                    )
                else:
                    metadata_content += f"\n{replacement}"
                    print(
                        f"Added new dependency: {package_name} ({version_specifier})"
                    )
            else:
                (metadata_content, count) = pattern.subn("", metadata_content)
                if count > 0:
                    print(f"Removed dependency: {package_name}")
        metadata_content = "\n".join(
            line for line in metadata_content.splitlines() if line.strip()
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata_content)
        new_wheel_path = os.path.join(output_dir, wheel_filename)
        if os.path.abspath(wheel_path) == os.path.abspath(new_wheel_path):
            os.remove(wheel_path)
        with zipfile.ZipFile(
            new_wheel_path, "w", zipfile.ZIP_DEFLATED
        ) as new_wheel_zip:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_wheel_zip.write(file_path, arcname)
        print(f"Repacked wheel to: {new_wheel_path}")
        return new_wheel_path
    finally:
        delete(temp_dir)


def build_faiss():
    from definers import free, git, set_cuda_env

    with cwd():
        git("YaronKoresh", "faiss", parent="./xfaiss")
    set_cuda_env()
    cmake = "/usr/local/cmake/bin/cmake"
    try:
        with cwd("./xfaiss"):
            print("faiss - stage 1")
            run(
                f"{cmake} -B build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_MKL=OFF -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DPython_EXECUTABLE={sys.executable} -DPython_INCLUDE_DIR={sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor} -DPython_LIBRARY={sys.prefix}/lib/libpython{sys.version_info.major}.{sys.version_info.minor}.so -DPython_NumPy_INCLUDE_DIRS={sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/numpy/core/include ."
            )
            print("faiss - stage 2")
            run(f"{cmake} --build build -j {cores()} --target faiss")
            print("faiss - stage 3")
            run(f"{cmake} --build build -j {cores()} --target swigfaiss")
        temp_dir = tmp(dir=True)
        with cwd("./xfaiss/build/faiss/python"):
            print(
                "faiss - stage 4: Building wheel with numpy==1.26.4 constraint"
            )
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as reqs:
                reqs.write("numpy==1.26.4\n")
                constraints_path = reqs.name
            try:
                run(
                    f"{sys.executable} -m pip wheel . -w {temp_dir} -c {constraints_path}"
                )
            finally:
                os.remove(constraints_path)
        with cwd():
            delete("./xfaiss")
        free()
        any_wheel_path = paths(f"{temp_dir}/faiss-*.whl")[0]
        repaired_wheel_dir = tmp(dir=True)
        print("faiss - stage 5: Repairing wheel")
        run(
            f"{sys.executable} -m auditwheel repair {any_wheel_path} -w {repaired_wheel_dir}"
        )
        repaired_wheel_path = paths(f"{repaired_wheel_dir}/faiss-*.whl")[0]
        print(
            "faiss - stage 6: Modifying final wheel metadata for runtime constraints"
        )
        dependency_constraints = {"numpy": "==1.26.4"}
        final_wheel_path = modify_wheel_requirements(
            repaired_wheel_path, dependency_constraints
        )
        return final_wheel_path
    except subprocess.CalledProcessError as e:
        catch(f"Error during installation: {e}")
    except FileNotFoundError as e:
        catch(f"File not found error: {e}")
    except Exception as e:
        catch(f"An unexpected error occurred: {e}")


def exist(*p):
    joined = os.path.join(*[str(_p).strip() for _p in p])
    if not joined or not joined.strip():
        return False
    expanded = os.path.expanduser(joined)
    absolute = os.path.abspath(expanded)
    return os.path.exists(absolute)


def add_path(*p):
    import definers as _d

    joined = os.path.join(*[str(_p).strip() for _p in p]) if p else ""
    path = joined if joined == "" else full_path(*p)
    if path not in sys.path:
        _d.permit(path)
        sys.path.insert(0, path)
        site.addsitedir(path)
        importlib.invalidate_caches()
    if get_os_name() == "linux" or get_os_name() == "darwin":
        cmd = f'export PATH="{path}:$PATH"'
        if exist("~/.bashrc"):
            content = read("~/.bashrc")
            if content is not None:
                write("~/.bashrc", "\n".join([content, cmd]))
        elif exist("~/.zshrc"):
            content = read("~/.zshrc")
            if content is not None:
                write("~/.zshrc", "\n".join([content, cmd]))
        return run(cmd)
    if get_os_name() == "windows":
        return run(f'setx PATH "%PATH%;{path}"')


def normalize_path(p):
    return os.path.normpath(p)


def full_path(*p):
    joined = os.path.join(*[str(_p).strip() for _p in p])
    expanded = os.path.expanduser(joined)
    return os.path.abspath(expanded)


def paths(*patterns):
    import definers as _d

    patterns = [full_path(p) for p in patterns]
    path_list = []
    for p in patterns:
        try:
            lst = list(_d.glob(p, recursive=True))
            path_list = [*path_list, *lst]
        except Exception:
            pass
    return sorted(list(set(path_list)))


def copy(src, dst):
    src_path = Path(full_path(src))
    if src_path.is_symlink():
        resolved = src_path.resolve()
        if os.path.isdir(str(resolved)):
            shutil.copytree(
                str(src),
                str(dst),
                symlinks=False,
                ignore_dangling_symlinks=True,
            )
        else:
            shutil.copy(str(src), str(dst))
    elif os.path.isdir(str(src_path)):
        shutil.copytree(
            str(src), str(dst), symlinks=False, ignore_dangling_symlinks=True
        )
    else:
        shutil.copy(str(src), str(dst))


def big_number(zeros=10):
    return int("1" + "0" * zeros)


def find_package_paths(package_name):
    import definers as _d

    package_paths_found = []
    package_dir_name = package_name.replace("-", "_")
    site_packages_dirs = _d.site.getsitepackages()
    for site_packages_dir in site_packages_dirs:
        package_path = os.path.join(site_packages_dir, package_dir_name)
        if os.path.exists(package_path) and os.path.isdir(package_path):
            package_paths_found.append(package_path)
    for path in sys.path:
        if path:
            potential_package_path = os.path.join(path, package_dir_name)
            if os.path.exists(potential_package_path) and os.path.isdir(
                potential_package_path
            ):
                package_paths_found.append(potential_package_path)
    for site_packages_dir in site_packages_dirs:
        dist_packages_dir = site_packages_dir.replace(
            "site-packages", "dist-packages"
        )
        if dist_packages_dir != site_packages_dir:
            package_path = os.path.join(dist_packages_dir, package_dir_name)
            if os.path.exists(package_path) and os.path.isdir(package_path):
                package_paths_found.append(package_path)
    unique_paths = unique(package_paths_found)
    return unique_paths


def unique(arr):
    return sorted(list(set(arr)))


def tmp(suffix: str = ".data", keep: bool = True, dir=False):
    import definers as _d

    if dir:
        temp_dir_path = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir_path.__enter__()
        if not keep:
            _d.delete(temp_dir_path)
        return temp_dir_path
    if not suffix.startswith("."):
        if len(suffix.split(".")) > 1:
            suffix = suffix.split(".")
            suffix = suffix[len(suffix) - 1]
            if len(suffix) < 1:
                suffix = "tmp"
        suffix = "." + suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp_name = temp.name
    if not keep:
        _d.delete(temp_name)
    return temp_name


def get_process_pid(process_name):
    try:
        pid = int(subprocess.check_output(["pidof", process_name]).strip())
        return pid
    except subprocess.CalledProcessError:
        return None
    except ValueError:
        return None


def send_signal_to_process(pid, signal_number):
    try:
        os.kill(pid, signal_number)
        return True
    except OSError as e:
        print(f"Error sending signal: {e}")
        return False


@contextmanager
def cwd(dir=None):
    if not dir:
        dir = "."
    if not dir.startswith("/") and (not dir.startswith("~")):
        dir = full_path(os.path.dirname(__file__), dir)
    else:
        dir = full_path(dir)
    owd = full_path(os.getcwd())
    try:
        os.chdir(dir)
        yield dir
    finally:
        try:
            os.chdir(owd)
        except:
            pass


def log(subject, data, status=None):
    import definers as _d

    now = _d.datetime.now().time()
    if status is True:
        print(
            f"\n >>> {now} <<< \nOK OK OK OK OK OK OK\n{str(data)}\nOK OK OK OK OK OK OK\n >>> {subject} <<< \n"
        )
    elif status is False:
        print(
            f"\n >>> {now} <<< \nx ERR x ERR x ERR x\n{str(data)}\nx ERR x ERR x ERR x\n >>> {subject} <<< \n"
        )
    elif status is None:
        print(
            f"\n >>> {now} <<< \n===================\n{str(data)}\n===================\n >>> {subject} <<< \n"
        )
    elif isinstance(status, str) and status.strip() != "":
        print(
            f"\n >>> {now} <<< \n{status}\n{str(data)}\n{status}\n >>> {subject} <<< \n"
        )
    else:
        print(f"\n{now}\n{str(data)}\n{subject}\n")


def catch(e):
    import definers as _d

    _d.logger.exception(e)


def directory(dir, exist_ok=True):
    dir = full_path(str(dir))
    os.makedirs(dir, exist_ok=exist_ok)


def move(src, dest):
    src_path = full_path(str(src))
    if not exist(src_path):
        raise FileNotFoundError(f"Source path not found: {src}")
    copy(src, dest)
    delete(src)


def is_directory(*p):
    return Path(os.path.join(*[str(_p).strip() for _p in p])).is_dir()


def is_symlink(*p):
    return Path(os.path.join(*[str(_p).strip() for _p in p])).is_symlink()


def remove_readonly(func, path, excinfo):
    exception_instance = excinfo[1]

    if exception_instance.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise


def shutil_rmtree_readonly_handler(func, path, exc_info):
    exception_instance = (
        exc_info[1] if isinstance(exc_info, tuple) else exc_info
    )

    if exception_instance.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise


def delete(path):
    resolved = full_path(str(path))
    p = Path(resolved)
    if p.is_symlink():
        p.unlink()
        return
    if not exist(resolved):
        return
    if is_directory(resolved):
        try:
            shutil.rmtree(resolved, on_exc=shutil_rmtree_readonly_handler)
        except TypeError:
            shutil.rmtree(resolved, onerror=shutil_rmtree_readonly_handler)
    else:
        p.unlink(missing_ok=True)


def remove(path):
    delete(path)


def load(path):
    path = full_path(str(path))
    permit(path)
    if not exist(path):
        return None
    if is_directory(path):
        return sorted([p.name for p in Path(path).iterdir()])
    else:
        raw = Path(path).read_bytes()
        if b"\x00" in raw or not _is_text(raw):
            return raw
        try:
            return raw.decode("utf-8").replace("\r\n", "\n")
        except (UnicodeDecodeError, ValueError):
            return raw


def _is_text(data):
    if not data:
        return True
    text_chars = set(range(32, 127)) | {9, 10, 13} | set(range(128, 256))
    return all(b in text_chars for b in data[:8192])


def read(path):
    return load(path)


def write(path, txt=""):
    return save(path, txt)


def parent_directory(p, levels: int = 1):
    path = Path(str(p))
    for _ in range(levels):
        path = path.parent
    return os.path.normpath(str(path))


def save(path, text=""):
    os.makedirs(parent_directory(path), exist_ok=True)
    with open(path, "w+", encoding="utf8") as file:
        file.write(str(text))


def save_temp_text(text_content):
    if text_content is None:
        return None
    temp_path = tmp(".data")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    return temp_path


def run_linux(command, silent=False, env=None):
    import pty

    if env is None:
        env = {}
    original_env = os.environ.copy()
    modified_env = {**original_env, **env}
    if isinstance(command, list):
        command = "\n".join(command)
    in_lines = command.strip().splitlines()
    cmds = [i.strip() for i in in_lines if i.strip() != ""]
    if len(cmds) > 0:
        script = "\n".join(cmds)
        name = tmp(".sh")
        try:
            write(name, "#!/bin/bash --login\n" + script)
            permit(name)
            (master, slave) = pty.openpty()
            pid = os.fork()
            if pid == 0:
                os.setsid()
                try:
                    with open(os.devnull) as stdin:
                        os.dup2(stdin.fileno(), 0)
                    os.dup2(slave, 1)
                    os.dup2(slave, 2)
                    os.close(master)
                    os.close(slave)
                    os.environ.update(modified_env)
                    os.execl(
                        "/bin/bash", "/bin/bash", "--login", "-c", name, "&"
                    )
                except Exception as e:
                    print(f"Execution Error: {e}")
                finally:
                    delete(name)
                    os.environ.update(original_env)
                    os._exit(0)
            else:
                os.close(slave)
                output_bytes = b""
                output = ""
                while True:
                    (rlist, _, _) = select.select([master], [], [])
                    if master in rlist:
                        try:
                            chunk = os.read(master, 1024)
                            if not chunk:
                                break
                            output_bytes += chunk
                            try:
                                chunk_utf = chunk.decode(
                                    "utf-8", errors="replace"
                                )
                                if not silent:
                                    print(chunk_utf, end="", flush=True)
                                output += chunk_utf
                            except UnicodeDecodeError:
                                continue
                        except OSError:
                            break
                os.close(master)
                returncode = os.waitpid(pid, 0)[1] >> 8
                if returncode != 0:
                    if not silent:
                        log(f"Script failed [{returncode}]", script)
                    return False
                if not silent:
                    log("Script completed", script)
                out_lines = output.strip().splitlines()
                ret_lines = [o.strip() for o in out_lines if o.strip() != ""]
                return ret_lines
        except OSError as e:
            catch(e)
            return False
    return False


def run_windows(command, silent=False, env=None):
    try:
        if env is None:
            env = {}
        if isinstance(command, list):
            cmds = command
            command_to_run = " && ".join([c.strip() for c in cmds if c.strip()])
        else:
            cmds = command.strip().splitlines()
            if len(cmds) > 1:
                command_to_run = " && ".join(
                    [c.strip() for c in cmds if c.strip()]
                )
            else:
                command_to_run = command.strip()
        if not command_to_run:
            return False
        modified_env = {**os.environ.copy(), **env}
        process = subprocess.Popen(
            command_to_run,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=modified_env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (stdout, stderr) = process.communicate()
        if stdout is None:
            stdout = ""
        if stderr is None:
            stderr = ""
        returncode = process.returncode
        if not silent:
            if stdout:
                print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", flush=True)
        if returncode != 0:
            if not silent:
                log(f"Script failed [{returncode}]", command_to_run)
                log(f"Stderr: {stderr.strip()}", "")
            return False
        else:
            if not silent:
                log("Script completed", command_to_run)
            out_lines = stdout.strip().splitlines()
            ret_lines = [o.strip() for o in out_lines if o.strip()]
            return ret_lines
    except Exception as e:
        catch(e)
        return False


def run(command, silent=False, env=None):
    import definers as _d

    if env is None:
        env = {}
    if sys.platform.startswith("win"):
        return _d.run_windows(command, silent=silent, env=env)
    else:
        return _d.run_linux(command, silent=silent, env=env)


def thread(func, *args, **kwargs):

    def _wrapper(*a, **kw):
        try:
            func(*a, **kw)
        except Exception as e:
            catch(e)

    t = threading.Thread(target=_wrapper, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


def wait(*threads):
    for t in threads:
        t.join()


def permit(path):
    try:
        if not exist(path):
            return False
        if get_os_name() == "linux":
            subprocess.run(["chmod", "-R", "a+xrw", path], check=True)
            return True
        if get_os_name() == "windows":
            subprocess.run(
                ["icacls", path, "/grant", "Everyone:F", "/T"], check=True
            )
            return True
        return False
    except Exception:
        return False


def check_version_wildcard(version_spec, version_actual):
    version_spec = version_spec.replace(".", "\\.").replace("*", ".*")
    pattern = re.compile(f"^{version_spec}$")
    return bool(pattern.match(version_actual))


def installed(pack, version=None):
    import definers as _d

    pack_lower = pack.lower().strip()
    version_lower = None
    if version:
        version_lower = version.lower().strip()
    system = _d.get_os_name()
    if system == "windows":
        cmd = 'powershell.exe -Command "Get-ItemProperty HKLM:\\Software\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Select-Object DisplayName, DisplayVersion | Format-Table -HideTableHeaders'
        try:
            lines = _d.run(cmd, silent=True)
            for line in lines:
                parts = re.split("\\s{2,}", line.strip())
                if not parts or not parts[0]:
                    continue
                name = parts[0].lower().strip()
                ver = parts[1].strip() if len(parts) > 1 else ""
                if pack_lower in name:
                    if (
                        version_lower is None
                        or (ver and ver.startswith(version_lower))
                        or (
                            ver
                            and "*" in version_lower
                            and check_version_wildcard(version_lower, ver)
                        )
                    ):
                        return True
        except Exception:
            pass
    elif system == "linux":
        which_result = _d.shutil.which(pack)
        if which_result:
            if version_lower is None:
                return True
            try:
                lines = _d.run(f"{pack} --version", silent=True)
                if not lines:
                    lines = _d.run(f"{pack} -v", silent=True)
                if lines:
                    match = re.search("(\\d+\\.\\d+(\\.\\d+)*)", lines[0])
                    if match:
                        actual_version = match.group(0)
                        if actual_version.startswith(version_lower) or (
                            "*" in version_lower
                            and check_version_wildcard(
                                version_lower, actual_version
                            )
                        ):
                            return True
            except Exception:
                pass
    try:
        lines = _d.run("pip list", silent=True)
        if lines:
            for line in lines:
                parts = re.sub("( ){2,}", ";", line).split(";")
                if len(parts) == 2:
                    n = parts[0].lower().strip()
                    v = parts[1].lower().strip()
                    if n == pack_lower and (
                        version_lower is None
                        or v.startswith(version_lower)
                        or (
                            "*" in version_lower
                            and check_version_wildcard(version_lower, v)
                        )
                    ):
                        return True
                else:
                    continue
        return False
    except subprocess.CalledProcessError:
        raise
    except FileNotFoundError:
        return False


def importable(name):
    if not isinstance(name, str):
        return False
    module_name = name.strip()
    if not module_name:
        return False
    try:
        module_specification = importlib.util.find_spec(module_name)
        if module_specification is None:
            return False
        return True
    except Exception:
        return False


def runnable(cmd):
    if not isinstance(cmd, str):
        return False
    command_line = cmd.strip()
    if not command_line:
        return False
    try:
        command_parts = shlex.split(command_line, posix=False)
    except ValueError:
        command_parts = command_line.split()
    if len(command_parts) == 0:
        return False
    command_name = command_parts[0].strip('"').strip("'")
    if not command_name:
        return False
    return shutil.which(command_name) is not None


def is_package_path(package_path, package_name=None):
    if (
        exist(package_path)
        and os.path.isdir(package_path)
        and (
            os.path.exists(os.path.join(package_path, "__init__.py"))
            or os.path.exists(
                os.path.join(package_path, os.path.basename(package_path))
            )
            or os.path.exists(os.path.join(package_path, "src"))
        )
        and (
            package_name is None
            or package_name == os.path.basename(package_path)
        )
    ):
        return True
    return False


def get_python_version():
    import definers as _d

    try:
        version_info = _d.sys.version_info
        if not hasattr(version_info, "major"):
            raise AttributeError(
                "sys.version_info is missing essential version attributes"
            )
        major = version_info.major
        minor = getattr(version_info, "minor", 0)
        micro = getattr(version_info, "micro", 0)
        version_str = f"{major}.{minor}.{micro}"
        return version_str
    except Exception as e:
        print(f"Error getting Python version: {e}")
        return None


def get_linux_distribution():
    import definers as _d

    try:
        try:
            _d.subprocess.run(
                ["apt-get", "update"],
                capture_output=True,
                text=True,
                check=True,
            )
            _d.subprocess.run(
                ["apt-get", "install", "-y", "lsb_release"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = _d.subprocess.run(
                ["lsb_release", "-a"],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout
            distro_match = re.search("Distributor ID:\\s*([^\\n]+)", output)
            release_match = re.search("Release:\\s*([^\\n]+)", output)
            if distro_match and release_match:
                distro = distro_match.group(1).strip().lower().split(" ")[0]
                release = release_match.group(1).strip()
                return (distro, release)
            else:
                return (None, None)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        try:
            with open("/etc/os-release") as f:
                os_release_content = f.read()
            name_match = re.search('NAME="([^"]+)"', os_release_content)
            version_match = re.search(
                'VERSION_ID="([^"]+)"', os_release_content
            )
            if name_match and version_match:
                distro = name_match.group(1).strip()
                release = version_match.group(1).strip()
                return (distro, release)
        except FileNotFoundError:
            pass
        return (None, None)
    except Exception:
        return (None, None)


def cores():
    return os.cpu_count()


def get_ext(input_path):
    return str(Path(str(input_path)).suffix).strip(".").lower()


def is_ai_model(input_path):
    extension = get_ext(input_path)
    return extension in ai_model_extensions


def compress(dir: str, format: str = "zip", keep_name: bool = True):
    if keep_name:
        target = str(Path(dir).parent) + "/" + str(Path(dir).name)
    else:
        from definers import random_string

        target = str(Path(dir).parent) + "/" + random_string()
    shutil.make_archive(
        target, format, str(Path(dir).parent), str(Path(dir).name)
    )
    return target + "." + format


def extract(arcv, dest=None, format=None):
    if not dest:
        dest = str(Path(arcv).parent)
    if format:
        shutil.unpack_archive(arcv, dest, format)
    else:
        shutil.unpack_archive(arcv, dest)


def path_end(p: str):
    return Path(p.rstrip("/").rstrip("\\")).name


def path_ext(p):
    if is_directory(p):
        return None
    try:
        return "".join(Path(p).suffixes)
    except:
        return None


def path_name(p):
    return str(Path(p).stem)


def pre_install():
    import pathlib

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

    import definers

    try:
        import torch.fx.experimental.proxy_tensor as proxy_mod

        if not hasattr(proxy_mod, "get_proxy_mode"):
            proxy_mod.get_proxy_mode = lambda: None
    except Exception:
        pass
    if not hasattr(_np, "_no_nep50_warning"):
        _np._no_nep50_warning = lambda *a, **_kw: None
    definers.free()


def install_faiss():
    import definers

    if definers.importable("faiss"):
        return False
    try:
        faiss_dir = "_faiss_"
        with definers.cwd() as d:
            faiss_dir = d
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/faiss.git",
                faiss_dir,
            ],
            check=True,
        )
        py = definers.sys.executable
        prefix = definers.sys.prefix
        major = definers.sys.version_info.major
        minor = definers.sys.version_info.minor
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
            ["make", "-C", f"{faiss_dir}/build", "-j16", "faiss"], check=True
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
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def apt_install(upgrade=False):
    import definers

    definers.pre_install()
    basic_apt = "build-essential gcc cmake swig gdebi git git-lfs wget curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev initramfs-tools libgirepository1.0-dev libdbus-1-dev libdbus-glib-1-dev libsecret-1-0 libmanette-0.2-0 libharfbuzz0b libharfbuzz-icu0 libenchant-2-2 libhyphen0 libwoff1 libgraphene-1.0-0 libxml2-dev libxmlsec1-dev"
    audio_apt = "libportaudio2 libasound2-dev sox libsox-fmt-all praat ffmpeg libavcodec-extra libavif-dev"
    visual_apt = "libopenblas-dev libgflags-dev libgles2 libgtk-3-0 libgtk-4-1 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libatspi2.0-0 libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-gl"
    definers.run("apt-get update")
    definers.run(f"apt-get install -y {basic_apt} {audio_apt} {visual_apt}")
    if upgrade:
        definers.run("apt-get upgrade -y")
    definers.post_install()
