import contextlib
import sys
import types


def test_faiss_python_cmake_args_follow_sysconfig_and_numpy_layout(
    monkeypatch,
    tmp_path,
):
    import definers.system.installation as installation_module

    include_dir = tmp_path / "python-include"
    lib_dir = tmp_path / "python-libs"
    numpy_dir = tmp_path / "numpy" / "_core" / "include"
    include_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)
    numpy_dir.mkdir(parents=True)
    library_path = lib_dir / "python313.lib"
    library_path.write_text("")
    fake_executable = tmp_path / "Python 3.13" / "python.exe"

    monkeypatch.setattr(
        installation_module.sys,
        "executable",
        str(fake_executable),
    )
    monkeypatch.setattr(
        installation_module.sys,
        "prefix",
        str(tmp_path / "venv"),
    )
    monkeypatch.setattr(
        installation_module.sys,
        "base_prefix",
        str(tmp_path / "base"),
    )
    monkeypatch.setattr(
        installation_module.sysconfig,
        "get_path",
        lambda name: {
            "include": str(include_dir),
            "platinclude": str(include_dir),
        }.get(name),
    )
    monkeypatch.setattr(
        installation_module.sysconfig,
        "get_config_var",
        lambda name: {
            "ABIFLAGS": "",
            "INCLUDEPY": str(include_dir),
            "LIBDIR": str(lib_dir),
            "LIBPL": str(lib_dir),
            "LDLIBRARY": "python313.lib",
            "LIBRARY": "python313.lib",
            "py_version_nodot": "313",
        }.get(name),
    )
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(get_include=lambda: str(numpy_dir)),
    )

    args = installation_module._faiss_python_cmake_args()

    assert f"-DPython_EXECUTABLE={fake_executable}" in args
    assert f"-DPython_INCLUDE_DIR={include_dir}" in args
    assert f"-DPython_LIBRARY={library_path}" in args
    assert f"-DPython_NumPy_INCLUDE_DIRS={numpy_dir}" in args
    assert all("site-packages/numpy/core/include" not in arg for arg in args)
    assert all("/include/python3.13" not in arg for arg in args)


def test_system_build_faiss_uses_shared_python_cmake_args(
    monkeypatch,
    tmp_path,
):
    import definers.ml as ml_module
    import definers.system as system_module
    import definers.system.installation as installation_module

    captured_commands = []

    @contextlib.contextmanager
    def fake_cwd(_path=None):
        yield str(tmp_path)

    def fake_run(command, *args, **kwargs):
        captured_commands.append(command)
        raise RuntimeError("stop")

    monkeypatch.setattr(system_module, "cwd", fake_cwd)
    monkeypatch.setattr(system_module, "run", fake_run)
    monkeypatch.setattr(system_module, "catch", lambda *args, **kwargs: None)
    monkeypatch.setattr(ml_module, "git", lambda *args, **kwargs: None)
    monkeypatch.setattr("definers.cuda.set_cuda_env", lambda: None)
    monkeypatch.setattr("definers.cuda.free", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        installation_module,
        "_faiss_python_cmake_args",
        lambda: [
            "-DPython_EXECUTABLE=C:/Program Files/Python313/python.exe",
            "-DPython_INCLUDE_DIR=C:/Python/include",
        ],
    )

    installation_module.build_faiss()

    assert isinstance(captured_commands[0], list)
    assert captured_commands[0][1:3] == ["-B", "build"]
    assert (
        "-DPython_EXECUTABLE=C:/Program Files/Python313/python.exe"
        in captured_commands[0]
    )
    assert "-DPython_INCLUDE_DIR=C:/Python/include" in captured_commands[0]


def test_install_faiss_uses_shared_python_cmake_args(monkeypatch, tmp_path):
    import definers.system as system_module
    import definers.system.installation as installation_module

    recorded_commands = []

    @contextlib.contextmanager
    def fake_cwd(_path=None):
        yield str(tmp_path)

    def fake_subprocess_run(command, *args, **kwargs):
        recorded_commands.append(command)
        if command and command[0] == "cmake":
            raise RuntimeError("stop")
        return None

    monkeypatch.setattr(system_module, "importable", lambda name: False)
    monkeypatch.setattr(system_module, "cwd", fake_cwd)
    monkeypatch.setattr(
        installation_module.subprocess, "run", fake_subprocess_run
    )
    monkeypatch.setattr(
        installation_module,
        "_faiss_python_cmake_args",
        lambda: [
            "-DPython_EXECUTABLE=C:/Program Files/Python314/python.exe",
            "-DPython_INCLUDE_DIR=C:/Python/include",
        ],
    )

    installation_module.install_faiss()

    assert recorded_commands[1][0] == "cmake"
    assert (
        "-DPython_EXECUTABLE=C:/Program Files/Python314/python.exe"
        in recorded_commands[1]
    )
    assert "-DPython_INCLUDE_DIR=C:/Python/include" in recorded_commands[1]


def test_ml_build_faiss_uses_shared_python_cmake_args(monkeypatch, tmp_path):
    import definers.ml as ml_module
    import definers.system.installation as installation_module

    captured_commands = []

    @contextlib.contextmanager
    def fake_cwd(_path=None):
        yield str(tmp_path)

    def fake_run(command, *args, **kwargs):
        captured_commands.append(command)
        raise RuntimeError("stop")

    monkeypatch.setattr(ml_module, "cwd", fake_cwd)
    monkeypatch.setattr(ml_module, "run", fake_run)
    monkeypatch.setattr(ml_module, "catch", lambda *args, **kwargs: None)
    monkeypatch.setattr(ml_module, "git", lambda *args, **kwargs: None)
    monkeypatch.setattr(ml_module, "set_cuda_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(ml_module, "free", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "definers.system.download_activity.create_activity_reporter",
        lambda total: lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "definers.system.output_paths.managed_output_session_dir",
        lambda section, stem=None: str(tmp_path / (stem or section)),
    )
    monkeypatch.setattr(
        installation_module,
        "_faiss_python_cmake_args",
        lambda: [
            "-DPython_EXECUTABLE=C:/Program Files/Python313/python.exe",
            "-DPython_INCLUDE_DIR=C:/Python/include",
        ],
    )

    ml_module.build_faiss()

    assert isinstance(captured_commands[0], list)
    assert captured_commands[0][1:3] == ["-B", "build"]
    assert (
        "-DPython_EXECUTABLE=C:/Program Files/Python313/python.exe"
        in captured_commands[0]
    )
    assert "-DPython_INCLUDE_DIR=C:/Python/include" in captured_commands[0]
