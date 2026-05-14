import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from definers.runtime_numpy import get_array_module, get_numpy_module

_np = get_numpy_module()
np = get_array_module()

from definers import regex_utils
from definers.ml.analysis import (
    compile_model,
    get_model_instructions,
    kmeans_k_suggestions,
)
from definers.ml.answer.service import answer
from definers.ml.health_api import (
    get_ml_health_snapshot,
    ml_health_markdown,
    validate_ml_health,
)
from definers.ml.inference import (
    extract_text_features,
    features_to_text,
)
from definers.ml.introspection import (
    get_cluster_content,
    is_clusters_model,
    lang_code_to_name,
)
from definers.ml.regression_api import (
    initialize_linear_regression,
    linear_regression,
    predict_linear_regression,
    train_linear_regression,
)
from definers.ml.runtime import (
    SklearnWrapper,
    check_parameter,
    choose_random_words,
    pipe,
    validate_str_param,
)
from definers.ml.rvc import find_latest_rvc_checkpoint
from definers.ml.text.api import (
    map_reduce_summary,
    optimize_prompt_realism,
    preprocess_prompt,
    summarize,
    summary,
)
from definers.ml.training import (
    HybridModel,
    LinearRegressionTorch,
    feed,
    fit,
)

from . import (
    contracts,
    health,
    health_api,
    inference,
    introspection,
    regression_api,
    regression_predictor,
    repository_sync,
    rvc,
    safe_deserialization,
    text as _text,
    trainer_plan,
    training,
)

try:
    from definers.audio import (
        features_to_audio,
        normalize_audio_to_peak,
        predict_audio,
        separate_stems,
        stem_mixer,
    )
except Exception:
    features_to_audio = None
    normalize_audio_to_peak = None
    predict_audio = None
    separate_stems = None
    stem_mixer = None
from definers.constants import (
    MODELS,
    PROCESSORS,
    TOKENIZERS,
    common_audio_formats,
    tasks,
)

try:
    from definers.cuda import device, free, set_cuda_env
except Exception:

    def device():
        return "cpu"

    def free(*args, **kwargs):
        return None

    def set_cuda_env(*args, **kwargs):
        return None


try:
    from definers.data.arrays import (
        cupy_to_numpy,
        dtype,
        get_max_shapes,
        guess_numpy_type,
        numpy_to_cupy,
        one_dim_numpy,
        reshape_numpy,
    )
    from definers.data.exports import (
        get_prediction_file_extension,
    )
    from definers.data.loaders import load_as_numpy
    from definers.data.vectorizers import (
        create_vectorizer,
    )
except Exception:
    create_vectorizer = None
    cupy_to_numpy = None
    dtype = None
    get_prediction_file_extension = None
    get_max_shapes = None
    guess_numpy_type = None
    load_as_numpy = None
    numpy_to_cupy = None
    one_dim_numpy = None
    reshape_numpy = None
try:
    from definers.image import (
        features_to_image,
        save_image,
    )
except Exception:
    features_to_image = None
    save_image = None
from definers.logger import init_logger

try:
    from definers.system import (
        add_path,
        big_number,
        catch,
        copy,
        cores,
        cwd,
        delete,
        directory,
        exist,
        full_path,
        get_ext,
        get_os_name,
        is_directory,
        log,
        modify_wheel_requirements,
        move,
        normalize_path,
        path_end,
        path_ext,
        paths,
        read,
        run,
        secure_path,
        thread,
        tmp,
        wait,
        write,
    )
except Exception:

    def _missing_runtime(*args, **kwargs):
        raise RuntimeError("definers.system is unavailable")

    add_path = _missing_runtime
    big_number = _missing_runtime

    def catch(*args, **kwargs):
        return None

    copy = _missing_runtime
    cores = _missing_runtime
    cwd = _missing_runtime
    delete = _missing_runtime
    directory = _missing_runtime
    exist = _missing_runtime
    full_path = _missing_runtime
    get_ext = _missing_runtime
    get_os_name = _missing_runtime
    is_directory = _missing_runtime

    def log(*args, **kwargs):
        return None

    modify_wheel_requirements = _missing_runtime
    move = _missing_runtime
    normalize_path = _missing_runtime
    path_end = _missing_runtime
    path_ext = _missing_runtime
    paths = _missing_runtime
    read = _missing_runtime
    run = _missing_runtime
    secure_path = _missing_runtime
    thread = _missing_runtime
    tmp = _missing_runtime
    wait = _missing_runtime
    write = _missing_runtime
try:
    from definers.text import (
        random_string,
        simple_text,
    )
except Exception:
    random_string = None
    simple_text = None
try:
    from definers.video.helpers import features_to_video, write_video
except Exception:
    features_to_video = None
    write_video = None
try:
    from definers.media.web_transfer import (
        download_file,
        google_drive_download,
    )
except Exception:
    download_file = None
    google_drive_download = None

_FAILED_MODEL_LOADS: dict[str, str] = {}

logger = init_logger("definers.ml")


@dataclass(frozen=True, slots=True)
class _TrainingArrayAdapter:
    catch: object
    cupy_to_numpy: object
    get_max_shapes: object
    numpy_to_cupy: object
    reshape_numpy: object


def _rvc_package_root() -> str:
    return str(Path(__file__).resolve().parent.parent)


def _training_array_adapter():
    return _TrainingArrayAdapter(
        catch=catch,
        cupy_to_numpy=cupy_to_numpy,
        get_max_shapes=get_max_shapes,
        numpy_to_cupy=numpy_to_cupy,
        reshape_numpy=reshape_numpy,
    )


def _concatenate_training_rows():
    cupy_module = getattr(np, "cuda", None)
    if cupy_module is not None:
        concatenate_module = getattr(cupy_module, "cupy", None)
        if concatenate_module is not None:
            concatenate = getattr(concatenate_module, "concatenate", None)
            if concatenate is not None:
                return concatenate
    return np.concatenate


def _normalize_model_task(task: str) -> str:
    task_text = str(task).strip()
    if not task_text:
        raise ValueError("task is required")

    if task_text in tasks:
        return task_text

    if repository_sync.is_huggingface_reference(
        task_text
    ) or repository_sync.is_http_url(task_text):
        return task_text

    if (
        task_text.startswith(("/", ".", "~"))
        or os.sep in task_text
        or (os.altsep is not None and os.altsep in task_text)
        or re.match(r"^[a-zA-Z]:[\\/]", task_text) is not None
    ):
        raise ValueError(f"Unsupported task reference: {task_text!r}")

    raise ValueError(f"Unsupported task reference: {task_text!r}")


def init_model_file(task: str, turbo: bool = True, model_type: str = None):
    normalized_task = _normalize_model_task(task)
    return repository_sync.init_model_file(
        normalized_task,
        turbo=turbo,
        model_type=model_type,
    )


def build_faiss():
    from pathlib import Path

    from definers.system.download_activity import create_activity_reporter
    from definers.system.installation import _faiss_python_cmake_args
    from definers.system.output_paths import managed_output_session_dir

    report = create_activity_reporter(7)
    source_dir = Path(managed_output_session_dir("faiss", stem="source"))
    wheel_dir = managed_output_session_dir("faiss", stem="wheel")
    repaired_wheel_dir = managed_output_session_dir("faiss", stem="repair")

    report(
        1,
        "Clone FAISS source",
        detail="Downloading the FAISS source repository.",
    )
    git("YaronKoresh", "faiss", parent=str(source_dir))
    set_cuda_env()
    cmake = shutil.which("cmake") or "/usr/local/cmake/bin/cmake"
    try:
        with cwd(str(source_dir)):
            print("faiss - stage 1")
            report(
                2,
                "Configure FAISS build",
                detail="Configuring the CMake build directory.",
            )
            run(
                [
                    cmake,
                    "-B",
                    "build",
                    "-DBUILD_TESTING=OFF",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DFAISS_ENABLE_MKL=OFF",
                    "-DFAISS_ENABLE_C_API=ON",
                    "-DFAISS_ENABLE_GPU=ON",
                    "-DFAISS_ENABLE_PYTHON=ON",
                    *_faiss_python_cmake_args(),
                    ".",
                ]
            )
            print("faiss - stage 2")
            report(
                3,
                "Build FAISS core",
                detail="Compiling the FAISS core library.",
            )
            run(
                [
                    cmake,
                    "--build",
                    "build",
                    "-j",
                    str(cores()),
                    "--target",
                    "faiss",
                ]
            )
            print("faiss - stage 3")
            report(
                4,
                "Build Python bindings",
                detail="Compiling the FAISS Python bindings.",
            )
            run(
                [
                    cmake,
                    "--build",
                    "build",
                    "-j",
                    str(cores()),
                    "--target",
                    "swigfaiss",
                ]
            )
        with cwd(str(source_dir / "build" / "faiss" / "python")):
            print(
                "faiss - stage 4: Building wheel with numpy==1.26.4 constraint"
            )
            report(
                5,
                "Build wheel",
                detail="Building the FAISS wheel artifact.",
            )
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as reqs:
                reqs.write("numpy==1.26.4\n")
                constraints_path = reqs.name
            try:
                run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "wheel",
                        ".",
                        "-w",
                        wheel_dir,
                        "-c",
                        constraints_path,
                    ]
                )
            finally:
                os.remove(constraints_path)
        delete(str(source_dir))
        free()
        any_wheel_path = paths(f"{wheel_dir}/faiss-*.whl")[0]
        print("faiss - stage 5: Repairing wheel")
        report(
            6,
            "Repair wheel",
            detail="Repairing platform-specific wheel metadata.",
        )
        run(
            [
                sys.executable,
                "-m",
                "auditwheel",
                "repair",
                any_wheel_path,
                "-w",
                repaired_wheel_dir,
            ]
        )
        repaired_wheel_path = paths(f"{repaired_wheel_dir}/faiss-*.whl")[0]
        print(
            "faiss - stage 6: Modifying final wheel metadata for runtime constraints"
        )
        report(
            7,
            "Finalize wheel",
            detail="Applying final dependency constraints to the wheel.",
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


def init_custom_model(model_type: str, path: str | list):
    from definers.ml.safe_deserialization import (
        load_serialized_model,
    )

    if not path or model_type not in ("onnx", "pkl"):
        return None

    if not isinstance(path, str) and not isinstance(path, list):
        raise ValueError("model path must be a string or a list")

    try:
        path = secure_path(path)

        if model_type == "onnx":
            with open(path, "rb") as f:
                import onnx

                model = onnx.load(f)
        else:
            model = load_serialized_model(path, model_type)

        return model

    except Exception as e:
        catch(f"Error initializing model: {e}")
        return None


def git(user: str, repo: str, branch: str = "main", parent: str = "."):
    import requests

    from definers.system import secure_path

    user = user.replace(" ", "_")
    repo = repo.replace(" ", "-")

    try:
        parent = secure_path(parent)
    except Exception as e:
        raise ValueError(f"Invalid parent path for git(): {e}")
    directory(parent)
    clone_dir = tmp(dir=True)
    repo_url = f"https://github.com/{user}/{repo}.git"
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    if not isinstance(branch, str) or not re.fullmatch(
        r"[A-Za-z0-9._/\-]+", branch
    ):
        raise ValueError(f"Invalid git branch name: {branch!r}")

    run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            repo_url,
            clone_dir,
        ],
        env=env,
    )

    def _lfs(_dir):
        entries = read(_dir)
        if entries is None:
            return
        for p in entries:
            if is_directory(p) and path_end(p) == ".git":
                continue
            if is_directory(p):
                _lfs(p)
                continue
            content = None
            try:
                content = read(p)
            except Exception as e:
                catch(e)
                continue
            try:
                if content.startswith(
                    "version https://git-lfs.github.com/spec"
                ):
                    filepath_in_repo = normalize_path(
                        os.path.relpath(p, clone_dir)
                    )
                    asset_url = f"https://media.githubusercontent.com/media/{user}/{repo}/{branch}/{filepath_in_repo}"
                    try:
                        download_file(asset_url, p)
                    except Exception as e3:
                        print(
                            f"Warning: Could not download asset '{filepath_in_repo}'. Error: {e3}"
                        )
            except Exception:
                continue

    _lfs(clone_dir)
    ps = paths(f"{clone_dir}/*")
    for p in ps:
        n = path_end(p)
        if n == ".git":
            continue
        move(p, f"{parent}/{n}")
    delete(clone_dir)


def init_model_repo(task: str, turbo: bool = True):
    import torch

    from definers.optional_dependencies import ensure_ml_task_runtime

    global MODELS
    global TOKENIZERS
    free()
    model = None
    ensure_ml_task_runtime(task)
    if task in ["translate"]:
        import nltk
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from definers.model_installation import hf_snapshot_download

        nltk.download("punkt_tab")
        local_repo_path = hf_snapshot_download(
            repo_id=tasks[task],
            item_label=str(tasks[task]),
            detail="Downloading translation model source files.",
        )
        TOKENIZERS[task] = AutoTokenizer.from_pretrained(local_repo_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_repo_path).to(
            device()
        )
    elif task in ["tts"]:
        from definers.audio.text_to_speech import LocalTextToSpeech

        model = LocalTextToSpeech.from_pretrained(device_name=device())
    elif task in ["svc"]:
        from definers.model_installation import (
            download_rvc_assets,
            has_enhanced_rvc_fork_folders,
        )

        rvc_root = _rvc_package_root()
        if not has_enhanced_rvc_fork_folders(rvc_root):
            logger.info(
                "Initializing RVC by cloning the enhanced fork with LFS assets into %s.",
                rvc_root,
            )
            download_rvc_assets(rvc_root)
            log("RVC initialization", "Initialization complete.", True)
        else:
            log(
                "RVC initialization",
                "RVC files already exist, skipping initialization.",
                True,
            )
        return None
    elif task in ["speech-recognition"]:
        from transformers import pipeline

        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            repo_id=tasks["speech-recognition"],
            item_label=str(tasks["speech-recognition"]),
            detail="Downloading speech recognition model source files.",
        )

        model = pipeline(
            "automatic-speech-recognition",
            model=local_repo_path,
            device=device(),
        )
    elif task in ["audio-classification"]:
        from transformers import pipeline

        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            repo_id=tasks["audio-classification"],
            item_label=str(tasks["audio-classification"]),
            detail="Downloading audio classification model source files.",
        )

        model = pipeline(
            "audio-classification",
            model=local_repo_path,
            device=device(),
        )
    elif task in ["detect"]:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            T5ForConditionalGeneration,
            T5Tokenizer,
            TFAutoModel,
            pipeline,
        )

        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            repo_id=tasks[task],
            item_label=str(tasks[task]),
            detail="Downloading detection model source files.",
        )
        config = AutoConfig.from_pretrained(local_repo_path)
        try:
            model = AutoModel.from_pretrained(
                local_repo_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype(),
            ).to(device())
        except:
            model = TFAutoModel.from_pretrained(
                local_repo_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype(),
            ).to(device())
    elif task in ["music"]:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            repo_id=tasks[task],
            item_label=str(tasks[task]),
            detail="Downloading music generation model source files.",
        )

        PROCESSORS[task] = AutoProcessor.from_pretrained(local_repo_path)
        model = MusicgenForConditionalGeneration.from_pretrained(
            local_repo_path
        ).to(device())
    elif task in ["answer"]:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
        )

        from definers.model_installation import (
            hf_file_download,
            hf_snapshot_download,
        )

        package_name = "phi4_package"
        print(f"Downloading source files for {tasks[task]}...")
        not_win = get_os_name() != "windows"
        snapshot_dir = Path(
            hf_snapshot_download(
                repo_id=tasks[task],
                allow_patterns=["*.txt", "*.py", "*.json", "*.safetensors"],
                revision="33e62acdd07cd7d6635badd529aa0a3467bb9c6a",
                item_label=str(tasks[task]),
                detail="Downloading answer model source files.",
                local_dir_use_symlinks=not_win,
            )
        )
        print(f"Source files downloaded to: {snapshot_dir}")
        prepare_inputs_for_generation_code = '\n    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):\n        if past_key_values:\n            input_ids = input_ids[:, -1:]\n\n        return {\n            "input_ids": input_ids,\n            "past_key_values": past_key_values,\n            "attention_mask": attention_mask,\n            **kwargs,\n        }\n'
        target_file = snapshot_dir / "modeling_phi4mm.py"
        target_class_line = "class Phi4MMModel(Phi4MMPreTrainedModel):"
        print(f"Preparing to inject patch into {target_file}...")
        original_code_lines = target_file.read_text().splitlines()
        if any(
            "dynamically injected to fix a compatibility issue" in line
            for line in original_code_lines
        ):
            print(
                "âœ… Source code appears to be already patched. Skipping injection."
            )
        else:
            try:
                line_number = original_code_lines.index(target_class_line)
                injection_point = line_number + 1
                original_code_lines.insert(
                    injection_point, prepare_inputs_for_generation_code
                )
                patched_code = "\n".join(original_code_lines)
                target_file.write_text(patched_code)
                print(
                    "âœ…âœ…âœ… SUCCESS: Method injected directly into source code."
                )
            except ValueError:
                print(
                    "âš ï¸ Could not find the target class declaration line. Aborting patch."
                )
        print("Rewriting relative imports to absolute...")
        for py_file in snapshot_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            modified_content = regex_utils.sub(
                r"from \\.(\\w_+)", "from \\1", content
            )
            modified_content = regex_utils.sub(
                "import \\.([\\w_]+)", "import \\1", modified_content
            )
            if content != modified_content:
                print(f"  - Rewrote imports in {py_file.name}")
                py_file.write_text(modified_content)
        py_modules = [p.stem for p in snapshot_dir.glob("*.py")]
        print(py_modules)
        pyproject_toml_content = f'\n[build-system]\nrequires = ["setuptools>=61.0"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "{package_name}"\nversion = "0.0.1"\ndescription = "A dynamically generated package for the Phi-4 model code."\n\n[tool.setuptools]\npy-modules = {py_modules}\n        '
        (snapshot_dir / "pyproject.toml").write_text(pyproject_toml_content)
        print("Dynamically created pyproject.toml file.")
        print(f"Installing '{package_name}' from local source...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(snapshot_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Successfully installed '{package_name}'.")
        str_snapshot_dir = str(snapshot_dir)
        add_path(str_snapshot_dir)
        print("Loading tokenizer, processor, and model via patched loader...")
        AutoTokenizer.from_pretrained(str_snapshot_dir)
        PROCESSORS["answer"] = AutoProcessor.from_pretrained(
            str_snapshot_dir, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str_snapshot_dir,
            torch_dtype=dtype(),
            trust_remote_code=True,
            _attn_implementation="eager",
        ).to(device())
        print("âœ… Phi-4 model loaded successfully!")
    elif task in ["summary"]:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            repo_id=tasks[task],
            item_label=str(tasks[task]),
            detail="Downloading summary model source files.",
        )
        TOKENIZERS[task] = T5Tokenizer.from_pretrained(local_repo_path)
        free()
        model = T5ForConditionalGeneration.from_pretrained(
            local_repo_path, torch_dtype=dtype()
        ).to(device())
    elif task in ["video"]:
        import torch
        from diffusers import (
            HunyuanVideoImageToVideoPipeline,
            HunyuanVideoTransformer3DModel,
        )

        from definers.model_installation import hf_snapshot_download

        local_repo_path = hf_snapshot_download(
            repo_id=tasks[task],
            item_label=str(tasks[task]),
            detail="Downloading video generation model source files.",
        )
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            local_repo_path, subfolder="transformer", torch_dtype=dtype()
        )
        model = HunyuanVideoImageToVideoPipeline.from_pretrained(
            local_repo_path, transformer=transformer, torch_dtype=dtype()
        )
        model.to(device())
    elif task in ["image"]:
        import torch
        from diffusers import FluxPipeline
        from safetensors.torch import load_file

        from definers.model_installation import (
            hf_file_download,
            hf_snapshot_download,
        )

        local_repo_path = hf_snapshot_download(
            repo_id=tasks[task],
            item_label=str(tasks[task]),
            detail="Downloading image generation model source files.",
        )
        model = FluxPipeline.from_pretrained(
            local_repo_path, torch_dtype=dtype(), use_safetensors=True
        )
        srpo_path = hf_file_download(
            repo_id=tasks["image-spro"],
            filename="diffusion_pytorch_model.safetensors",
            item_label="SRPO transformer weights",
            detail="Downloading image refinement weights.",
        )
        state_dict = load_file(srpo_path)
        model.transformer.load_state_dict(state_dict)
        model = model.to(device())
    elif task not in tasks:
        from transformers import AutoModel

        from definers.model_installation import hf_snapshot_download

        model_source = (
            hf_snapshot_download(
                repo_id=task,
                item_label=task,
                detail="Downloading model source files.",
            )
            if is_huggingface_repo(task)
            else task
        )

        model = AutoModel.from_pretrained(
            model_source, torch_dtype=dtype(), trust_remote_code=True
        ).to(device())
    if turbo:
        try:
            model.vae.enable_slicing()
        except:
            pass
        try:
            model.vae.enable_tiling()
        except:
            pass
        optimizations = [
            "enable_vae_slicing",
            "enable_vae_tiling",
            "enable_model_cpu_offload",
            "enable_sequential_cpu_offload",
            "enable_attention_slicing",
        ]
        for opt in optimizations:
            try:
                if opt == "enable_attention_slicing":
                    getattr(model, opt)(1)
                else:
                    getattr(model, opt)()
            except AttributeError:
                pass
            except Exception as e:
                print(f"Could not apply optimization {opt}: {e}")
    MODELS[task] = model
    free()


def is_huggingface_repo(repo_id: str) -> bool:

    if not isinstance(repo_id, str) or not repo_id:
        return False
    repo_id = repo_id.strip()
    if "/" not in repo_id:
        return False
    user, name = repo_id.split("/", 1)
    if not user or not name:
        return False
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    return all(c in allowed_chars for c in user) and all(
        c in allowed_chars for c in name
    )


def normalize_huggingface_repo_id(value: str) -> str:
    if not isinstance(value, str):
        return value

    cleaned_value = value.strip()
    if not cleaned_value.startswith(("http://", "https://")):
        return cleaned_value

    parsed = urlparse(cleaned_value)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return cleaned_value

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return cleaned_value

    if len(parts) >= 3 and parts[2] in {"resolve", "blob", "tree"}:
        return cleaned_value

    candidate = f"{parts[0]}/{parts[1]}"
    if is_huggingface_repo(candidate):
        return candidate

    return cleaned_value


def init_pretrained_model(task: str, turbo: bool = True):
    repo_tasks_override = ["svc", "tts"]

    normalized_task = normalize_huggingface_repo_id(task)

    if normalized_task in MODELS and MODELS[normalized_task]:
        return
    if normalized_task in _FAILED_MODEL_LOADS:
        raise RuntimeError(_FAILED_MODEL_LOADS[normalized_task])

    try:
        if (
            normalized_task in repo_tasks_override
            or (
                normalized_task in tasks
                and is_huggingface_repo(tasks[normalized_task])
            )
            or is_huggingface_repo(normalized_task)
        ):
            from definers.model_installation import (
                install_fast_huggingface_download_hooks,
            )

            install_fast_huggingface_download_hooks()
            return init_model_repo(normalized_task, turbo)
        return init_model_file(normalized_task, turbo)
    except Exception as error:
        message = f"Failed to initialize model '{normalized_task}': {error}"
        _FAILED_MODEL_LOADS[normalized_task] = message
        raise RuntimeError(message) from error


def rvc_to_onnx(model_path):
    from definers.ml.rvc import import_rvc_symbol
    from definers.system import secure_path

    try:
        model_path = secure_path(
            model_path,
            trust=_trusted_paths_from_environment(),
        )
    except Exception as e:
        logger.error(f"Unsafe model path in rvc_to_onnx: {e}")
        return None
    try:
        init_pretrained_model("svc")
    except Exception as e:
        catch(e)
        return None
    try:
        export_onnx = import_rvc_symbol(
            "export_onnx",
            "definers.ml.backends.onnx.export",
            "definers.infer.modules.onnx.export",
            "infer.modules.onnx.export",
        )
    except Exception as e:
        catch(e)
        return None
    if export_onnx is None:
        return None

    try:
        export_onnx(
            model_path,
            model_path.replace(".pth", "") + ".onnx",
        )
        logger.info("ONNX export complete.")
        return model_path.replace(".pth", "") + ".onnx"
    except Exception as e:
        log("An error occurred during ONNX export!", status="")
        catch(e)

    return None


def export_files_rvc(experiment: str):
    logger.info(f"Exporting files for experiment: {experiment}")
    try:
        from definers.system import secure_path

        experiment = secure_path(experiment, basename=True)
    except Exception as e:
        logger.error(f"Invalid experiment name: {e}")
        return []

    rvc_root = _rvc_package_root()
    weight_root = os.path.join(rvc_root, "assets", "weights")
    index_root = os.path.join(rvc_root, "logs")

    exp_path = os.path.join(index_root, experiment)
    latest_checkpoint_filename = find_latest_checkpoint(weight_root, experiment)
    if latest_checkpoint_filename is None:
        error_message = f"Error: No latest checkpoint found for experiment '{experiment}' in '{exp_path}'. Cannot export."
        logger.error(error_message)
        return []
    pth_path = os.path.join(weight_root, latest_checkpoint_filename)
    logger.info(f"Found latest checkpoint: {pth_path}")
    index_file = ""
    exp_path = secure_path(exp_path)
    for root, dirs, files in os.walk(exp_path, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_file = os.path.join(root, name)
                index_file = secure_path(index_file)
                logger.info(f"Found index file: {index_file}")
                break
        if index_file:
            break
    onnx_path = rvc_to_onnx(pth_path)
    if onnx_path is not None:
        try:
            onnx_path = secure_path(
                onnx_path,
                trust=_trusted_paths_from_environment(),
            )
        except Exception as e:
            logger.error(f"Invalid ONNX export path: {e}")
            onnx_path = None
    exported_files = [pth_path]
    if onnx_path is not None and exist(onnx_path):
        exported_files.append(onnx_path)
        logger.info(f"Added ONNX file to exported list: {onnx_path}")
    elif onnx_path is not None:
        logger.warning(f"ONNX file not found after export attempt: {onnx_path}")
    if exist(index_file):
        exported_files.append(index_file)
        logger.info(f"Added index file to exported list: {index_file}")
    else:
        logger.warning(f"Index file not found: {index_file}")
    logger.info(f"Exported files: {exported_files}")
    return exported_files


def _trusted_paths_from_environment() -> list[str] | None:
    configured_paths = os.environ.get("DEFINERS_TRUSTED_PATHS", "")
    trusted_paths = [
        path.strip()
        for path in configured_paths.split(os.pathsep)
        if path.strip()
    ]
    return trusted_paths or None


def find_latest_checkpoint(folder_path: str, model_name: str) -> str | None:
    from definers.system import secure_path

    logger.info(
        f"Searching for latest checkpoint in '{folder_path}' with model name '{model_name}'"
    )
    try:
        folder_path = secure_path(
            folder_path,
            trust=_trusted_paths_from_environment(),
        )
    except Exception as e:
        logger.error(f"Invalid checkpoint folder: {e}")
        return None
    if not is_directory(folder_path):
        logger.error(f"Error: Folder not found at {folder_path}")
        return None
    pattern = re.compile(f"^{re.escape(model_name)}_e(\\d+)_s(\\d+)\\.pth$")
    latest_checkpoint = None
    latest_epoch = -1
    latest_global_step = -1
    try:
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                global_step = int(match.group(2))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_global_step = global_step
                    latest_checkpoint = filename
                elif epoch == latest_epoch and global_step > latest_global_step:
                    latest_global_step = global_step
                    latest_checkpoint = filename
    except Exception as e:
        logger.error(
            f"An error occurred while scanning the folder for checkpoints: {e}"
        )
        return None
    if latest_checkpoint:
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    else:
        logger.warning(
            f"No checkpoint found matching the pattern in '{folder_path}'"
        )
    return latest_checkpoint


def train_model_rvc(
    experiment: str, path: str, lvl: int = 1, f0method: str = "crepe"
):
    from definers.ml.rvc import import_rvc_symbol
    from definers.system import secure_path

    logger.info(f"Starting RVC training for experiment: {experiment}")

    try:
        experiment = secure_path(experiment, basename=True)
    except Exception as e:
        logger.error(f"Invalid experiment name: {e}")
        return None

    try:
        path = secure_path(path, trust=_trusted_paths_from_environment())
    except Exception as e:
        logger.error(f"Invalid audio path for training: {e}")
        return None
    rvc_root = _rvc_package_root()
    index_root = os.path.join(rvc_root, "logs")
    try:
        init_pretrained_model("svc")
        import torch
    except Exception as e:
        logger.error(f"RVC training dependencies unavailable: {e}")
        return None
    Config = import_rvc_symbol(
        "Config",
        "definers.configs.config",
        "configs.config",
    )
    if Config is None:
        return None

    def _run_rvc_command(command):
        with cwd(rvc_root):
            return run(command)

    path = normalize_audio_to_peak(path)
    (path, music) = separate_stems(path)
    path = normalize_audio_to_peak(path)
    config = Config()
    gpus = (
        "-".join([str(i) for i in range(torch.cuda.device_count())])
        if torch.cuda.is_available()
        else ""
    )
    gpu_memories = (
        [
            int(
                torch.cuda.get_device_properties(i).total_memory / 1024**3 + 0.4
            )
            for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [0]
    )
    default_batch_size = (
        math.floor(min(gpu_memories) // 2)
        if gpu_memories and min(gpu_memories) > 0
        else 1
    )
    if default_batch_size == 0:
        default_batch_size = 1

    exp_dir = experiment
    exp_path = os.path.join(index_root, exp_dir)
    exp_path = full_path(secure_path(exp_path))
    logger.info(f"Experiment directory: {exp_path}")

    directory(os.path.join(exp_path, "1_16k_wavs"))
    directory(os.path.join(exp_path, "0_gt_wavs"))
    input_root = os.path.join(exp_path, "input_root")
    directory(input_root)
    input_path = os.path.join(input_root, "input.wav")
    logger.info(f"Moving input audio '{path}' to '{input_path}'")
    try:
        move(path, input_path)
    except Exception as e:
        logger.error(f"Failed to move input audio file: {e}")
        catch(e)
        return None
    filelist_path = os.path.join(exp_path, "filelist.txt")
    logger.info(f"Creating filelist: {filelist_path}")
    try:
        write(filelist_path)
    except Exception as e:
        logger.error(f"Failed to create filelist.txt: {e}")
        catch(e)
        return None
    sr = 96000
    n_p = int(_np.ceil(config.n_cpu / 1.5))
    log_file_preprocess = os.path.join(exp_path, "preprocess.log")
    if_f0 = True
    gpus_rmvpe = f"{gpus}-{gpus}"
    log_file_f0_feature = os.path.join(exp_path, "extract_f0_feature.log")
    logger.info("Starting preprocessing...")
    import shlex

    try:
        with open(log_file_preprocess, "w") as f_preprocess:
            cmd_preprocess = [
                config.python_cmd,
                "-m",
                "infer.modules.train.preprocess",
                input_root,
                str(sr),
                str(n_p),
                exp_path,
            ]
            logger.info("Execute: " + " ".join(cmd_preprocess))
            _run_rvc_command(cmd_preprocess)
        with open(log_file_preprocess) as f_preprocess:
            log_content = f_preprocess.read()
            logger.info("Preprocessing Log:\n" + log_content)
    except Exception as e:
        logger.error(
            f"Preprocessing failed with return code {e.returncode}: {e}"
        )
        try:
            with open(log_file_preprocess, encoding="utf-8") as f:
                error_output = f.read()
            logger.error(f"Preprocessing output:\n{error_output}")
        except Exception as log_e:
            logger.error(f"Could not read preprocess log file: {log_e}")
        catch(e)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        catch(e)
        return None
    logger.info("Preprocessing complete.")
    logger.info("Starting feature extraction...")
    try:
        with open(log_file_f0_feature, "w") as f_f0_feature:
            import shlex

            if if_f0:
                logger.info(f"Extracting F0 using method: {f0method}")
                if torch.cuda.is_available() and f0method == "rmvpe":
                    f0method = "rmvpe_gpu"
                if f0method != "rmvpe_gpu":
                    cmd_f0 = f'"{config.python_cmd}" -m infer.modules.train.extract.extract_f0_print "{exp_path}" {n_p} {f0method}'
                    logger.info("Execute: " + cmd_f0)

                    _run_rvc_command(shlex.split(cmd_f0))
                else:
                    gpus_rmvpe_split = gpus_rmvpe.split("-")
                    leng = len(gpus_rmvpe_split)
                    ps = []
                    logger.info(
                        f"Using {leng} GPUs for RMVPE extraction: {gpus_rmvpe_split}"
                    )
                    for idx, n_g in enumerate(gpus_rmvpe_split):
                        cmd_f0_rmvpe = f'"{config.python_cmd}" -m infer.modules.train.extract.extract_f0_rmvpe {leng} {idx} {n_g} "{exp_path}" {config.is_half}'
                        logger.info(f"Execute (GPU {n_g}): " + cmd_f0_rmvpe)
                        p = thread(
                            _run_rvc_command,
                            shlex.split(cmd_f0_rmvpe),
                        )
                        ps.append(p)
                    wait(*ps)
            logger.info("Extracting features...")
            leng = len(gpus.split("-"))
            ps = []
            logger.info(
                f"Using {leng} GPUs for feature extraction: {gpus.split('-')}"
            )
            for idx, n_g in enumerate(gpus.split("-")):
                cmd_feature_print = f'"{config.python_cmd}" -m infer.modules.train.extract_feature_print {config.device} {leng} {idx} "{exp_path}" v2'
                logger.info(f"Execute (GPU {n_g}): " + cmd_feature_print)
                p = thread(
                    _run_rvc_command,
                    shlex.split(cmd_feature_print),
                )
                ps.append(p)
            wait(*ps)
        with open(log_file_f0_feature) as f_f0_feature:
            log_content = f_f0_feature.read()
            logger.info("F0 and Feature Extraction Log:\n" + log_content)
    except Exception as e:
        catch("An error occurred during F0 or feature extraction")
        catch(e)
        return None
    logger.info("Feature extraction complete.")
    logger.info("Starting index training...")
    feature_dir = os.path.join(exp_path, "3_feature768")
    listdir_res = []
    if exist(feature_dir):
        listdir_res = os.listdir(feature_dir)
    if not exist(feature_dir) or not any(listdir_res):
        error_message = f"Error: Feature directory '{feature_dir}' is missing or empty! Cannot train index."
        catch(error_message)
        return None
    try:
        npys = []
        for name in sorted(listdir_res):
            if name.endswith(".npy"):
                phone = _np.load(os.path.join(feature_dir, name))
                npys.append(phone)
        if not npys:
            error_message = f"Error: No .npy files found in '{feature_dir}'! Cannot train index."
            logger.error(error_message)
            return None
        big_npy = _np.concatenate(npys, 0)
        logger.info(f"Concatenated features shape: {big_npy.shape}")
        big_npy_idx = _np.arange(big_npy.shape[0])
        _np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]
    except Exception as e:
        catch(
            "An error occurred while loading and concatenating features for index training"
        )
        catch(e)
        return None
    try:
        from sklearn.cluster import MiniBatchKMeans

        big_npy = (
            MiniBatchKMeans(
                n_clusters=100,
                verbose=False,
                batch_size=256 * config.n_cpu,
                compute_labels=False,
                init="random",
                n_init=3,
            )
            .fit(big_npy)
            .cluster_centers_
        )
        logger.info(f"KMeans cluster centers shape: {big_npy.shape}")
        import faiss

        feature_dimension = big_npy.shape[1]
        n_ivf = min(
            int(16 * _np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39
        )
        n_ivf = max(1, n_ivf)
        logger.info(
            f"Training Faiss index with dimension {feature_dimension} and n_ivf {n_ivf}"
        )
        index = faiss.index_factory(feature_dimension, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        logger.info(f"Faiss index nprobe set to: {index_ivf.nprobe}")
        logger.info("Training Faiss index...")
        index.train(big_npy)
        logger.info("Faiss index training complete.")
        trained_index_path = os.path.join(
            exp_path,
            f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_v2.index",
        )
        faiss.write_index(index, trained_index_path)
        logger.info(f"Trained Faiss index saved to: {trained_index_path}")
        logger.info("Adding features to Faiss index...")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])
        logger.info("Features added to Faiss index.")
        added_index_filename = f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_v2.index"
        added_index_path = os.path.join(exp_path, added_index_filename)
        faiss.write_index(index, added_index_path)
        logger.info(f"Final Faiss index saved to: {added_index_path}")
        target_link_path = os.path.join(index_root, added_index_filename)
        logger.info(
            f"Creating link from '{added_index_path}' to '{target_link_path}'"
        )
        try:
            if exist(target_link_path) or os.path.islink(target_link_path):
                os.remove(target_link_path)
                logger.warning(
                    f"Removed existing file/link at {target_link_path}"
                )
            if get_os_name() != "windows":
                os.symlink(added_index_path, target_link_path)
            else:
                shutil.copy(added_index_path, target_link_path)
            logger.info("Index linking successful.")
        except Exception as e:
            logger.error(f"Linking index failed: {e}")
            catch(e)
    except Exception as e:
        logger.error(f"An error occurred during index training: {e}")
        catch(e)
        return None
    logger.info("Index training complete.")
    logger.info("Starting model training...")
    try:
        pretrained_G = "assets/pretrained_v2/f0G48k.pth"
        pretrained_D = "assets/pretrained_v2/f0D48k.pth"
        batch_size = default_batch_size
        total_epoch = 250 * lvl
        save_epoch = 250
        if_save_latest = 1
        if_cache_gpu = 1
        if_save_every_weights = 1
        gpus_str = gpus
        config_path = "v2/48k.json"
        config_save_path = os.path.join(exp_path, "config.json")
        if not exist(config_save_path):
            logger.info(f"Saving training config to: {config_save_path}")
            try:
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(
                        config.json_config.get(config_path, {}),
                        f,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    )
                    f.write("\n")
            except Exception as e:
                logger.error(f"Failed to save training config file: {e}")
                catch(e)
        log_file_train = os.path.join(exp_path, "train.log")
        logger.info("Executing training command...")
        with open(log_file_train, "w") as f_train:
            cmd_train = [
                config.python_cmd,
                "-m",
                "infer.modules.train.train",
                "-e",
                exp_dir,
                "-sr",
                "96k",
                "-f0",
                "1",
                "-bs",
                str(batch_size),
                "-g",
                gpus_str,
                "-te",
                str(total_epoch),
                "-se",
                str(save_epoch),
                "-pg",
                pretrained_G,
                "-pd",
                pretrained_D,
                "-l",
                str(if_save_latest),
                "-c",
                str(if_cache_gpu),
                "-sw",
                str(if_save_every_weights),
                "-v",
                "v2",
            ]
            logger.info("Execute: " + " ".join(cmd_train))
            _run_rvc_command(cmd_train)
        with open(log_file_train) as f_train:
            log_content = f_train.read()
            logger.info("Training Log:\n" + log_content)
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Model training failed with return code {e.returncode}: {e}"
        )
        logger.error(
            f"Training output:\n{(e.stdout.decode() if e.stdout else 'N/A')}\n{(e.stderr.decode() if e.stderr else 'N/A')}"
        )
        catch(e)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        catch(e)
        return None
    logger.info("Model training complete.")
    logger.info("Training complete, exporting files...")
    exp_files = export_files_rvc(exp_dir)
    tmps = []
    for exp_file in exp_files:
        ext = path_ext(exp_file)
        tmps.append(tmp(ext))
        copy(exp_file, tmps[-1])
    return tmps


def convert_vocal_rvc(experiment: str, path: str):
    from definers.ml.rvc import import_rvc_symbol
    from definers.system import secure_path

    logger.info(f"Starting vocal conversion for experiment: {experiment}")
    try:
        experiment = secure_path(experiment, basename=True)
    except Exception as e:
        logger.error(f"Invalid experiment name: {e}")
        return None

    try:
        path = secure_path(path, trust=_trusted_paths_from_environment())
    except Exception as e:
        logger.error(f"Invalid audio path for conversion: {e}")
        return None
    rvc_root = _rvc_package_root()
    weight_root = os.path.join(rvc_root, "assets", "weights")
    index_root = os.path.join(rvc_root, "logs")

    try:
        init_pretrained_model("svc")
    except Exception as e:
        logger.error(f"Vocal conversion feature unavailable: {e}")
        return None

    Config = import_rvc_symbol(
        "Config",
        "definers.configs.config",
        "configs.config",
    )
    VC = import_rvc_symbol(
        "VC",
        "definers.ml.backends.vc.modules",
        "definers.infer.modules.vc.modules",
        "infer.modules.vc.modules",
    )
    if Config is None or VC is None:
        return None

    path = normalize_audio_to_peak(path)
    (path, music) = separate_stems(path)
    config = Config()
    vc = VC(config)
    exp_path = os.path.join(index_root, experiment)
    latest_checkpoint_filename = find_latest_checkpoint(weight_root, experiment)
    if latest_checkpoint_filename is None:
        error_message = f"Error: No latest checkpoint found for experiment '{experiment}' in '{exp_path}'. Cannot perform conversion."
        logger.error(error_message)
        return None
    pth_path = os.path.join(weight_root, latest_checkpoint_filename)
    logger.info(f"Using model checkpoint: {pth_path}")
    idx_path = None
    for root, dirs, files in os.walk(exp_path, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                idx_path = os.path.join(root, name)
                logger.info(f"Found index file: {idx_path}")
                break
        if idx_path:
            break
    if idx_path is None:
        logger.warning(
            f"No index file found for experiment '{experiment}' in '{exp_path}'. Conversion may be less effective."
        )
    filter_radius = 7
    semitones = 0
    index_rate = 1.0
    protect = 0.5
    f0_mean_pooling = 0
    rms_mix_rate = 0.0
    try:
        with cwd(rvc_root):
            vc.get_vc(
                latest_checkpoint_filename,
                index_rate,
                f0_mean_pooling,
            )
        logger.info("VC model loaded.")
    except Exception as e:
        logger.error(f"Failed to load VC model: {e}")
        catch(e)
        return None
    try:
        with cwd(rvc_root):
            (message, (sr, aud)) = vc.vc_single(
                sid=0,
                input_audio_path=path,
                f0_up_key=semitones,
                f0_file=None,
                f0_method="harvest",
                file_index=idx_path,
                file_index2=None,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=0,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )
        logger.info(f"Vocal conversion message: {message}")
    except Exception as e:
        logger.error(f"An error occurred during vocal conversion: {e}")
        catch(e)
        return None
    if aud is not None and isinstance(aud, _np.ndarray) and (aud.size > 0):
        logger.info(
            f"Conversion successful, saving output audio with shape {aud.shape} at sample rate {sr}"
        )
        try:
            out_voice = tmp("wav")
            import pydub
            import soundfile as sf

            sf.write(out_voice, aud, sr)
            out_path = stem_mixer([out_voice, music], "mp3")
            logger.info(f"Output audio saved to: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to save output audio file: {e}")
            catch(e)
            return None
    else:
        logger.warning("Vocal conversion did not produce valid audio data.")
        return None


class AutoTrainer:
    def __init__(
        self,
        source=None,
        target=None,
        model=None,
        model_type: str | None = None,
        model_path: str | None = None,
        task: str | None = None,
        *,
        batch_size: int = 32,
        source_type: str = "parquet",
        revision: str | None = None,
        validation_split: float = 0.0,
        test_split: float = 0.0,
        auto_tune: bool = True,
        early_stopping: bool | None = None,
        patience: int | None = None,
        cv_folds: int = 0,
    ):
        self.source = source
        self.target = target
        self.model = model
        self.model_type = model_type
        self.model_path = model_path
        self.task = task
        self.batch_size = batch_size
        self.source_type = source_type
        self.revision = revision
        self.validation_split = validation_split
        self.test_split = test_split
        self.auto_tune = bool(auto_tune)
        self.early_stopping = early_stopping
        self.patience = patience
        self.cv_folds = max(0, int(cv_folds or 0))
        self.vectorizer = None
        self.label_mapping = None
        self.last_training_plan = None
        self.last_auto_tune_notes: tuple[str, ...] = ()
        self.cv_scores: list[float] = []

    def use(self, source=None, target=None, *, task: str | None = None):
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target
        if task is not None:
            self.task = task
        return self

    def _coerce_reference(self, value):
        if hasattr(value, "name") and not isinstance(value, (str, bytes)):
            return getattr(value, "name")
        return value

    def _safe_local_path(self, value):
        value = self._coerce_reference(value)
        if value is None:
            return None
        str_value = str(value).strip()
        if not str_value:
            return None
        from definers.system import secure_path

        try:
            return secure_path(str_value)
        except Exception:
            return None

    def _has_known_path_suffix(self, text: str) -> bool:
        known_suffixes = (
            ".csv",
            ".json",
            ".xlsx",
            ".txt",
            ".wav",
            ".mp3",
            ".flac",
            ".png",
            ".jpg",
            ".jpeg",
            ".mp4",
            ".mkv",
            ".joblib",
            ".pth",
            ".pkl",
        )
        return text.lower().endswith(known_suffixes)

    def _looks_like_path(self, value) -> bool:
        value = self._coerce_reference(value)
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        safe_path = self._safe_local_path(text)
        if safe_path is None:
            return False
        if os.path.exists(safe_path):
            return True
        return self._has_known_path_suffix(text)

    def _looks_like_remote_source(self, value) -> bool:
        value = self._coerce_reference(value)
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        safe_path = self._safe_local_path(text)
        if safe_path is not None and os.path.exists(safe_path):
            return False
        if text.startswith(("http://", "https://")):
            return True
        normalized_text = normalize_huggingface_repo_id(text)
        return is_huggingface_repo(normalized_text)

    def _looks_like_path_collection(self, value) -> bool:
        value = self._coerce_reference(value)
        if isinstance(value, (list, tuple)) and value:
            return all(self._looks_like_path(item) for item in value)
        return self._looks_like_path(value)

    def _coerce_path_collection(self, value):
        value = self._coerce_reference(value)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [str(self._coerce_reference(item)) for item in value]
        return [str(value)]

    def _normalize_text_list(self, value):
        value = self._coerce_reference(value)
        if value is None:
            return None
        if isinstance(value, str):
            normalized_value = validate_str_param("value", value)
            parts = [
                part.strip()
                for part in normalized_value.replace(",", ";").split(";")
            ]
            return [part for part in parts if part]
        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                item_value = self._coerce_reference(item)
                if item_value is None:
                    continue
                item_text = str(item_value).strip()
                if item_text:
                    parts.append(item_text)
            return parts or None
        text = str(value).strip()
        return [text] if text else None

    def _normalize_optional_text(self, name: str, value):
        value = self._coerce_reference(value)
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return validate_str_param(name, text)

    def _normalize_selected_rows(self, value):
        value = self._coerce_reference(value)
        if not check_parameter(value):
            return None
        return simple_text(validate_str_param("selected_rows", str(value)))

    def _estimate_sample_count(self, source, target) -> int | None:
        for candidate in (source, target):
            try:
                if candidate is None:
                    continue
                if isinstance(candidate, dict):
                    for key in ("features", "X", "data"):
                        if key in candidate:
                            return len(candidate[key])
                    continue
                if hasattr(candidate, "shape") and getattr(
                    candidate, "shape", None
                ):
                    return int(candidate.shape[0])
                if isinstance(candidate, (list, tuple)) and candidate:
                    if all(self._looks_like_path(item) for item in candidate):
                        continue
                    return len(candidate)
            except Exception:
                continue
        return None

    def _resolve_auto_tuned_settings(
        self,
        *,
        source,
        target,
        batch_size,
        validation_split,
        test_split,
        early_stopping,
        patience,
        cv_folds,
    ):
        from definers.ml.trainer_plan import suggest_training_defaults

        if not self.auto_tune:
            return {
                "batch_size": batch_size,
                "validation_split": validation_split,
                "test_split": test_split,
                "early_stopping": bool(early_stopping)
                if early_stopping is not None
                else False,
                "patience": int(patience) if patience else 0,
                "cv_folds": max(0, int(cv_folds or 0)),
                "notes": (),
            }
        provided_batch = (
            None if batch_size is None or batch_size == 32 else batch_size
        )
        provided_val = (
            None
            if validation_split is None or validation_split == 0.0
            else validation_split
        )
        provided_test = None if test_split is None else test_split
        provided_early = early_stopping
        n_samples = self._estimate_sample_count(source, target)
        defaults = suggest_training_defaults(
            n_samples,
            batch_size=provided_batch,
            validation_split=provided_val,
            test_split=provided_test,
            early_stopping=provided_early,
            cv_folds=cv_folds,
        )
        return {
            "batch_size": defaults.batch_size,
            "validation_split": defaults.validation_split,
            "test_split": defaults.test_split,
            "early_stopping": defaults.early_stopping,
            "patience": int(patience)
            if patience is not None
            else defaults.patience,
            "cv_folds": defaults.cv_folds,
            "notes": defaults.notes,
        }

    def training_plan(
        self,
        data=None,
        target=None,
        *,
        resume_from: str | None = None,
        revision: str | None = None,
        source_type: str | None = None,
        label_columns=None,
        drop=None,
        select=None,
        order_by=None,
        stratify=None,
        validation_split: float | None = None,
        test_split: float | None = None,
        batch_size: int | None = None,
        auto_tune: bool | None = None,
        early_stopping: bool | None = None,
        patience: int | None = None,
        cv_folds: int | None = None,
    ):
        from definers.ml.trainer_plan import build_training_plan

        source, target_value = self._resolve_training_source(data, target)
        active_revision = self.revision if revision is None else revision
        active_source_type = (
            self.source_type if source_type is None else source_type
        )
        active_validation_split = (
            self.validation_split
            if validation_split is None
            else validation_split
        )
        active_test_split = (
            self.test_split if test_split is None else test_split
        )
        active_batch_size = (
            self.batch_size if batch_size is None else batch_size
        )
        active_early = (
            self.early_stopping if early_stopping is None else early_stopping
        )
        active_patience = self.patience if patience is None else patience
        active_cv = self.cv_folds if cv_folds is None else cv_folds

        previous_auto_tune = self.auto_tune
        if auto_tune is not None:
            self.auto_tune = bool(auto_tune)
        try:
            tuned = self._resolve_auto_tuned_settings(
                source=source,
                target=target_value,
                batch_size=active_batch_size,
                validation_split=active_validation_split,
                test_split=active_test_split,
                early_stopping=active_early,
                patience=active_patience,
                cv_folds=active_cv,
            )
        finally:
            self.auto_tune = previous_auto_tune

        normalized_select = self._normalize_selected_rows(select)
        normalized_drop = self._normalize_text_list(drop)
        normalized_label_columns = self._normalize_text_list(label_columns)
        plan = build_training_plan(
            source=source,
            target=target_value,
            batch_size=tuned["batch_size"],
            source_type=active_source_type,
            revision=active_revision,
            validation_split=tuned["validation_split"],
            test_split=tuned["test_split"],
            label_columns=normalized_label_columns,
            drop_columns=normalized_drop,
            order_by=self._normalize_optional_text("order_by", order_by),
            stratify=self._normalize_optional_text("stratify", stratify),
            selected_rows=normalized_select,
            resume_from=self._coerce_reference(resume_from),
            is_remote_dataset=self._is_remote_dataset(source),
            is_file_dataset=self._is_file_dataset(source),
            early_stopping=tuned["early_stopping"],
            patience=tuned["patience"],
            cv_folds=tuned["cv_folds"],
            auto_tuned=tuned["notes"],
        )
        self.last_training_plan = plan
        self.last_auto_tune_notes = tuned["notes"]
        return plan

    def _resolve_training_source(self, data=None, target=None):
        active_source = self.source if data is None else data
        active_target = self.target if target is None else target
        active_source = self._coerce_reference(active_source)
        active_target = self._coerce_reference(active_target)
        if isinstance(active_source, dict):
            features = active_source.get(
                "features", active_source.get("X", active_source.get("data"))
            )
            labels = active_source.get(
                "labels", active_source.get("y", active_target)
            )
            return features, labels
        if (
            active_target is None
            and isinstance(active_source, tuple)
            and len(active_source) == 2
        ):
            return active_source[0], active_source[1]
        return active_source, active_target

    def _is_file_target(self, value) -> bool:
        return self._looks_like_path_collection(value)

    def _is_remote_dataset(self, value) -> bool:
        value = self._coerce_reference(value)
        if isinstance(value, (list, tuple)):
            return False
        return self._looks_like_remote_source(value)

    def _is_file_dataset(self, value) -> bool:
        value = self._coerce_reference(value)
        if isinstance(value, (list, tuple)):
            return bool(value) and all(
                self._looks_like_path(item) for item in value
            )
        return self._looks_like_path(value)

    def _extract_features_and_labels(self, data, labels=None):
        if isinstance(data, dict):
            features = data.get("features", data.get("X", data.get("data")))
            extracted_labels = data.get("labels", data.get("y", labels))
            return features, extracted_labels
        if labels is None and isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        return data, labels

    def _encode_text_features(self, rows):
        normalized_rows = [str(row) for row in rows]
        if create_vectorizer is None:
            return np.asarray(normalized_rows, dtype=object).reshape(-1, 1)
        if self.vectorizer is None:
            self.vectorizer = create_vectorizer(normalized_rows)
        matrix = self.vectorizer.transform(normalized_rows)
        return np.asarray(matrix.toarray())

    def _coerce_feature_data(self, data):
        if data is None:
            return None
        if isinstance(data, str):
            return self._encode_text_features([data])
        if (
            isinstance(data, (list, tuple))
            and data
            and all(isinstance(item, str) for item in data)
        ):
            return self._encode_text_features(data)
        array = np.asarray(data)
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array

    def _coerce_label_data(self, labels):
        if labels is None:
            return None
        if isinstance(labels, str):
            labels = [labels]
        if (
            isinstance(labels, (list, tuple))
            and labels
            and all(isinstance(item, str) for item in labels)
        ):
            ordered_labels = [str(item) for item in labels]
            unique_labels = list(dict.fromkeys(ordered_labels))
            self.label_mapping = {
                label: index for index, label in enumerate(unique_labels)
            }
            return np.asarray(
                [self.label_mapping[label] for label in ordered_labels]
            )
        array = np.asarray(labels)
        if array.ndim > 1 and array.shape[-1] == 1:
            return array.reshape(-1)
        return array

    def load(self, model_path: str | None = None):
        from definers.ml.safe_deserialization import (
            load_serialized_model,
        )
        from definers.system import secure_path

        resolved_model_path = self._coerce_reference(
            model_path or self.model_path
        )
        if not resolved_model_path:
            return None
        try:
            safe_model_path = secure_path(resolved_model_path)
        except Exception as error:
            catch(error)
            return None
        model_type = self.model_type or path_ext(safe_model_path)
        if model_type not in (
            "joblib",
            "pkl",
            "pt",
            "pth",
            "safetensors",
            "onnx",
        ):
            model_type = "joblib"
        self.model = repository_sync.load_model(safe_model_path, model_type)
        self.model_path = str(resolved_model_path)
        return self.model

    def save(self, model_path: str | None = None):
        import joblib

        resolved_model_path = self._coerce_reference(
            model_path or self.model_path or f"model_{random_string()}.joblib"
        )
        joblib.dump(self.model, resolved_model_path)
        self.model_path = resolved_model_path
        return resolved_model_path

    def feed(self, data=None, target=None, epochs: int = 1):
        feature_data, label_data = self._extract_features_and_labels(
            self.source if data is None else data,
            self.target if target is None else target,
        )
        feature_array = self._coerce_feature_data(feature_data)
        label_array = self._coerce_label_data(label_data)
        self.model = feed(
            self.model,
            feature_array,
            label_array,
            epochs=epochs,
            logger=log,
            concatenate=_concatenate_training_rows(),
        )
        return self.model

    def fit(self, data=None, target=None, epochs: int = 1, model=None):
        if model is not None:
            self.model = model
        if data is not None or target is not None or self.source is not None:
            self.feed(data, target, epochs=epochs)
        if self.model is None:
            self.model = HybridModel()
        self.model = fit(
            self.model,
            array_adapter=_training_array_adapter(),
            logger=log,
            error_handler=catch,
        )
        return self.model

    def _dataset_loaders(
        self,
        source,
        target,
        *,
        source_type: str,
        revision: str | None,
        label_columns,
        drop,
        select,
        order_by,
        stratify,
        validation_split: float,
        test_split: float,
        batch_size: int,
    ):
        from definers.data.loaders import (
            drop_columns,
            fetch_dataset,
            files_to_dataset,
            select_rows,
        )
        from definers.data.preparation import (
            prepare_data,
            to_loader,
        )

        loaders = []
        if check_parameter(select):
            if self._is_remote_dataset(source):
                dataset = fetch_dataset(source, source_type, revision)
            else:
                dataset = files_to_dataset(
                    self._coerce_path_collection(source),
                    self._coerce_path_collection(target)
                    if self._is_file_target(target)
                    else target,
                )
            dataset = drop_columns(dataset, drop)
            log("Full dataset length", len(dataset))
            for part in select.split():
                if "-" in part:
                    start_end = part.split("-")
                    loaders.append(
                        to_loader(
                            select_rows(
                                dataset,
                                int(start_end[0]) - 1,
                                int(start_end[-1]),
                            )
                        )
                    )
                else:
                    loaders.append(
                        to_loader(
                            select_rows(dataset, int(part) - 1, int(part))
                        )
                    )
            return loaders

        prepared_data = prepare_data(
            remote_src=source if self._is_remote_dataset(source) else None,
            features=self._coerce_path_collection(source)
            if self._is_file_dataset(source)
            else source,
            labels=self._coerce_path_collection(target)
            if self._is_file_target(target)
            else target,
            url_type=source_type,
            revision=revision,
            drop=drop,
            order_by=order_by,
            stratify=stratify,
            val_frac=validation_split,
            test_frac=test_split,
            batch_size=batch_size,
        )
        if prepared_data is None:
            return []
        loaders.append(prepared_data.train)
        if prepared_data.val is not None:
            loaders.append(prepared_data.val)
        if prepared_data.test is not None:
            loaders.append(prepared_data.test)
        try:
            dataset_len = len(prepared_data.train.dataset)
        except Exception:
            dataset_len = None
        if dataset_len is not None:
            log("Full dataset length", dataset_len)
        return loaders

    def _train_batches(self, loaders, label_columns):
        from definers.data.arrays import numpy_to_cupy
        from definers.data.loaders import split_columns
        from definers.data.preparation import pad_sequences
        from definers.data.tokenization import (
            init_tokenizer,
            tokenize_and_pad,
        )

        tokenizer = init_tokenizer()
        is_supervised = check_parameter(label_columns)
        for loader_index, loader in enumerate(loaders):
            logger.info(f"Loader {loader_index + 1}")
            for batch_index, batch in enumerate(loader):
                logger.info(f"Batch {batch_index + 1}: {batch}")
                if is_supervised:
                    features_batch, labels_batch = split_columns(
                        batch,
                        label_columns,
                        is_batch=True,
                    )
                    features_batch = pad_sequences(
                        tokenize_and_pad(features_batch, tokenizer)
                    )
                    labels_batch = tokenize_and_pad(labels_batch, tokenizer)
                    self.model = feed(
                        self.model,
                        numpy_to_cupy(features_batch),
                        numpy_to_cupy(labels_batch),
                        logger=log,
                        concatenate=_concatenate_training_rows(),
                    )
                    continue
                features_batch = pad_sequences(
                    tokenize_and_pad(batch, tokenizer)
                )
                self.model = feed(
                    self.model,
                    numpy_to_cupy(features_batch),
                    logger=log,
                    concatenate=_concatenate_training_rows(),
                )
        return self.model

    def train_url(
        self,
        url: str,
        target=None,
        *,
        save_as: str | None = None,
        revision: str | None = None,
        source_type: str | None = None,
        **kwargs,
    ):
        self.use(source=url, target=target)
        return self.train(
            save_as=save_as,
            revision=revision,
            source_type=source_type,
            **kwargs,
        )

    def train_files(
        self, files, target=None, *, save_as: str | None = None, **kwargs
    ):
        self.use(source=files, target=target)
        return self.train(save_as=save_as, **kwargs)

    def _resolve_inference_model_type(
        self, model_source, model, model_type: str | None
    ):
        if model_type:
            return str(model_type).lower()
        try:
            return get_ext(model_source).lower()
        except Exception:
            pass
        if hasattr(model, "predict"):
            return "joblib"
        if hasattr(model, "get_inputs") and callable(model.get_inputs):
            return "onnx"
        return "pt"

    def train(
        self,
        data=None,
        target=None,
        *,
        save_as: str | None = None,
        resume_from: str | None = None,
        revision: str | None = None,
        source_type: str | None = None,
        label_columns=None,
        drop=None,
        select=None,
        order_by=None,
        stratify=None,
        validation_split: float | None = None,
        test_split: float | None = None,
        batch_size: int | None = None,
        auto_tune: bool | None = None,
        early_stopping: bool | None = None,
        patience: int | None = None,
        cv_folds: int | None = None,
    ):
        source, target_value = self._resolve_training_source(data, target)
        active_revision = self.revision if revision is None else revision
        active_source_type = (
            self.source_type if source_type is None else source_type
        )
        active_validation_split = (
            self.validation_split
            if validation_split is None
            else validation_split
        )
        active_test_split = (
            self.test_split if test_split is None else test_split
        )
        active_batch_size = (
            self.batch_size if batch_size is None else batch_size
        )
        active_early = (
            self.early_stopping if early_stopping is None else early_stopping
        )
        active_patience = self.patience if patience is None else patience
        active_cv = self.cv_folds if cv_folds is None else cv_folds

        previous_auto_tune = self.auto_tune
        if auto_tune is not None:
            self.auto_tune = bool(auto_tune)
        try:
            tuned = self._resolve_auto_tuned_settings(
                source=source,
                target=target_value,
                batch_size=active_batch_size,
                validation_split=active_validation_split,
                test_split=active_test_split,
                early_stopping=active_early,
                patience=active_patience,
                cv_folds=active_cv,
            )
        finally:
            self.auto_tune = previous_auto_tune

        active_batch_size = tuned["batch_size"]
        active_validation_split = tuned["validation_split"]
        active_test_split = tuned["test_split"]
        self.last_auto_tune_notes = tuned["notes"]
        if tuned["notes"]:
            for note in tuned["notes"]:
                log("Auto-tune", note)

        normalized_select = self._normalize_selected_rows(select)
        normalized_drop = self._normalize_text_list(drop)
        normalized_label_columns = self._normalize_text_list(label_columns)

        if self._is_remote_dataset(source):
            validate_str_param("remote_src", str(source))

        if resume_from is not None:
            self.model_path = self._coerce_reference(resume_from)
            self.load(self.model_path)

        if self._is_remote_dataset(source) or self._is_file_dataset(source):
            resolved_label_columns = normalized_label_columns
            if resolved_label_columns is None and not self._is_file_target(
                target_value
            ):
                resolved_label_columns = self._normalize_text_list(target_value)
            loaders = self._dataset_loaders(
                source,
                target_value,
                source_type=active_source_type,
                revision=active_revision,
                label_columns=resolved_label_columns,
                drop=normalized_drop,
                select=normalized_select,
                order_by=order_by,
                stratify=stratify,
                validation_split=active_validation_split,
                test_split=active_test_split,
                batch_size=active_batch_size,
            )
            if not loaders:
                return None
            self._train_batches(loaders, resolved_label_columns)
            self.fit()
            return self.save(save_as)

        cv_folds = max(0, int(tuned.get("cv_folds", 0) or 0))
        if cv_folds >= 2:
            self._run_cross_validation(source, target_value, cv_folds)
        self.fit(source, target_value)
        return self.save(save_as)

    def _run_cross_validation(self, source, target_value, cv_folds: int):
        try:
            feature_data, label_data = self._extract_features_and_labels(
                source, target_value
            )
            features = self._coerce_feature_data(feature_data)
            labels = self._coerce_label_data(label_data)
            if features is None or labels is None:
                return
            n = int(getattr(features, "shape", [len(features)])[0])
            folds = max(2, min(cv_folds, n))
            scores: list[float] = []
            fold_size = n // folds
            for fold_index in range(folds):
                start = fold_index * fold_size
                end = n if fold_index == folds - 1 else start + fold_size
                val_X = features[start:end]
                val_y = labels[start:end]
                indices = list(range(0, start)) + list(range(end, n))
                if not indices:
                    continue
                train_X = (
                    features[indices]
                    if hasattr(features, "__getitem__")
                    else features
                )
                train_y = (
                    labels[indices]
                    if hasattr(labels, "__getitem__")
                    else labels
                )
                fold_model = HybridModel()
                try:
                    fold_model.fit(train_X, train_y)
                    if hasattr(fold_model, "score"):
                        scores.append(float(fold_model.score(val_X, val_y)))
                    elif hasattr(fold_model, "predict"):
                        preds = fold_model.predict(val_X)
                        try:
                            correct = float(
                                (np.asarray(preds) == np.asarray(val_y)).mean()
                            )
                            scores.append(correct)
                        except Exception:
                            pass
                except Exception as fold_error:
                    catch(fold_error)
            self.cv_scores = scores
            if scores:
                avg = sum(scores) / len(scores)
                log(
                    "Cross-validation mean score",
                    f"{avg:.4f} across {len(scores)} folds",
                )
        except Exception as error:
            catch(error)

    def _predict_from_file(
        self, prediction_file: str, model_path: str | None = None
    ):
        from definers.ml.safe_deserialization import (
            load_serialized_model,
        )
        from definers.system import secure_path

        resolved_prediction_file = str(self._coerce_reference(prediction_file))
        resolved_model_path = self._coerce_reference(
            model_path or self.model_path
        )
        if not resolved_model_path:
            return None
        try:
            safe_model_path = secure_path(resolved_model_path)
        except Exception as error:
            catch(error)
            return None
        model = load_serialized_model(safe_model_path, "joblib")
        if model is None:
            return None
        ext = os.path.splitext(resolved_prediction_file)[1].lstrip(".").lower()
        if ext in common_audio_formats:
            return predict_audio(model, resolved_prediction_file)
        if ext == "txt":
            text_data = read(resolved_prediction_file)
            vectorizer = create_vectorizer([text_data])
            features = extract_text_features(text_data, vectorizer)
        else:
            features = load_as_numpy(resolved_prediction_file)
            if features is None:
                return None
        prediction = model.predict(one_dim_numpy(numpy_to_cupy(features)))
        if prediction is None:
            return None
        if is_clusters_model(model):
            prediction = get_cluster_content(model, int(prediction[0]))
        output_type = guess_numpy_type(prediction)
        if output_type == "text":
            from definers.system.output_paths import managed_output_path

            text_output = features_to_text(prediction)
            path = managed_output_path(
                "txt",
                section="train",
                stem="prediction",
            )
            with open(path, "w", encoding="utf-8") as file_obj:
                file_obj.write(text_output)
            return path
        if output_type == "image":
            from definers.system.output_paths import managed_output_path

            image_output = features_to_image(prediction)
            path = managed_output_path(
                "png",
                section="train",
                stem="prediction",
            )
            import imageio.v3 as iio

            iio.imwrite(path, cupy_to_numpy(image_output))
            return path
        return None

    def predict(self, data, model_path: str | None = None):
        data = self._coerce_reference(data)
        if self._looks_like_path_collection(data):
            if isinstance(data, (list, tuple)):
                data = data[0]
            return self._predict_from_file(str(data), model_path=model_path)

        model = self.model or self.load(model_path)
        if model is None or not hasattr(model, "predict"):
            return None
        feature_data, _ = self._extract_features_and_labels(data)
        feature_array = self._coerce_feature_data(feature_data)
        if feature_array is None:
            return None
        prediction_input = np.asarray(feature_array)
        if prediction_input.ndim == 1:
            prediction_input = prediction_input.reshape(1, -1)
        prediction = None
        try:
            prediction = model.predict(prediction_input)
        except Exception:
            try:
                prediction = model.forward(prediction_input)
            except Exception:
                prediction = model.__call__(prediction_input)
        if cupy_to_numpy is not None:
            try:
                return cupy_to_numpy(prediction)
            except Exception:
                return prediction
        return prediction

    def infer(
        self, data, task: str | None = None, model_type: str | None = None
    ):
        resolved_task = self._coerce_reference(
            task or self.task or self.model_path
        )
        resolved_data = self._coerce_reference(data)
        if resolved_task is None:
            return self.predict(resolved_data)
        if isinstance(resolved_data, (list, tuple)):
            resolved_data = self._coerce_reference(resolved_data[0])
        if resolved_data is None:
            return None

        vec = None
        input_data = None
        model_key = str(resolved_task)
        model_source = tasks.get(model_key, model_key)

        if not (model_key in MODELS and MODELS[model_key]):
            init_model_file(model_key)
        model = MODELS[model_key]
        if model is None:
            return None
        active_model_type = self._resolve_inference_model_type(
            model_source,
            model,
            model_type,
        )

        file_ext = get_ext(str(resolved_data))
        if file_ext == "txt":
            text_data = read(str(resolved_data))
            if isinstance(text_data, (tuple, list)):
                text_data = "".join(text_data)
            vec = create_vectorizer([text_data])
            input_data = numpy_to_cupy(extract_text_features(text_data, vec))
        elif file_ext in common_audio_formats:
            return predict_audio(model, str(resolved_data))
        else:
            input_data = numpy_to_cupy(load_as_numpy(str(resolved_data)))
        if input_data is None:
            log("Could not load input data", resolved_data, status=False)
            return None

        input_numpy = cupy_to_numpy(one_dim_numpy(input_data))
        try:
            if active_model_type in ["joblib", "pkl"]:
                prediction = model.predict(input_numpy)
            elif active_model_type in ["pt", "pth", "safetensors"]:
                import torch

                input_tensor = torch.from_numpy(input_numpy).to(device())
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                prediction = output_tensor.cpu().numpy()
            elif active_model_type == "onnx":
                input_name = model.get_inputs()[0].name
                prediction = model.run(
                    None,
                    {input_name: input_numpy.astype(np.float32)},
                )[0]
            else:
                return None
        except Exception as error:
            logging.error(
                f"Model prediction failed for type '{active_model_type}': {error}"
            )
            return None

        if prediction is None:
            logging.error("Model prediction returned None.")
            return None
        if is_clusters_model(model):
            prediction = one_dim_numpy(
                get_cluster_content(model, int(prediction[0]))
            )
        prediction_type = guess_numpy_type(prediction)
        if vec is not None:
            prediction = features_to_text(prediction)
            prediction_type = "text"
        elif prediction_type == "text":
            prediction = features_to_text(prediction)
        elif prediction_type == "audio":
            prediction = features_to_audio(prediction)
        elif prediction_type == "image":
            prediction = features_to_image(prediction)
        elif prediction_type == "video":
            prediction = features_to_video(prediction)
        output_filename = f"{random_string()}.{get_prediction_file_extension(prediction_type)}"
        handlers = {
            "video": lambda: write_video(prediction, 24),
            "image": lambda: __import__("imageio").imwrite(
                output_filename,
                (prediction * 255).astype(np.uint8),
            ),
            "audio": lambda: __import__(
                "scipy.io", fromlist=["wavfile"]
            ).wavfile.write(output_filename, 32000, prediction),
            "text": lambda: open(output_filename, "w", encoding="utf-8").write(
                prediction
            ),
        }
        handler = handlers.get(prediction_type)
        if handler is None:
            logging.error(f"Unsupported prediction type: {prediction_type}")
            return None
        try:
            handler()
        except Exception as error:
            catch(error)
            return None
        print(f"Prediction saved to {output_filename}")
        return output_filename


@dataclass(slots=True)
class AutoTrainingResult:
    artifact_path: str | None
    plan_markdown: str = ""
    status_markdown: str = ""
    use_result_markdown: str = ""
    inspection_report: dict[str, object] | None = None
    trainer: AutoTrainer | None = None

    @property
    def model_path(self) -> str | None:
        return self.artifact_path

    def __bool__(self) -> bool:
        return bool(self.artifact_path)

    def __str__(self) -> str:
        return self.artifact_path or ""

    def __eq__(self, other) -> bool:
        if isinstance(other, AutoTrainingResult):
            return self.artifact_path == other.artifact_path
        if isinstance(other, (str, bytes, os.PathLike)):
            return self.artifact_path == os.fspath(other)
        return NotImplemented

    def __fspath__(self) -> str:
        if not self.artifact_path:
            raise TypeError("auto-training did not produce a model artifact")
        return self.artifact_path

    def predict(self, data):
        trainer = self.trainer or AutoTrainer(model_path=self.artifact_path)
        return trainer.predict(data)

    def infer(
        self, data, task: str | None = None, model_type: str | None = None
    ):
        trainer = self.trainer or AutoTrainer(
            model_path=self.artifact_path,
            task=task,
        )
        return trainer.infer(data, task=task, model_type=model_type)

    def retrain(self, data=None, target=None, **kwargs):
        return auto_train(
            data,
            target,
            resume_from=self.artifact_path,
            **kwargs,
        )


def _auto_training_direct_option_keys() -> tuple[str, ...]:
    return (
        "label_columns",
        "drop",
        "select",
        "order_by",
        "stratify",
        "validation_split",
        "test_split",
        "batch_size",
        "auto_tune",
        "early_stopping",
        "patience",
        "cv_folds",
    )


def _looks_like_guided_auto_training_input(
    data,
    *,
    remote_src=None,
    local_collection_path=None,
) -> bool:
    if remote_src is not None or local_collection_path is not None:
        return True
    if data is None:
        return False
    trainer = AutoTrainer()
    if isinstance(data, tuple) and len(data) == 2:
        return False
    if trainer._looks_like_remote_source(data):
        return True
    return trainer._looks_like_path_collection(data)


def _guided_auto_training_material(data, *, remote_src=None):
    trainer = AutoTrainer()
    if remote_src is not None:
        return None, remote_src
    if data is not None and trainer._looks_like_remote_source(data):
        return None, data
    return data, None


def _render_auto_training_status(
    artifact_path, *, blocked_reason: str | None = None
):
    if artifact_path:
        return _render_training_status_markdown(
            "Auto Training",
            "ready",
            [f"Artifact: {artifact_path}"],
        )
    details = [blocked_reason or "No artifact produced."]
    return _render_training_status_markdown("Auto Training", "blocked", details)


def _render_training_status_markdown(title: str, status: str, details):
    lines = [f"## {title}", f"- Status: {status}"]
    for detail in details:
        lines.append(f"- {detail}")
    return "\n".join(lines)


def _direct_auto_train_result(
    data=None,
    target=None,
    *,
    save_as: str | None = None,
    resume_from: str | None = None,
    task: str | None = None,
    source_type: str = "parquet",
    revision: str | None = None,
    **kwargs,
) -> AutoTrainingResult:
    from definers.ml.trainer_plan import render_training_plan_markdown

    train_options = {
        key: kwargs[key]
        for key in _auto_training_direct_option_keys()
        if key in kwargs
    }
    trainer = AutoTrainer(
        source=data,
        target=target,
        task=task,
        source_type=source_type,
        auto_tune=bool(train_options.get("auto_tune", True)),
        early_stopping=train_options.get("early_stopping"),
        cv_folds=int(train_options.get("cv_folds", 0) or 0),
    )
    plan = trainer.training_plan(
        resume_from=resume_from,
        revision=revision,
        source_type=source_type,
        **train_options,
    )
    plan_markdown = render_training_plan_markdown(plan)
    artifact_path = trainer.train(
        save_as=save_as,
        resume_from=resume_from,
        revision=revision,
        source_type=source_type,
        **train_options,
    )
    return AutoTrainingResult(
        artifact_path=artifact_path,
        plan_markdown=plan_markdown,
        status_markdown=_render_auto_training_status(artifact_path),
        trainer=trainer,
    )


def _auto_train_result(
    data=None,
    target=None,
    *,
    save_as: str | None = None,
    resume_from: str | None = None,
    task: str | None = None,
    source_type: str = "parquet",
    revision: str | None = None,
    remote_src=None,
    local_collection_path=None,
    intent: str | None = None,
    resolving_choice: str | None = None,
    **kwargs,
):

    use_guided = target is None and _looks_like_guided_auto_training_input(
        data,
        remote_src=remote_src,
        local_collection_path=local_collection_path,
    )
    if not use_guided:
        result = _direct_auto_train_result(
            data,
            target,
            save_as=save_as,
            resume_from=resume_from,
            task=task,
            source_type=source_type,
            revision=revision,
            **kwargs,
        )
        return result

    from definers.ui.apps.train.coach import parse_train_coach_state
    from definers.ui.apps.train.coach_handlers import (
        inspect_train_coach_request,
        preview_train_coach_plan,
        run_train_coach_workflow,
    )

    uploaded_files, resolved_remote = _guided_auto_training_material(
        data,
        remote_src=remote_src,
    )
    requested_intent = intent
    if requested_intent is None:
        if resume_from:
            requested_intent = "resume"
        elif resolved_remote:
            requested_intent = "dataset"
        else:
            requested_intent = "files"
    inspected = inspect_train_coach_request(
        requested_intent,
        uploaded_files,
        resolved_remote,
        revision,
        resume_from,
        save_as,
        local_collection_path=local_collection_path,
        resolving_choice=resolving_choice,
    )
    state_payload = inspected[0]
    state = parse_train_coach_state(state_payload)
    inspection_report = json.loads(state_payload) if state_payload else None
    if state is None or not state.ready:
        plan_markdown = preview_train_coach_plan(state_payload)
        blocked_reason = "Guided validation did not pass."
        if state is not None and state.resolving_question is not None:
            blocked_reason = state.resolving_question.prompt
        elif state is not None and state.unresolved_questions:
            blocked_reason = state.unresolved_questions[0]
        result = AutoTrainingResult(
            artifact_path=None,
            plan_markdown=plan_markdown,
            status_markdown=_render_auto_training_status(
                None,
                blocked_reason=blocked_reason,
            ),
            use_result_markdown=inspected[4],
            inspection_report=inspection_report,
        )
        return result

    artifact_path, plan_markdown, status_markdown, use_result_markdown = (
        run_train_coach_workflow(state_payload)
    )
    result = AutoTrainingResult(
        artifact_path=artifact_path,
        plan_markdown=plan_markdown,
        status_markdown=status_markdown,
        use_result_markdown=use_result_markdown,
        inspection_report=inspection_report,
    )
    return result


def auto_train(
    data,
    target=None,
    *,
    save_as: str | None = None,
    task: str | None = None,
    source_type: str = "parquet",
    cv_folds: int = 0,
    early_stopping: bool | None = None,
    auto_tune: bool = True,
    guided: bool | None = None,
    resume_from: str | None = None,
    **kwargs,
):

    use_guided = guided
    if use_guided is None:
        use_guided = target is None and _looks_like_guided_auto_training_input(
            data,
            remote_src=kwargs.get("remote_src"),
            local_collection_path=kwargs.get("local_collection_path"),
        )
    if use_guided:
        result = _auto_train_result(
            data,
            target,
            save_as=save_as,
            resume_from=resume_from,
            task=task,
            source_type=source_type,
            cv_folds=cv_folds,
            early_stopping=early_stopping,
            auto_tune=auto_tune,
            **kwargs,
        )
        return result

    result = _direct_auto_train_result(
        data,
        target,
        save_as=save_as,
        resume_from=resume_from,
        task=task,
        source_type=source_type,
        cv_folds=cv_folds,
        early_stopping=early_stopping,
        auto_tune=auto_tune,
        **kwargs,
    )
    return result


text = _text

__all__ = [glb for glb in globals() if not glb.startswith("_")]
