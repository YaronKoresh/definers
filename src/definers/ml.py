import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from time import sleep, time
from types import SimpleNamespace

import numpy as _np
import numpy as np

from definers import regex_utils
from definers.application_ml import answer as _answer
from definers.application_ml.inference import (
    extract_text_features as _extract_text_features,
    features_to_text as _features_to_text,
    predict_linear_regression as _predict_linear_regression,
)
from definers.application_ml.training import (
    HybridModel as _HybridModel,
    LinearRegressionTorch as _LinearRegressionTorch,
    feed as _feed,
    fit as _fit,
    initialize_linear_regression as _initialize_linear_regression,
    linear_regression as _linear_regression,
    train_linear_regression as _train_linear_regression,
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
    language_codes,
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
    from definers.data import (
        create_vectorizer,
        cupy_to_numpy,
        dtype,
        get_max_shapes,
        get_prediction_file_extension,
        guess_numpy_type,
        load_as_numpy,
        numpy_to_cupy,
        one_dim_numpy,
        reshape_numpy,
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
    from definers.video import features_to_video, write_video
except Exception:
    features_to_video = None
    write_video = None
try:
    from definers.web import download_file, google_drive_download
except Exception:
    download_file = None
    google_drive_download = None

logger = init_logger()


def _answer_runtime_adapter():
    return SimpleNamespace(MODELS=MODELS, PROCESSORS=PROCESSORS)


def _linear_regression_runtime_adapter():
    return SimpleNamespace(device=device)


def _train_linear_regression_runtime_adapter():
    return SimpleNamespace(
        device=device,
        initialize_linear_regression=initialize_linear_regression,
    )


def _training_array_adapter():
    return SimpleNamespace(
        catch=catch,
        cupy_to_numpy=cupy_to_numpy,
        get_max_shapes=get_max_shapes,
        numpy_to_cupy=numpy_to_cupy,
        reshape_numpy=reshape_numpy,
    )


def _concatenate_training_rows():
    cupy_module = getattr(np, "cuda", None)
    if cupy_module is not None:
        concatenate = getattr(cupy_module, "cupy", None)
        if concatenate is not None:
            concatenate = getattr(concatenate, "concatenate", None)
            if concatenate is not None:
                return concatenate
    return np.concatenate


def map_reduce_summary(text, max_words):
    from definers.application_ml.text_generation import (
        map_reduce_summary as _map_reduce_summary,
    )

    return _map_reduce_summary(text, max_words)


def optimize_prompt_realism(prompt):
    from definers.application_ml.text_generation import (
        optimize_prompt_realism as _optimize_prompt_realism,
    )

    return _optimize_prompt_realism(prompt)


def preprocess_prompt(prompt):
    from definers.application_ml.text_generation import (
        preprocess_prompt as _preprocess_prompt,
    )

    return _preprocess_prompt(prompt)


def summarize(text_to_summarize):
    from definers.application_ml.text_generation import summarize as _summarize

    return _summarize(text_to_summarize)


def summary(text, max_words=20, min_loops=1):
    from definers.application_ml.text_generation import summary as _summary

    return _summary(text, max_words=max_words, min_loops=min_loops)


def answer(history: list):
    return _answer(history, runtime=_answer_runtime_adapter())


def linear_regression(X, y, learning_rate=0.01, epochs=50):
    return _linear_regression(
        X,
        y,
        learning_rate=learning_rate,
        epochs=epochs,
    )


def initialize_linear_regression(input_dim, model_path):
    return _initialize_linear_regression(
        input_dim,
        model_path,
        runtime=_linear_regression_runtime_adapter(),
        factory=_LinearRegressionTorch,
        logger=logger.info,
    )


def train_linear_regression(X, y, model_path, learning_rate=0.01):
    return _train_linear_regression(
        X,
        y,
        model_path,
        learning_rate=learning_rate,
        runtime=_train_linear_regression_runtime_adapter(),
    )


def init_model_file(task: str, turbo: bool = True, model_type: str = None):
    from definers.application_ml.repository_sync import (
        init_model_file as _init_model_file,
    )

    return _init_model_file(task, turbo=turbo, model_type=model_type)


def kmeans_k_suggestions(X, k_range=range(2, 20), random_state=None):
    try:
        from cuml.cluster import KMeans as cluster_factory
    except Exception:
        from sklearn.cluster import KMeans as cluster_factory

    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    wcss_values = {}
    silhouette_scores = {}
    davies_bouldin_indices = {}
    calinski_harabasz_indices = {}
    suggested_k_elbow = None
    suggested_k_silhouette = None
    suggested_k_davies_bouldin = None
    suggested_k_calinski_harabasz = None
    final_suggestion_k = None
    kmeans_lib = cluster_factory
    is_cupy_available = getattr(np, "__name__", "").lower() == "cupy"
    if is_cupy_available and (kmeans_lib is not None):
        print(
            "GPU acceleration with CuPy (cuML) is available and will be used."
        )
    else:
        print(
            "Warning: CuPy (cuML) is unavailable, falling back to CPU with scikit-learn KMeans."
        )
    X_array = np.asarray(X)
    if len(k_range) < 2:
        return {
            "wcss": wcss_values,
            "silhouette_scores": silhouette_scores,
            "davies_bouldin_indices": davies_bouldin_indices,
            "calinski_harabasz_indices": calinski_harabasz_indices,
            "suggested_k_elbow": suggested_k_elbow,
            "suggested_k_silhouette": suggested_k_silhouette,
            "suggested_k_davies_bouldin": suggested_k_davies_bouldin,
            "suggested_k_calinski_harabasz": suggested_k_calinski_harabasz,
            "final_suggestion": final_suggestion_k,
            "notes": "K-range too small to provide meaningful suggestions. Try a range with at least 2 different k values.",
        }
    for k in k_range:
        if k <= 1:
            wcss_values[k] = 0
            silhouette_scores[k] = np.nan
            davies_bouldin_indices[k] = np.nan
            calinski_harabasz_indices[k] = np.nan
            continue
        kmeans = kmeans_lib(
            n_clusters=int(k), random_state=random_state, init="k-means++"
        )
        labels = kmeans.fit_predict(X_array)
        numpy_labels = np.asnumpy(labels) if is_cupy_available else labels
        numpy_X = np.asnumpy(X_array) if is_cupy_available else X_array
        wcss_values[k] = kmeans.inertia_
        silhouette_scores[k] = silhouette_score(numpy_X, numpy_labels)
        davies_bouldin_indices[k] = davies_bouldin_score(numpy_X, numpy_labels)
        calinski_harabasz_indices[k] = calinski_harabasz_score(
            numpy_X, numpy_labels
        )
    wcss_ratios = {}
    if len(k_range) > 2:
        for i in range(len(k_range) - 1):
            k1 = k_range[i]
            k2 = k_range[i + 1]
            if wcss_values[k1] > 0:
                ratio = wcss_values[k2] / wcss_values[k1]
                wcss_ratios[k2] = ratio
        if wcss_ratios:
            suggested_k_elbow = min(wcss_ratios, key=wcss_ratios.get)
    suggested_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
    suggested_k_davies_bouldin = min(
        davies_bouldin_indices, key=davies_bouldin_indices.get
    )
    suggested_k_calinski_harabasz = max(
        calinski_harabasz_indices, key=calinski_harabasz_indices.get
    )
    if suggested_k_elbow is not None:
        final_suggestion_k = suggested_k_elbow
    elif (
        suggested_k_silhouette is not None
        and silhouette_scores[suggested_k_silhouette] > 0.5
    ):
        final_suggestion_k = suggested_k_silhouette
    elif suggested_k_calinski_harabasz is not None:
        final_suggestion_k = suggested_k_calinski_harabasz
    else:
        final_suggestion_k = None
    return {
        "wcss": wcss_values,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_indices": davies_bouldin_indices,
        "calinski_harabasz_indices": calinski_harabasz_indices,
        "suggested_k_elbow": suggested_k_elbow,
        "suggested_k_silhouette": suggested_k_silhouette,
        "suggested_k_davies_bouldin": suggested_k_davies_bouldin,
        "suggested_k_calinski_harabasz": suggested_k_calinski_harabasz,
        "final_suggestion": final_suggestion_k,
        "random_state": random_state,
        "notes": "Suggestions are based on heuristics. Visualize metrics and use domain knowledge for final k selection. GPU acceleration is automatically used if available.",
    }


def fit(model):
    return _fit(
        model,
        array_adapter=_training_array_adapter(),
        logger=log,
        error_handler=catch,
    )


def feed(model, X_new, y_new=None, epochs=1):
    return _feed(
        model,
        X_new,
        y_new,
        epochs=epochs,
        logger=log,
        concatenate=_concatenate_training_rows(),
    )


from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH


def _validate_str_param(name: str, value: str) -> str:
    if value is None:
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value) > MAX_INPUT_LENGTH:
        raise ValueError(f"{name} too long ({len(value)} > {MAX_INPUT_LENGTH})")
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in value:
        raise ValueError(f"{name} contains too many consecutive spaces")
    return value


def train(
    model_path=None,
    remote_src=None,
    revision=None,
    url_type="parquet",
    features=None,
    labels=None,
    dataset_label_columns=None,
    drop_list=None,
    selected_rows=None,
    order_by=None,
    stratify=None,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    batch_size: int = 1,
):
    import joblib

    from definers.application_data.arrays import numpy_to_cupy
    from definers.application_data.loaders import (
        drop_columns,
        fetch_dataset,
        files_to_dataset,
        select_rows,
        split_columns,
    )
    from definers.application_data.preparation import (
        pad_sequences,
        prepare_data,
        to_loader,
    )
    from definers.application_data.tokenization import (
        init_tokenizer,
        tokenize_and_pad,
    )
    from definers.system import log, secure_path

    if check_parameter(remote_src):
        remote_src = _validate_str_param("remote_src", remote_src)
    got_inp = check_parameter(features) or check_parameter(remote_src)

    if check_parameter(selected_rows):
        selected_rows = _validate_str_param("selected_rows", selected_rows)
    is_supv = check_parameter(dataset_label_columns) or check_parameter(labels)
    tokenizer = init_tokenizer()
    model = None
    if check_parameter(model_path):
        try:
            model_path = secure_path(model_path)
        except Exception as e:
            print(f"Unsafe model path in train(): {e}")
            return None
        model = joblib.load(model_path)
        print(f"cuML model loaded from {model_path}")
        if model is None:
            logging.error(f"Could not load model from {model_path}")
            return None
    model_path = f"model_{random_string()}.joblib"
    if not got_inp:
        return None

    loaders = []
    if check_parameter(selected_rows):
        selected_rows = _validate_str_param("selected_rows", selected_rows)
        if check_parameter(remote_src):
            dataset = fetch_dataset(remote_src, url_type, revision)
        else:
            dataset = files_to_dataset(features, labels)
        dataset = drop_columns(dataset, drop_list)
        log("Full dataset length", len(dataset))
        selected_rows = simple_text(selected_rows).split()
        for part in selected_rows:
            if "-" in part:
                start_end = part.split("-")
                loaders.append(
                    to_loader(
                        select_rows(
                            dataset, int(start_end[0]) - 1, int(start_end[-1])
                        )
                    )
                )
            else:
                loaders.append(
                    to_loader(select_rows(dataset, int(part) - 1, int(part)))
                )
    else:
        td = prepare_data(
            remote_src=remote_src,
            features=features,
            labels=labels,
            url_type=url_type,
            revision=revision,
            drop=drop_list,
            order_by=order_by,
            stratify=stratify,
            val_frac=val_frac,
            test_frac=test_frac,
            batch_size=batch_size,
        )
        if td is None:
            return None

        loaders.append(td.train)
        if td.val is not None:
            loaders.append(td.val)
        if td.test is not None:
            loaders.append(td.test)
        try:
            dataset_len = len(td.train.dataset)
        except Exception:
            dataset_len = None
        if dataset_len is not None:
            log("Full dataset length", dataset_len)
    if is_supv:
        for l, loader in enumerate(loaders):
            logger.info(f"Loader {l + 1}")
            for i, b in enumerate(loader):
                logger.info(f"Batch {i + 1}: {b}")
                (X, y) = split_columns(b, dataset_label_columns, is_batch=True)
                X = tokenize_and_pad(X, tokenizer)
                y = tokenize_and_pad(y, tokenizer)
                X = pad_sequences(X)
                X = numpy_to_cupy(X)
                y = numpy_to_cupy(y)
                logger.info("Feeding model")
                model = feed(model, X, y)
    else:
        for l, loader in enumerate(loaders):
            logger.info(f"Loader {l + 1}")
            for i, b in enumerate(loader):
                logger.info(f"Batch {i + 1}: {b}")
                X = tokenize_and_pad(b, tokenizer)
                X = pad_sequences(X)
                X = numpy_to_cupy(X)
                logger.info("Feeding model")
                model = feed(model, X)
    logger.info("Fitting model")
    fit(model)
    try:
        joblib.dump(model, model_path)
        log("Trained model path", model_path, status=True)
        return model_path
    except Exception as e:
        print(f"Error saving cuML model: {e}")
        return None


def extract_text_features(text, vectorizer=None):
    return _extract_text_features(text, vectorizer)


def predict_linear_regression(X_new, model_path):
    return _predict_linear_regression(
        X_new,
        model_path,
        factory=_LinearRegressionTorch,
    )


def features_to_text(predicted_features, vectorizer=None, vocabulary=None):
    return _features_to_text(
        predicted_features,
        vectorizer=vectorizer,
        vocabulary=vocabulary,
    )


def lang_code_to_name(code):
    from definers.application_ml.introspection import (
        lang_code_to_name as _lang_code_to_name,
    )

    return _lang_code_to_name(code)


def find_latest_rvc_checkpoint(folder_path: str, model_name: str) -> str | None:
    from definers.application_ml.rvc import (
        find_latest_rvc_checkpoint as _find_latest_rvc_checkpoint,
    )

    return _find_latest_rvc_checkpoint(folder_path, model_name)


def get_cluster_content(model, cluster_index):
    from definers.application_ml.introspection import (
        get_cluster_content as _get_cluster_content,
    )

    return _get_cluster_content(model, cluster_index)


def is_clusters_model(model):
    from definers.application_ml.introspection import (
        is_clusters_model as _is_clusters_model,
    )

    return _is_clusters_model(model)


def build_faiss():
    with cwd():
        git("YaronKoresh", "faiss", parent="./xfaiss")
    set_cuda_env()
    cmake = "/usr/local/cmake/bin/cmake"
    try:
        with cwd("./xfaiss"):
            print("faiss - stage 1")
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
                    f"-DPython_EXECUTABLE={sys.executable}",
                    f"-DPython_INCLUDE_DIR={sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}",
                    f"-DPython_LIBRARY={sys.prefix}/lib/libpython{sys.version_info.major}.{sys.version_info.minor}.so",
                    f"-DPython_NumPy_INCLUDE_DIRS={sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/numpy/core/include",
                    ".",
                ]
            )
            print("faiss - stage 2")
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
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "wheel",
                        ".",
                        "-w",
                        temp_dir,
                        "-c",
                        constraints_path,
                    ]
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


def predict(prediction_file: str, model_path: str | list):
    import joblib

    try:
        model_path = secure_path(model_path)
    except Exception as e:
        catch(e)
        return None

    model = joblib.load(model_path)
    if model is None:
        return None
    ext = os.path.splitext(prediction_file)[1].lstrip(".").lower()
    if ext in common_audio_formats:
        return predict_audio(model, prediction_file)
    if ext == "txt":
        data = read(prediction_file)
        vectorizer = create_vectorizer([data])
        features = extract_text_features(data, vectorizer)
    else:
        features = load_as_numpy(prediction_file)
        if features is None:
            return None
    gpu_features = numpy_to_cupy(features)
    flat = one_dim_numpy(gpu_features)
    prediction = model.predict(flat)
    if prediction is None:
        return None
    if is_clusters_model(model):
        prediction = get_cluster_content(model, int(prediction[0]))
    output_type = guess_numpy_type(prediction)
    if output_type == "text":
        text = features_to_text(prediction)
        path = random_string() + ".txt"
        with open(path, "w") as f:
            f.write(text)
        return path
    elif output_type == "image":
        import imageio.v3 as iio

        img = features_to_image(prediction)
        img_np = cupy_to_numpy(img)
        path = random_string() + ".png"
        iio.imwrite(path, img_np)
        return path
    return None


def init_custom_model(model_type: str, path: str | list):

    if not path or model_type not in ("onnx", "pkl"):
        return None

    if not isinstance(path, str) and not isinstance(path, list):
        raise ValueError("model path must be a string or a list")

    try:
        path = secure_path(path)

        with open(path, "rb") as f:
            if model_type == "onnx":
                import onnx

                model = onnx.load(f)
            elif model_type == "pkl":
                import pickle

                model = pickle.load(f)

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

    global MODELS
    global TOKENIZERS
    free()
    model = None
    if task in ["translate"]:
        import nltk
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        nltk.download("punkt_tab")
        TOKENIZERS[task] = AutoTokenizer.from_pretrained(tasks[task])
        model = AutoModelForSeq2SeqLM.from_pretrained(tasks[task]).to(device())
    elif task in ["tts"]:
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device())
    elif task in ["svc"]:
        with cwd():
            if not exist("./infer"):
                logger.info("Initializing RVC by downloading necessary files.")
                git("YaronKoresh", "definers rvc files")
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

        model = pipeline(
            "automatic-speech-recognition",
            model=tasks["speech-recognition"],
            device=device(),
        )
    elif task in ["audio-classification"]:
        from transformers import pipeline

        model = pipeline(
            "audio-classification",
            model=tasks["audio-classification"],
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

        config = AutoConfig.from_pretrained(tasks[task])
        try:
            model = AutoModel.from_pretrained(
                tasks[task],
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype(),
            ).to(device())
        except:
            model = TFAutoModel.from_pretrained(
                tasks[task],
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype(),
            ).to(device())
    elif task in ["music"]:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        PROCESSORS[task] = AutoProcessor.from_pretrained(
            "facebook/musicgen-small"
        )
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        ).to(device())
    elif task in ["answer"]:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
        )

        package_name = "phi4_package"
        print(f"Downloading source files for {tasks[task]}...")
        not_win = get_os_name() != "windows"
        snapshot_dir = Path(
            snapshot_download(
                repo_id=tasks[task],
                allow_patterns=["*.txt", "*.py", "*.json", "*.safetensors"],
                revision="33e62acdd07cd7d6635badd529aa0a3467bb9c6a",
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
                "✅ Source code appears to be already patched. Skipping injection."
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
                    "✅✅✅ SUCCESS: Method injected directly into source code."
                )
            except ValueError:
                print(
                    "⚠️ Could not find the target class declaration line. Aborting patch."
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
        print("✅ Phi-4 model loaded successfully!")
    elif task in ["summary"]:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        TOKENIZERS[task] = T5Tokenizer.from_pretrained(tasks[task])
        free()
        model = T5ForConditionalGeneration.from_pretrained(
            tasks[task], torch_dtype=dtype()
        ).to(device())
    elif task in ["video"]:
        import torch
        from diffusers import (
            HunyuanVideoImageToVideoPipeline,
            HunyuanVideoTransformer3DModel,
        )

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            tasks[task], subfolder="transformer", torch_dtype=dtype()
        )
        model = HunyuanVideoImageToVideoPipeline.from_pretrained(
            tasks[task], transformer=transformer, torch_dtype=dtype()
        )
        model.to(device())
    elif task in ["image"]:
        import torch
        from diffusers import FluxPipeline
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = FluxPipeline.from_pretrained(
            tasks[task], torch_dtype=dtype(), use_safetensors=True
        )
        srpo_path = hf_hub_download(
            repo_id=tasks["image-spro"],
            filename="diffusion_pytorch_model.safetensors",
        )
        state_dict = load_file(srpo_path)
        model.transformer.load_state_dict(state_dict)
        model = model.to(device())
    elif task not in tasks:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            task, torch_dtype=dtype(), trust_remote_code=True
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


def init_pretrained_model(task: str, turbo: bool = True):
    repo_tasks_override = ["svc", "tts"]
    if task in MODELS and MODELS[task]:
        return
    if (
        task in repo_tasks_override
        or (task in tasks and is_huggingface_repo(tasks[task]))
        or is_huggingface_repo(task)
    ):
        return init_model_repo(task, turbo)
    return init_model_file(task, turbo)


def choose_random_words(word_list, num_words=10):
    if not word_list:
        return []
    list_length = len(word_list)
    if num_words > list_length:
        num_words = list_length - 1
    if num_words == 0:
        num_words = 1
    elif num_words == -1:
        return []
    chosen_words = random.sample(word_list, num_words)
    return chosen_words


def pipe(
    task: str,
    *a,
    prompt: str = "",
    path: str = "",
    resolution: str = "640x640",
    length: int = 3,
    fps: int = 24,
):
    import cv2
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    params1 = []
    params2 = {}
    if task in ["image", "video"]:
        log("Pipe activated", prompt, status="")
        (width, height) = resolution.split("x")
        (width, height) = (int(width), int(height))
        if task == "video":
            length = length * fps
        else:
            length = 1
        params2["prompt"] = prompt
        params2["height"] = height
        params2["width"] = width
        params2["guidance_scale"] = 5.0
        if task == "video":
            params2["num_videos_per_prompt"] = 1
            params2["num_frames"] = length
        else:
            params2["max_sequence_length"] = 512
        params2["num_inference_steps"] = 100
        params2["generator"] = torch.Generator(device()).manual_seed(
            random.randint(0, big_number())
        )
    elif task == "detect":
        image = Image.open(path)
        params1.append(image)
    from transformers import AutoTokenizer

    if task in ["detect"]:
        tokenizer = AutoTokenizer.from_pretrained(tasks[task])
        inputs = tokenizer(*params1, **params2, return_tensors="tf")
    elif task in ["image", "video"]:
        inputs = params2
    try:
        outputs = MODELS[task](**inputs)
    except Exception as e:
        catch(e)
        if task == "image":
            outputs = MODELS["video"](**inputs)
        elif task == "video":
            outputs = MODELS["image"](**inputs)
    if task == "video":
        sample = outputs.frames[0]
        path = tmp("mp4")
        export_to_video(sample, path, fps=24)
        return path
    elif task == "image":
        sample = outputs.images[0]
        return save_image(sample)
    elif task == "answer":
        return outputs
    elif task == "detect":
        preds = {}
        for pred in outputs:
            if pred["label"] not in preds:
                preds[pred["label"]] = []
            preds[pred["label"]].append(pred["box"])
        return preds


def check_parameter(p):
    return p is not None and (
        not (
            isinstance(p, list)
            and (len(p) == 0 or (isinstance(p[0], str) and p[0].strip() == ""))
            or (isinstance(p, str) and p.strip() == "")
        )
    )


HybridModel = _HybridModel


LinearRegressionTorch = _LinearRegressionTorch


def SklearnWrapper(sklearn_model, is_classification=False):
    import torch

    class _SklearnWrapper(torch.nn.Module):
        def __init__(self, sklearn_model, is_classification=False):
            super().__init__()
            self.sklearn_model = sklearn_model
            self.is_classification = is_classification

        def forward(self, x, y=None, y_mask=None):
            del y_mask
            x_numpy = self._to_numpy(x)
            if (
                hasattr(self.sklearn_model, "predict_proba")
                and self.is_classification
            ):
                predictions = self.sklearn_model.predict_proba(x_numpy)
            elif (
                hasattr(self.sklearn_model, "decision_function")
                and self.is_classification
            ):
                predictions = self.sklearn_model.decision_function(x_numpy)
            else:
                predictions = self.sklearn_model.predict(x_numpy)
            return torch.tensor(
                predictions, dtype=torch.float32, device=x.device
            )

        def fit(self, x, y=None):
            x_numpy = self._to_numpy(x)
            y_numpy = self._to_numpy(y) if y is not None else None
            if y_numpy is not None:
                self.sklearn_model.fit(x_numpy, y_numpy)
            elif len(x_numpy.shape) > 2:
                logging.warning(
                    "Fitting model on 3D input without labels. Fitting on each sequence independently."
                )
                for i in range(x_numpy.shape[0]):
                    self.sklearn_model.fit(x_numpy[i])
            else:
                self.sklearn_model.fit(x_numpy)

        def _to_numpy(self, tensor_or_array):
            if tensor_or_array is None:
                return None
            if isinstance(tensor_or_array, np.ndarray):
                return tensor_or_array
            if isinstance(tensor_or_array, torch.Tensor):
                return tensor_or_array.cpu().numpy()
            raise ValueError(
                f"Expected torch.Tensor or numpy.ndarray, got {type(tensor_or_array)}"
            )

    return _SklearnWrapper(sklearn_model, is_classification)


def rvc_to_onnx(model_path):
    from definers.system import secure_path

    try:
        model_path = secure_path(model_path)
    except Exception as e:
        logger.error(f"Unsafe model path in rvc_to_onnx: {e}")
        return None
    if not os.path.exists("infer") and (not os.path.exists("infer/")):
        logger.info("Infer module not found, downloading...")
        google_drive_download(
            id="1kqMYQskvVKwKglcWQsK2Q5G3yPahnbtH", dest="./infer.zip"
        )

    try:
        init_pretrained_model("svc")
        from .infer.modules.onnx.export import export_onnx
    except ImportError as ie:
        log(
            "Import Error",
            "Failed to import ONNX export function",
            status=False,
        )
        catch(ie)
        return None
    except Exception as e:
        catch(e)
        return None

    try:
        export_onnx(model_path, model_path.replace(".pth", "") + ".onnx")
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

    now_dir = os.getcwd()
    now_dir = full_path(secure_path(now_dir))
    weight_root = os.path.join(now_dir, "assets", "weights")
    index_root = os.path.join(now_dir, "logs")
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
    onnx_path = secure_path(onnx_path)
    exported_files = [pth_path]
    if exist(onnx_path):
        exported_files.append(onnx_path)
        logger.info(f"Added ONNX file to exported list: {onnx_path}")
    else:
        logger.warning(f"ONNX file not found after export attempt: {onnx_path}")
    if exist(index_file):
        exported_files.append(index_file)
        logger.info(f"Added index file to exported list: {index_file}")
    else:
        logger.warning(f"Index file not found: {index_file}")
    logger.info(f"Exported files: {exported_files}")
    return exported_files


def find_latest_checkpoint(folder_path: str, model_name: str) -> str | None:
    from definers.system import secure_path

    logger.info(
        f"Searching for latest checkpoint in '{folder_path}' with model name '{model_name}'"
    )
    try:
        folder_path = secure_path(folder_path)
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
    from definers.system import secure_path

    logger.info(f"Starting RVC training for experiment: {experiment}")

    try:
        experiment = secure_path(experiment, basename=True)
    except Exception as e:
        logger.error(f"Invalid experiment name: {e}")
        return None

    try:
        path = secure_path(path)
    except Exception as e:
        logger.error(f"Invalid audio path for training: {e}")
        return None
    import pydub
    import torch

    from .configs.config import Config

    path = normalize_audio_to_peak(path)
    (path, music) = separate_stems(path)
    path = normalize_audio_to_peak(path)
    now_dir = os.getcwd()
    index_root = os.path.join(now_dir, "logs")
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
            run(cmd_preprocess)
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

                    run(shlex.split(cmd_f0))
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
                        p = thread(run, shlex.split(cmd_f0_rmvpe))
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
                p = thread(run, shlex.split(cmd_feature_print))
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
            run(cmd_train)
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
    from definers.system import secure_path

    logger.info(f"Starting vocal conversion for experiment: {experiment}")
    try:
        experiment = secure_path(experiment, basename=True)
    except Exception as e:
        logger.error(f"Invalid experiment name: {e}")
        return None

    try:
        from .configs.config import Config
        from .infer.modules.vc.modules import VC
    except ImportError as e:
        logger.error(f"Vocal conversion feature unavailable: {e}")
        return None

    try:
        path = secure_path(path)
    except Exception as e:
        logger.error(f"Invalid audio path for conversion: {e}")
        return None

    path = normalize_audio_to_peak(path)
    (path, music) = separate_stems(path)
    now_dir = os.getcwd()
    now_dir = secure_path(now_dir)
    index_root = os.path.join(now_dir, "logs")
    weight_root = os.path.join(now_dir, "assets", "weights")
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
        vc.get_vc(latest_checkpoint_filename, index_rate, f0_mean_pooling)
        logger.info("VC model loaded.")
    except Exception as e:
        logger.error(f"Failed to load VC model: {e}")
        catch(e)
        return None
    try:
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


def get_model_instructions(task: str, model_type: str) -> str:
    import torch
    import torch.nn as nn

    try:
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
        )
        from sklearn.pipeline import Pipeline

        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
    try:
        import onnxruntime

        ONNX_AVAILABLE = True
    except ImportError:
        ONNX_AVAILABLE = False
    profile = {
        "framework": "Unknown",
        "modalities": set(),
        "architecture": {"type": "Unknown", "details": []},
        "inputs": [],
        "outputs": [],
        "example_code": "",
        "notes": [],
    }

    def _analyze_architecture_pytorch(model_obj):
        if not isinstance(model_obj, nn.Module):
            return
        layer_counts = Counter(
            layer.__class__.__name__ for layer in model_obj.modules()
        )
        if (
            layer_counts["TransformerEncoderLayer"] > 0
            or layer_counts["MultiheadAttention"] > 0
        ):
            profile["architecture"]["type"] = "Transformer-based"
            if layer_counts["Conv2d"] > 2:
                profile["architecture"]["details"].append(
                    f"{layer_counts['TransformerEncoderLayer']} Transformer Blocks indicate a Vision Transformer (ViT) or hybrid architecture."
                )
            else:
                profile["architecture"]["details"].append(
                    f"{layer_counts['MultiheadAttention']} Attention Layers and {layer_counts['Embedding']} Embedding Layers form the core of this NLP/sequence model."
                )
        elif layer_counts["Conv2d"] > 4:
            profile["architecture"]["type"] = (
                "Convolutional Neural Network (CNN)"
            )
            profile["architecture"]["details"].extend(
                [
                    f"{layer_counts['Conv2d']} Conv2d layers",
                    f"{layer_counts['MaxPool2d']} Max-Pooling layers",
                    f"{layer_counts['Linear']} Fully-Connected layers",
                ]
            )
        elif layer_counts["Linear"] > 0:
            profile["architecture"]["type"] = "Multi-Layer Perceptron (MLP)"
            profile["architecture"]["details"].append(
                f"{layer_counts['Linear']} Linear layers"
            )

    def _probe_model_pytorch(model_obj):
        if not isinstance(model_obj, nn.Module):
            return
        try:
            sig = inspect.signature(model_obj.forward)
            dummy_inputs_kwargs = {}
            for param in sig.parameters.values():
                arg_name = param.name
                if arg_name in ["self", "args", "kwargs"]:
                    continue
                input_spec = next(
                    (
                        item
                        for item in profile["inputs"]
                        if item["name"] == arg_name
                    ),
                    None,
                )
                if not input_spec:
                    continue
                shape = tuple(
                    d if isinstance(d, int) else 2 for d in input_spec["shape"]
                )
                dtype_str = input_spec["dtype"]
                if "float" in dtype_str:
                    dummy_inputs_kwargs[arg_name] = torch.randn(
                        shape, dtype=getattr(torch, dtype_str)
                    )
                elif "long" in dtype_str or "int" in dtype_str:
                    vocab_size = next(
                        (
                            l.num_embeddings
                            for l in model_obj.modules()
                            if isinstance(l, nn.Embedding)
                        ),
                        2000,
                    )
                    dummy_inputs_kwargs[arg_name] = torch.randint(
                        0, vocab_size, shape, dtype=torch.long
                    )
            if not dummy_inputs_kwargs:
                profile["notes"].append(
                    "Dynamic probe skipped: could not determine input arguments for `forward` method."
                )
                return
            model_obj.eval()
            with torch.no_grad():
                output = model_obj(**dummy_inputs_kwargs)
            output_tensors = (
                [output]
                if isinstance(output, torch.Tensor)
                else output
                if isinstance(output, (list, tuple))
                else []
            )
            for i, out_tensor in enumerate(output_tensors):
                if isinstance(out_tensor, torch.Tensor):
                    profile["outputs"].append(
                        {
                            "name": f"output_{i}",
                            "shape": tuple(out_tensor.shape),
                            "dtype": str(out_tensor.dtype).replace(
                                "torch.", ""
                            ),
                        }
                    )
            profile["notes"].append(
                "Dynamic probe SUCCESS: Input/Output specifications confirmed."
            )
        except Exception as e:
            profile["notes"].append(
                f"Dynamic probe FAILED: Model `forward` pass raised an error, which may indicate complex input requirements not automatically detectable. Error: {e}"
            )

    def _generate_report():
        modalities_str = (
            ", ".join(sorted([m.capitalize() for m in profile["modalities"]]))
            if profile["modalities"]
            else "Undetermined"
        )
        report = f"## 🔬 Model Deep Dive Analysis: `{task}`\n\n"
        report += f"**Framework**: `{profile['framework']}`\n"
        report += f"**Detected Modality**: `{modalities_str}`\n"
        report += (
            f"**Detected Architecture**: `{profile['architecture']['type']}`\n"
        )
        if profile["architecture"]["details"]:
            details = "\n".join(
                [f"- {d}" for d in profile["architecture"]["details"]]
            )
            report += f"**Architectural Details**:\n{details}\n"
        report += "\n---\n### 📥 Input & 📤 Output Specification\n"
        if not profile["inputs"]:
            report += "**Inputs**: Could not be determined automatically.\n"
        for i, inp in enumerate(profile["inputs"]):
            report += f"- **INPUT `{i}` (`{inp.get('name', 'N/A')}`)**: Shape=`{inp['shape']}`, DType=`{inp['dtype']}`\n"
        if not profile["outputs"]:
            report += "**Outputs**: Not confirmed. Dynamic probe did not run or failed.\n"
        for i, out in enumerate(profile["outputs"]):
            report += f"- **OUTPUT `{i}` (`{out.get('name', 'N/A')}`)**: Shape=`{out['shape']}`, DType=`{out['dtype']}` (Confirmed by probe)\n"
        report += "\n---\n### ⚙️ Preprocessing & Usage Guide\n"
        (example_imports, prep_steps, example_body) = ("", "", "")
        if profile["framework"] == "PyTorch":
            example_imports = "import torch\n"
            example_body = f"model = YourModelClass() # Instantiate your defined model architecture\nmodel.load_state_dict(torch.load('path/to/{task}.pt'))\nmodel.eval()\n\ndummy_inputs = {{}}\n"
            for inp in profile["inputs"]:
                (shape, dtype, name) = (inp["shape"], inp["dtype"], inp["name"])
                if "image" in name or "pixel" in name:
                    (C, H, W) = (shape[1], shape[2], shape[3])
                    prep_steps += f"**For Input `{name}` (Image)**:\n1. Load image (e.g., with Pillow).\n2. Resize to `{H}x{W}`.\n3. Convert to a tensor and normalize (e.g., ImageNet stats).\n4. Ensure shape is `(1, {C}, {H}, {W})`.\n"
                    example_imports += "from PIL import Image\nimport torchvision.transforms as T\n"
                    example_body += "image = Image.open('path/to/image.jpg')\n"
                    example_body += f"preprocess = T.Compose([T.Resize(({H}, {W})), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])\n"
                    example_body += f"dummy_inputs['{name}'] = preprocess(image).unsqueeze(0)\n"
                elif "text" in name or "ids" in name:
                    prep_steps += f"**For Input `{name}` (Text)**:\n1. Use the specific tokenizer the model was trained with.\n2. Convert text to token IDs.\n3. Format as a `{dtype}` tensor with shape `{shape}`.\n"
                    example_body += f"# Use the model's specific tokenizer\ndummy_inputs['{name}'] = torch.randint(0, 1000, {shape}, dtype=torch.long)\n"
            example_body += "\nwith torch.no_grad():\n    output = model(**dummy_inputs)\n    print(f'Output: {output}')\n"
            profile["example_code"] = (
                f"```python\n{example_imports}\n# 1. Define or import your model class (YourModelClass)\n\n# 2. Load model and prepare inputs\n{example_body}```"
            )
        elif profile["framework"] == "scikit-learn":
            prep_steps += "1. Ensure your input data is in the correct format (NumPy array, Pandas DataFrame, or raw text for pipelines).\n2. Apply the exact same feature engineering and preprocessing steps used during training.\n"
            example_body = (
                f"import joblib\nmodel = joblib.load('path/to/{task}.pkl')\n"
            )
            if "TfidfVectorizer" in profile["architecture"]["details"]:
                example_body += "input_data = ['your first sentence', 'your second sentence']\n"
            else:
                n_features = profile["inputs"][0]["shape"][1]
                example_body += f"import numpy as np\n# Input must be a 2D array with shape (n_samples, {n_features})\ninput_data = np.random.rand(2, {n_features})\n"
            example_body += (
                "predictions = model.predict(input_data)\nprint(predictions)"
            )
            profile["example_code"] = f"```python\n{example_body}\n```"
        report += (
            prep_steps
            + "\n#### Example Usage Snippet\n"
            + profile["example_code"]
        )
        if profile["notes"]:
            report += "\n---\n### 🕵️ Analyst Notes\n" + "\n".join(
                [f"- {n}" for n in profile["notes"]]
            )
        return report

    model_object = MODELS.get(task)
    if model_object is None:
        log(
            f"Analysis Failed for '{task}'",
            f"Model file `{task}` could not be found or loaded.",
            status=False,
        )
        return
    if isinstance(model_object, nn.Module) or isinstance(
        model_object, (dict, OrderedDict)
    ):
        profile["framework"] = "PyTorch"
        if isinstance(model_object, (dict, OrderedDict)):
            profile["architecture"]["type"] = "Raw State Dictionary"
            profile["notes"].append(
                "Analysis is based on a state_dict, not a full model. Instantiate the model class before loading these weights."
            )
            (first_key, first_tensor) = next(iter(model_object.items()))
            in_features = (
                first_tensor.shape[1] if len(first_tensor.shape) == 2 else "N/A"
            )
            profile["inputs"].append(
                {
                    "name": "input_0",
                    "shape": f"(batch_size, {in_features})",
                    "dtype": str(first_tensor.dtype),
                }
            )
            profile["modalities"].add(
                "Tabular" if in_features != "N/A" else "Unknown"
            )
        else:
            sig = inspect.signature(model_object.forward)
            for param in sig.parameters.values():
                if param.name in ["self", "args", "kwargs"]:
                    continue
                name = param.name
                if "image" in name or "pixel" in name:
                    profile["modalities"].add("Image")
                    profile["inputs"].append(
                        {
                            "name": name,
                            "shape": (1, 3, 32, 32),
                            "dtype": "float32",
                        }
                    )
                elif "text" in name or "ids" in name:
                    profile["modalities"].add("Text")
                    profile["inputs"].append(
                        {"name": name, "shape": (1, 16), "dtype": "long"}
                    )
                else:
                    profile["modalities"].add("Tabular")
                    profile["inputs"].append(
                        {"name": name, "shape": (1, 64), "dtype": "float32"}
                    )
            _analyze_architecture_pytorch(model_object)
            _probe_model_pytorch(model_object)
    elif SKLEARN_AVAILABLE and hasattr(model_object, "predict"):
        profile["framework"] = "scikit-learn"
        if isinstance(model_object, Pipeline):
            profile["architecture"]["type"] = "Scikit-learn Pipeline"
            steps = [
                f"{name} ({step.__class__.__name__})"
                for (name, step) in model_object.steps
            ]
            profile["architecture"]["details"] = steps
            if any("TfidfVectorizer" in s for s in steps):
                profile["modalities"].add("Text")
                profile["inputs"].append(
                    {
                        "name": "raw_text",
                        "shape": "(n_samples,)",
                        "dtype": "string",
                    }
                )
        else:
            profile["architecture"]["type"] = (
                f"Standard Model ({model_object.__class__.__name__})"
            )
            profile["modalities"].add("Tabular")
            n_features = getattr(model_object, "n_features_in_", "N/A")
            profile["inputs"].append(
                {
                    "name": "X",
                    "shape": f"(n_samples, {n_features})",
                    "dtype": "float",
                }
            )
    elif ONNX_AVAILABLE and isinstance(
        model_object, onnxruntime.InferenceSession
    ):
        profile["framework"] = "ONNX"
        profile["architecture"]["type"] = "ONNX Graph"
        for inp in model_object.get_inputs():
            profile["inputs"].append(
                {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
            )
            if len(inp.shape) == 4 and inp.shape[1] in [1, 3]:
                profile["modalities"].add("Image")
        for out in model_object.get_outputs():
            profile["outputs"].append(
                {"name": out.name, "shape": out.shape, "dtype": out.type}
            )
    final_report = _generate_report()
    log(f"Deep Dive Analysis for '{task}'", final_report)


def infer(task: str, inference_file: str, model_type: str = None):
    import imageio as iio
    import torch
    from scipy.io import wavfile

    vec = None
    input_data = None
    model_path = task
    if task in tasks:
        model_path = tasks[task]
    if not model_type:
        model_type = get_ext(model_path)
    model_type = model_type.lower()
    if not (task in MODELS and MODELS[task]):
        init_model_file(task)
    mod = MODELS[task]
    if mod is None:
        return None
    file_ext = get_ext(inference_file)
    if file_ext == "txt":
        txt = read(inference_file)
        if isinstance(txt, (tuple, list)):
            txt = "".join(txt)
        vec = create_vectorizer([txt])
        input_data = numpy_to_cupy(extract_text_features(txt, vec))
    elif file_ext in common_audio_formats:
        out = predict_audio(mod, inference_file)
        print(f"Prediction saved to {out}")
        return out
    else:
        input_data = numpy_to_cupy(load_as_numpy(inference_file))
    if input_data is None:
        log("Could not load input data", inference_file, status=False)
        return None
    pred = None
    input_numpy = cupy_to_numpy(one_dim_numpy(input_data))
    try:
        if model_type in ["joblib", "pkl"]:
            pred = mod.predict(input_numpy)
        elif model_type in ["pt", "pth", "safetensors"]:
            input_tensor = torch.from_numpy(input_numpy).to(device())
            with torch.no_grad():
                output_tensor = mod(input_tensor)
            pred = output_tensor.cpu().numpy()
        elif model_type == "onnx":
            input_name = mod.get_inputs()[0].name
            onnx_output = mod.run(
                None, {input_name: input_numpy.astype(np.float32)}
            )
            pred = onnx_output[0]
    except Exception as e:
        logging.error(f"Model prediction failed for type '{model_type}': {e}")
        return None
    if pred is None:
        logging.error("Model prediction returned None.")
        return None
    if is_clusters_model(mod):
        pred = one_dim_numpy(get_cluster_content(mod, int(pred[0])))
    pred_type = guess_numpy_type(pred)
    output_filename = (
        f"{random_string()}.{get_prediction_file_extension(pred_type)}"
    )
    if vec is not None:
        pred = features_to_text(pred)
    elif pred_type == "text":
        pred = features_to_text(pred)
    elif pred_type == "audio":
        pred = features_to_audio(pred)
    elif pred_type == "image":
        pred = features_to_image(pred)
    elif pred_type == "video":
        pred = features_to_video(pred)
    handlers = {
        "video": lambda: write_video(pred, 24),
        "image": lambda: iio.imwrite(
            output_filename, (pred * 255).astype(np.uint8)
        ),
        "audio": lambda: wavfile.write(output_filename, 32000, pred),
        "text": lambda: open(output_filename, "w").write(pred),
    }
    if pred_type in handlers:
        try:
            handlers[pred_type]()
        except Exception as e:
            catch(e)
            return None
    else:
        logging.error(f"Unsupported prediction type: {pred_type}")
        return None
    print(f"Prediction saved to {output_filename}")
    return output_filename


def compile_model(model_or_pipeline):
    import inspect
    import types

    import torch

    try:
        from diffusers import DiffusionPipeline
        from diffusers.models.modeling_utils import ModelMixin
        from transformers import PreTrainedModel
    except ImportError:
        warnings.warn(
            "Please install `diffusers` and `transformers` for full functionality."
        )
        return model_or_pipeline
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile() is not available. Please use PyTorch 2.0 or newer."
        )
        return model_or_pipeline
    compile_kwargs = {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
    }

    def patch_forward(obj):
        if not hasattr(obj, "forward"):
            return obj
        orig_forward = obj.forward
        src = inspect.getsource(orig_forward)
        if ".to(sample.device)" in src:
            patched_src = src.replace(
                ".to(sample.device)", ".to(sample.device.type)"
            )
            globals_dict = orig_forward.__globals__.copy()
            exec(patched_src, globals_dict)
            new_forward = globals_dict[orig_forward.__name__]
            obj.forward = types.MethodType(new_forward, obj)
            print(
                f"⚡️ Patched {type(obj).__name__}.forward to avoid .to(sample.device) bug for torch.compile."
            )
        return obj

    if isinstance(model_or_pipeline, DiffusionPipeline):
        print(
            "✅ Detected a Diffusers pipeline. Dynamically compiling submodels..."
        )
        for attr_name, attr_value in model_or_pipeline.__dict__.items():
            if isinstance(attr_value, ModelMixin):
                try:
                    attr_value = patch_forward(attr_value)
                    print(f"   -> Compiling {attr_name}...")
                    if attr_name == "vae":
                        attr_value.decode = torch.compile(
                            attr_value.decode, **compile_kwargs
                        )
                    else:
                        setattr(
                            model_or_pipeline,
                            attr_name,
                            torch.compile(attr_value, **compile_kwargs),
                        )
                except Exception as e:
                    warnings.warn(
                        f"Could not compile submodel '{attr_name}'. Reason: {e}"
                    )
        return model_or_pipeline
    elif isinstance(model_or_pipeline, PreTrainedModel):
        print("✅ Detected a Transformers model. Compiling the model...")
        try:
            return torch.compile(model_or_pipeline, **compile_kwargs)
        except Exception as e:
            warnings.warn(f"Could not compile the model. Reason: {e}")
            return model_or_pipeline
    else:
        warnings.warn(
            "Object is not a recognized Diffusers pipeline or Transformers model. No action taken."
        )
        return model_or_pipeline


def keep_alive(fn, outputs: int = 1):
    import gradio as gr

    def worker(*args, **kwargs):
        yld = None
        if outputs >= 2:
            yld = (gr.update(),) * outputs
        elif outputs == 1:
            yld = gr.update()

        def thread_target(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                return catch(e)

        t = thread(thread_target, *args, **kwargs)
        sleep(5)
        counter = 5
        if outputs == 0:
            while t[1].is_alive():
                gr.Info(f"Time passed: {str(counter)}s", duration=1.0)
                sleep(5)
                counter += 5
        else:
            while t[1].is_alive():
                yield yld
                gr.Info(f"Time passed: {str(counter)}s", duration=1.0)
                sleep(5)
                counter += 5
        if outputs == 0:
            wait(t)
        elif outputs >= 1:
            yield wait(t)[0]

    return worker
