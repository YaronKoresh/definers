import collections
import collections.abc
import importlib
import os
import pickle
import random
import shutil
import site
import subprocess
import sys
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path

import joblib

try:
    import imageio as iio
except ImportError:
    iio = None
try:
    import onnx
except ImportError:
    onnx = None
try:
    import matchering as mg
except ImportError:
    mg = None
try:
    import pydub
except ImportError:
    pydub = None

import contextlib
import io


class _MissingTransformer:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ImportError("sox is not available")


from types import ModuleType
from typing import Any


class _SoxProxy:
    Transformer = _MissingTransformer

    def __getattr__(self, name: str) -> Any:
        raise ImportError("sox module is not available")


from typing import Optional


def _load_sox_module() -> ModuleType | None:
    try:
        buf = io.StringIO()
        import subprocess

        orig_run = subprocess.run
        orig_popen = subprocess.Popen

        def _silent_run(*args, **kwargs):
            kwargs.setdefault("stderr", subprocess.DEVNULL)
            kwargs.setdefault("stdout", subprocess.DEVNULL)
            return orig_run(*args, **kwargs)

        class _SilentPopen(orig_popen):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("stderr", subprocess.DEVNULL)
                kwargs.setdefault("stdout", subprocess.DEVNULL)
                super().__init__(*args, **kwargs)

        subprocess.run = _silent_run
        subprocess.Popen = _SilentPopen
        try:
            with (
                contextlib.redirect_stderr(buf),
                contextlib.redirect_stdout(buf),
            ):
                return importlib.import_module("sox")
        finally:
            subprocess.run = orig_run
            subprocess.Popen = orig_popen
    except ImportError:
        return None
    except Exception:
        return None


sox = _load_sox_module() or _SoxProxy()


def has_sox() -> bool:
    return not isinstance(sox, _SoxProxy)


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None
try:
    import pillow_heif
except ImportError:
    pillow_heif = None
try:
    from torch.utils.data import TensorDataset
except ImportError:
    TensorDataset = None
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None
try:
    from sklearn.preprocessing import Normalizer
except ImportError:
    Normalizer = None
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    SimpleImputer = None
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
try:
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
except ImportError:
    calinski_harabasz_score = None
    davies_bouldin_score = None
    silhouette_score = None
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
collections.MutableSequence = collections.abc.MutableSequence
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import numpy as _np

from definers._audio import (
    analyze_audio,
    analyze_audio_features,
    apply_compressor,
    audio_preview,
    audio_to_midi,
    autotune_song,
    beat_visualizer,
    calculate_active_rms,
    change_audio_speed,
    compact_audio,
    compute_gain_envelope,
    create_sample_audio,
    create_share_links,
    create_spectrum_visualization,
    detect_silence_mask,
    dj_mix,
    export_audio,
    export_to_pkl,
    extend_audio,
    extract_audio_features,
    features_to_audio,
    generate_music,
    generate_voice,
    get_active_audio_timeline,
    get_audio_duration,
    get_audio_feedback,
    get_color_palette,
    get_scale_notes,
    humanize_vocals,
    identify_instruments,
    loudness_maximizer,
    master,
    midi_to_audio,
    normalize_audio_to_peak,
    pitch_shift_vocals,
    predict_audio,
    process_audio_chunks,
    read_mp3,
    remove_silence,
    riaa_filter,
    separate_stems,
    split_audio,
    split_mp3,
    stem_mixer,
    stretch_audio,
    subdivide_beats,
    transcribe_audio,
    value_to_keys,
    write_mp3,
)
from definers._video_gui import (
    apply_global_overlays,
    apply_post_fx,
    draw_custom_element,
    draw_star_of_david,
    filter_styles,
    generate_preview_handler,
    generate_video_handler,
    get_rms_and_beat,
    normalize_arr,
    prepare_common_resources,
    render_frame_base,
)

try:
    __version__ = _pkg_version("definers")
except PackageNotFoundError:
    __version__ = "0.0.0"

from definers._chat import (
    css,
    get_chat_response,
    init_chat,
    init_stable_whisper,
    lyric_video,
    music_video,
    start,
    strip_nikud,
    theme,
)
from definers._constants import (
    CONFIGS,
    FFMPEG_URL,
    MADMOM_AVAILABLE,
    MODELS,
    PROCESSORS,
    STYLES_DB,
    SYSTEM_MESSAGE,
    TOKENIZERS,
    _negative_prompt_,
    _positive_prompt_,
    ai_model_extensions,
    beam_kwargs,
    common_audio_formats,
    higher_beams,
    iio_formats,
    language_codes,
    tasks,
    unesco_mapping,
    user_agents,
)
from definers._cuda import (
    cuda_toolkit,
    cuda_version,
    device,
    free,
    set_cuda_env,
)
from definers._data import (
    TrainingData,
    _find_spec,
    _init_cupy_numpy,
    check_onnx,
    convert_tensor_dtype,
    create_vectorizer,
    cupy_to_numpy,
    drop_columns,
    dtype,
    fetch_dataset,
    files_to_dataset,
    get_max_shapes,
    get_prediction_file_extension,
    guess_numpy_sample_rate,
    guess_numpy_type,
    init_tokenizer,
    is_gpu,
    load_as_numpy,
    merge_columns,
    numpy_to_cupy,
    numpy_to_list,
    numpy_to_str,
    one_dim_numpy,
    order_dataset,
    pad_nested,
    pad_sequences,
    patch_cupy_numpy,
    patch_torch_proxy_mode,
    prepare_data,
    process_rows,
    pytorch_to_onnx,
    read_as_numpy,
    reshape_numpy,
    select_columns,
    select_rows,
    split_columns,
    split_dataset,
    str_to_numpy,
    tensor_length,
    three_dim_numpy,
    to_loader,
    tokenize_and_pad,
    two_dim_numpy,
    unvectorize,
    vectorize,
)
from definers._image import (
    extract_image_features,
    features_to_image,
    get_max_resolution,
    image_resolution,
    init_upscale,
    resize_image,
    save_image,
    upscale,
    write_on_image,
)
from definers._logger import _init_logger
from definers._ml import (
    HybridModel,
    LinearRegressionTorch,
    SklearnWrapper,
    _summarize,
    answer,
    build_faiss,
    check_parameter,
    choose_random_words,
    compile_model,
    convert_vocal_rvc,
    export_files_rvc,
    extract_text_features,
    features_to_text,
    feed,
    find_latest_checkpoint,
    find_latest_rvc_checkpoint,
    fit,
    generate_song,
    get_cluster_content,
    get_model_instructions,
    git,
    infer,
    init_model_file,
    init_model_repo,
    init_pretrained_model,
    initialize_linear_regression,
    is_clusters_model,
    is_huggingface_repo,
    keep_alive,
    kmeans_k_suggestions,
    lang_code_to_name,
    linear_regression,
    map_reduce_summary,
    optimize_prompt_realism,
    pipe,
    predict_linear_regression,
    preprocess_prompt,
    rvc_to_onnx,
    simple_text,
    summary,
    train,
    train_linear_regression,
    train_model_rvc,
)
from definers._system import (
    _install_ffmpeg_linux,
    _install_ffmpeg_windows,
    add_path,
    apt_install,
    big_number,
    catch,
    check_version_wildcard,
    compress,
    copy,
    cores,
    cwd,
    delete,
    directory,
    exist,
    extract,
    find_package_paths,
    full_path,
    get_ext,
    get_linux_distribution,
    get_os_name,
    get_process_pid,
    get_python_version,
    importable,
    install_audio_effects,
    install_faiss,
    install_ffmpeg,
    installed,
    is_admin_windows,
    is_ai_model,
    is_directory,
    is_package_path,
    is_symlink,
    load,
    log,
    modify_wheel_requirements,
    move,
    normalize_path,
    parent_directory,
    path_end,
    path_ext,
    path_name,
    paths,
    permit,
    pip_install,
    post_install,
    pre_install,
    read,
    remove,
    run,
    run_linux,
    run_windows,
    runnable,
    save,
    save_temp_text,
    send_signal_to_process,
    thread,
    tmp,
    unique,
    wait,
    write,
)
from definers._text import (
    Database,
    ai_translate,
    camel_case,
    duck_translate,
    file_to_sha3_512,
    google_translate,
    language,
    number_to_hex,
    random_number,
    random_salt,
    random_string,
    set_system_message,
    string_to_bytes,
    string_to_sha3_512,
    translate_with_code,
)
from definers._video import (
    convert_video_fps,
    extract_video_features,
    features_to_video,
    read_video,
    resize_video,
    write_video,
)
from definers._web import (
    add_to_path_windows,
    download_and_unzip,
    download_file,
    extract_text,
    geo_new_york,
    google_drive_download,
    linked_url,
)

try:
    import cupy as np
except Exception:
    import numpy as np
logger = _init_logger()
importlib.util.find_spec = _find_spec
if _find_spec("dask"):
    import dask
    import dask.array
    import dask.dataframe
    import dask.diagnostics
    from dask import base
    from dask.graph_manipulation import bind, checkpoint, clone, wait_on
    from dask.optimization import cull, fuse, inline, inline_functions
    from dask.utils import key_split

    dask.dataframe.core = dask.dataframe
    dask.diagnostics.core = dask.diagnostics
    dask.array.core = dask.array
    sys.modules["dask.dataframe.core"] = sys.modules["dask.dataframe"]
    sys.modules["dask.diagnostics.core"] = sys.modules["dask.diagnostics"]
    sys.modules["dask.array.core"] = sys.modules["dask.array"]
    dask.core = base
    dask.core.fuse = fuse
    dask.core.cull = cull
    dask.core.inline = inline
    dask.core.inline_functions = inline_functions
    dask.core.key_split = key_split
    dask.core.checkpoint = checkpoint
    dask.core.bind = bind
    dask.core.wait_on = wait_on
    dask.core.clone = clone
    dask.core.get = dask.get

    def _visualize_wrapper(dsk, **kwargs):
        return dask.visualize(dsk, **kwargs)

    dask.core.visualize = _visualize_wrapper
    dask.core.to_graphviz = _visualize_wrapper
try:
    patch_torch_proxy_mode()
except Exception:
    pass
set_system_message(name="Phi", role="a helpful chat assistant")


def predict(prediction_file, model_path):
    from definers._system import sanitize_load_path

    try:
        model_path = sanitize_load_path(model_path)
    except Exception as e:
        from definers._system import catch

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
        vectorizer = create_vectorizer()
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
        img = features_to_image(prediction)
        img_np = cupy_to_numpy(img)
        path = random_string() + ".png"
        iio.imwrite(path, img_np)
        return path
    return None


def init_custom_model(model_type, path):
    if not path or model_type not in ("onnx", "pkl"):
        return None
    from definers._system import sanitize_load_path

    try:
        path = sanitize_load_path(path)
        with open(path, "rb") as f:
            if model_type == "onnx":
                model = onnx.load(f)
            elif model_type == "pkl":
                model = pickle.load(f)
        return model
    except Exception as e:
        catch(f"Error initializing model: {e}")
        return None


from . import _chat, _system
