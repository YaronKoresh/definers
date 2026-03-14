from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from definers.application_data.runtime_patches import init_cupy_numpy
from definers.application_data.vectorizers import create_vectorizer, vectorize


np, _np = init_cupy_numpy()
NumpyArray = NDArray[Any]


def _catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        return None


def _log(*args) -> None:
    try:
        from definers.system import log as runtime_log

        runtime_log(*args)
    except Exception:
        return None


def cupy_to_numpy(value: Any) -> Any:
    try:
        import cupy as cp

        return cp.asnumpy(value)
    except Exception:
        return value


def numpy_to_cupy(value: Any) -> Any:
    try:
        import cupy as cp

        return cp.array(value)
    except Exception:
        return value


def _is_cupy_value(value: Any) -> bool:
    return "cupy" in str(type(value))


def _is_array_value(value: Any) -> bool:
    runtime_array_type = getattr(np, "ndarray", _np.ndarray)
    if runtime_array_type is _np.ndarray:
        return isinstance(value, _np.ndarray)
    return isinstance(value, (_np.ndarray, runtime_array_type))


def numpy_to_str(value):
    return " ".join(value.flatten().astype(str).tolist())


def _coerce_existing_array(value: Any) -> NumpyArray:
    if _np.issubdtype(value.dtype, _np.str_) or _np.issubdtype(value.dtype, _np.bytes_):
        return cupy_to_numpy(str_to_numpy(numpy_to_str(value)))
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"CuPy array of dtype {value.dtype} is not supported.")
    return _np.asarray(cupy_to_numpy(value))


def _coerce_numpy_array(value: Any) -> NumpyArray:
    import torch

    if _is_cupy_value(value):
        value = cupy_to_numpy(value)
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, str):
        return _np.asarray(cupy_to_numpy(str_to_numpy(value)))
    if _is_array_value(value):
        return _coerce_existing_array(value)
    if isinstance(value, (list, tuple)):
        return _np.array(value)
    if np.issubdtype(type(value), _np.number):
        return _np.array([value])
    try:
        return _np.array(value).astype(float)
    except Exception as error:
        raise TypeError(f"Input of type {type(value)} is not supported: {error}")


def _reshape_to_two_dimensions(value: Any) -> NumpyArray:
    arr = _coerce_numpy_array(value)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, arr.shape[-1])


def _reshape_to_three_dimensions(value: Any) -> NumpyArray:
    arr = _coerce_numpy_array(value)
    if arr.ndim <= 2:
        return arr.reshape(-1, 1, 1)
    if arr.ndim == 3:
        return arr
    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])


def dtype(size=16, is_float=True):
    import torch

    if size == 16 and is_float and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if size == 16 and is_float:
        return torch.float16
    if size == 32 and is_float:
        return torch.float32
    if size == 32 and not is_float:
        return torch.int
    if size == 16 and not is_float:
        return torch.int16
    if size == 8 and not is_float:
        return torch.int8
    return None


def str_to_numpy(txt):
    if isinstance(txt, (tuple, list)):
        txt = "".join(txt)
    vec = create_vectorizer([txt])
    return numpy_to_cupy(vectorize(vec, [txt]))


def one_dim_numpy(value):
    return numpy_to_cupy(two_dim_numpy(value).flatten())


def two_dim_numpy(value):
    try:
        return numpy_to_cupy(_reshape_to_two_dimensions(value))
    except ValueError as error:
        shape = getattr(_coerce_numpy_array(value), "shape", None)
        raise ValueError(f"Cannot reshape array of shape {shape} to 2D: {error}")


def three_dim_numpy(value):
    try:
        return numpy_to_cupy(_reshape_to_three_dimensions(value))
    except ValueError as error:
        shape = getattr(_coerce_numpy_array(value), "shape", None)
        raise ValueError(f"Cannot reshape array of shape {shape} to 3D: {error}")


def numpy_to_list(np_arr):
    return np.concatenate(np_arr, axis=None).ravel().tolist()


def guess_numpy_sample_rate(
    audio_data,
    possible_sample_rates=None,
    window_type="hann",
    window_size=None,
    peak_prominence=0.01,
    peak_distance=10,
    frequency_threshold=0.05,
):
    from scipy import signal as scipy_signal

    audio_data = cupy_to_numpy(audio_data)
    if window_size is None:
        window_size = len(audio_data)
    window = scipy_signal.get_window(window_type, window_size)
    frequencies = _np.fft.fftfreq(window_size, d=1.0)
    spectrum = _np.abs(_np.fft.fft(audio_data[:window_size] * window))
    peak_indices = scipy_signal.find_peaks(
        spectrum,
        prominence=peak_prominence,
        distance=peak_distance,
    )[0]
    dominant_frequencies = frequencies[peak_indices]
    if possible_sample_rates is None:
        possible_sample_rates = [22050, 44100, 48000, 88200, 96000, 192000]
    for sample_rate in possible_sample_rates:
        nyquist_frequency = sample_rate / 2
        for frequency in dominant_frequencies:
            if abs(frequency) < nyquist_frequency and abs(frequency - round(frequency)) / nyquist_frequency < frequency_threshold:
                return sample_rate
    return None


def guess_numpy_type(data):
    numpy_to_list(data)
    mean = np.mean(data)
    std = np.std(data)
    ratio = std / mean if mean != 0 else float("inf")
    if data.shape and len(data.shape) > 3:
        return "video"
    if data.shape and len(data.shape) > 2:
        return "image"
    if str(data.dtype)[1] in ["U", "S"]:
        return "text"
    if data.ndim > 1 or str(data.dtype)[1] in ["f"] or ratio > 1:
        return "audio"
    return "text"


def infer_data_type(array):
    try:
        return guess_numpy_type(array)
    except Exception:
        return None


def get_max_shapes(*data):
    lengths: list[int] = []
    shapes = [np_arr.shape for np_arr in data]
    for shape in shapes:
        while len(lengths) < len(shape):
            lengths.append(0)
        for index, dim in enumerate(shape):
            lengths[index] = max(lengths[index], dim)
    return lengths


def pad_nested(nested_data, lengths, fill_value=0):
    if isinstance(nested_data, _np.ndarray):
        nested_data = nested_data.tolist()
    elif isinstance(nested_data, tuple):
        nested_data = list(nested_data)
    if not nested_data:
        return [fill_value] * lengths[0]
    if not isinstance(nested_data[0], list):
        diff = lengths[0] - len(nested_data)
        if diff > 0:
            nested_data.extend([fill_value] * diff)
        return nested_data
    ret = [pad_nested(arr, lengths[1:], fill_value) for arr in nested_data]
    diff = lengths[0] - len(ret)
    if diff > 0:
        ret.extend([pad_nested([], lengths[1:], fill_value)] * diff)
    return ret


def reshape_numpy(data, fill_value=0, lengths=None):
    if isinstance(data, _np.ndarray):
        data = data.tolist()
    if not data:
        return _np.array([])
    try:
        if lengths is None:
            lengths = get_max_shapes(data)
        _log("Reshaping data", lengths)
        reshaped_data = pad_nested(data, lengths, fill_value)
        _log("Reshaped data", lengths)
        return _np.array(reshaped_data)
    except (TypeError, IndexError) as error:
        _catch(error)
        return _np.array([])
    except Exception as error:
        _catch(error)
        return _np.array([])


def pad_or_reshape(arr_list):
    if not arr_list:
        return []
    max_lens = get_max_shapes(*arr_list)
    return [reshape_numpy(arr, lengths=max_lens) for arr in arr_list]


def convert_tensor_dtype(tensor):
    import torch

    if tensor.is_floating_point():
        if tensor.dtype == torch.float64:
            return tensor.to(torch.float32)
        return tensor
    if not torch.is_floating_point(tensor):
        max_val = tensor.max()
        min_val = tensor.min()
        if min_val >= 0:
            if max_val <= 255:
                return tensor.to(torch.uint8)
            if max_val <= 65535:
                return tensor.to(torch.uint16)
            if max_val <= 4294967295:
                return tensor.to(torch.uint32)
            return tensor.to(torch.uint64)
        if min_val >= -128 and max_val <= 127:
            return tensor.to(torch.int8)
        if min_val >= -32768 and max_val <= 32767:
            return tensor.to(torch.int16)
        if min_val >= -2147483648 and max_val <= 2147483647:
            return tensor.to(torch.int32)
        return tensor.to(torch.int64)
    return tensor


def tensor_length(tensor):
    ret = 1
    for num in list(tensor.size()):
        ret *= num
    return ret
