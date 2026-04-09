def torch_module():
    try:
        import torch

        return torch
    except Exception:
        return None


def runtime_numpy_modules():
    from definers.data.runtime_patches import init_cupy_numpy

    return init_cupy_numpy()


def runtime_vectorizers():
    from definers.data.vectorizers import create_vectorizer, vectorize

    return create_vectorizer, vectorize


def catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        return None


def log(*args) -> None:
    try:
        from definers.system import log as runtime_log

        runtime_log(*args)
    except Exception:
        return None


def cupy_to_numpy(value: object) -> object:
    try:
        import cupy as cp

        return cp.asnumpy(value)
    except Exception:
        return value


def numpy_to_cupy(value: object) -> object:
    try:
        import cupy as cp

        return cp.array(value)
    except Exception:
        return value


def is_cupy_value(value: object) -> bool:
    return "cupy" in str(type(value))


def is_array_value(value: object) -> bool:
    np_module, numpy_module = runtime_numpy_modules()
    runtime_array_type = getattr(np_module, "ndarray", numpy_module.ndarray)
    if runtime_array_type is numpy_module.ndarray:
        return isinstance(value, numpy_module.ndarray)
    return isinstance(value, (numpy_module.ndarray, runtime_array_type))


def numpy_to_str(value) -> str:
    return " ".join(value.flatten().astype(str).tolist())


def coerce_existing_array(value):
    np_module, numpy_module = runtime_numpy_modules()
    if numpy_module.issubdtype(
        value.dtype, numpy_module.str_
    ) or numpy_module.issubdtype(value.dtype, numpy_module.bytes_):
        return cupy_to_numpy(str_to_numpy(numpy_to_str(value)))
    if not np_module.issubdtype(value.dtype, np_module.number):
        raise TypeError(f"CuPy array of dtype {value.dtype} is not supported.")
    return numpy_module.asarray(cupy_to_numpy(value))


def coerce_numpy_array(value):
    np_module, numpy_module = runtime_numpy_modules()
    torch_runtime = torch_module()
    if is_cupy_value(value):
        value = cupy_to_numpy(value)
    if torch_runtime is not None and isinstance(value, torch_runtime.Tensor):
        return value.cpu().numpy()
    if isinstance(value, str):
        return numpy_module.asarray(cupy_to_numpy(str_to_numpy(value)))
    if is_array_value(value):
        return coerce_existing_array(value)
    if isinstance(value, (list, tuple)):
        return numpy_module.array(value)
    if np_module.issubdtype(type(value), numpy_module.number):
        return numpy_module.array([value])
    try:
        return numpy_module.array(value).astype(float)
    except Exception as error:
        raise TypeError(
            f"Input of type {type(value)} is not supported: {error}"
        )


def reshape_to_two_dimensions(value):
    array = coerce_numpy_array(value)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    return array.reshape(-1, array.shape[-1])


def reshape_to_three_dimensions(value):
    array = coerce_numpy_array(value)
    if array.ndim <= 2:
        return array.reshape(-1, 1, 1)
    if array.ndim == 3:
        return array
    return array.reshape(-1, array.shape[-2], array.shape[-1])


def dtype(size: int = 16, is_float: bool = True):
    torch_runtime = torch_module()
    if torch_runtime is None:
        return None
    if size == 16 and is_float and torch_runtime.cuda.is_bf16_supported():
        return torch_runtime.bfloat16
    if size == 16 and is_float:
        return torch_runtime.float16
    if size == 32 and is_float:
        return torch_runtime.float32
    if size == 32 and not is_float:
        return torch_runtime.int
    if size == 16 and not is_float:
        return torch_runtime.int16
    if size == 8 and not is_float:
        return torch_runtime.int8
    return None


def str_to_numpy(txt):
    if isinstance(txt, (tuple, list)):
        txt = "".join(txt)
    create_vectorizer, vectorize = runtime_vectorizers()
    vectorizer = create_vectorizer([txt])
    return numpy_to_cupy(vectorize(vectorizer, [txt]))


def one_dim_numpy(value):
    return numpy_to_cupy(two_dim_numpy(value).flatten())


def two_dim_numpy(value):
    try:
        return numpy_to_cupy(reshape_to_two_dimensions(value))
    except ValueError as error:
        shape = getattr(coerce_numpy_array(value), "shape", None)
        raise ValueError(
            f"Cannot reshape array of shape {shape} to 2D: {error}"
        )


def three_dim_numpy(value):
    try:
        return numpy_to_cupy(reshape_to_three_dimensions(value))
    except ValueError as error:
        shape = getattr(coerce_numpy_array(value), "shape", None)
        raise ValueError(
            f"Cannot reshape array of shape {shape} to 3D: {error}"
        )


def numpy_to_list(np_arr):
    np_module, _ = runtime_numpy_modules()
    return np_module.concatenate(np_arr, axis=None).ravel().tolist()


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

    _, numpy_module = runtime_numpy_modules()
    audio_data = cupy_to_numpy(audio_data)
    if window_size is None:
        window_size = len(audio_data)
    window = scipy_signal.get_window(window_type, window_size)
    frequencies = numpy_module.fft.fftfreq(window_size, d=1.0)
    spectrum = numpy_module.abs(
        numpy_module.fft.fft(audio_data[:window_size] * window)
    )
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
            if (
                abs(frequency) < nyquist_frequency
                and abs(frequency - round(frequency)) / nyquist_frequency
                < frequency_threshold
            ):
                return sample_rate
    return None


def guess_numpy_type(data) -> str:
    np_module, _ = runtime_numpy_modules()
    numpy_to_list(data)
    mean = np_module.mean(data)
    std = np_module.std(data)
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


def get_max_shapes(*data) -> list[int]:
    lengths: list[int] = []
    shapes = [np_arr.shape for np_arr in data]
    for shape in shapes:
        while len(lengths) < len(shape):
            lengths.append(0)
        for index, dim in enumerate(shape):
            lengths[index] = max(lengths[index], dim)
    return lengths


def pad_nested(nested_data, lengths, fill_value=0):
    _, numpy_module = runtime_numpy_modules()
    if isinstance(nested_data, numpy_module.ndarray):
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
    ret = [pad_nested(array, lengths[1:], fill_value) for array in nested_data]
    diff = lengths[0] - len(ret)
    if diff > 0:
        ret.extend([pad_nested([], lengths[1:], fill_value)] * diff)
    return ret


def reshape_numpy(data, fill_value=0, lengths=None):
    _, numpy_module = runtime_numpy_modules()
    if isinstance(data, numpy_module.ndarray):
        data = data.tolist()
    if not data:
        return numpy_module.array([])
    try:
        if lengths is None:
            lengths = get_max_shapes(data)
        log("Reshaping data", lengths)
        reshaped_data = pad_nested(data, lengths, fill_value)
        log("Reshaped data", lengths)
        return numpy_module.array(reshaped_data)
    except (TypeError, IndexError) as error:
        catch(error)
        return numpy_module.array([])
    except Exception as error:
        catch(error)
        return numpy_module.array([])


def pad_or_reshape(arr_list):
    if not arr_list:
        return []
    max_lens = get_max_shapes(*arr_list)
    return [reshape_numpy(arr, lengths=max_lens) for arr in arr_list]


def convert_tensor_dtype(tensor):
    torch_runtime = torch_module()
    if torch_runtime is None:
        return tensor
    if tensor.is_floating_point():
        if tensor.dtype == torch_runtime.float64:
            return tensor.to(torch_runtime.float32)
        return tensor
    if not torch_runtime.is_floating_point(tensor):
        max_val = tensor.max()
        min_val = tensor.min()
        if min_val >= 0:
            if max_val <= 255:
                return tensor.to(torch_runtime.uint8)
            if max_val <= 65535:
                return tensor.to(torch_runtime.uint16)
            if max_val <= 4294967295:
                return tensor.to(torch_runtime.uint32)
            return tensor.to(torch_runtime.uint64)
        if min_val >= -128 and max_val <= 127:
            return tensor.to(torch_runtime.int8)
        if min_val >= -32768 and max_val <= 32767:
            return tensor.to(torch_runtime.int16)
        if min_val >= -2147483648 and max_val <= 2147483647:
            return tensor.to(torch_runtime.int32)
        return tensor.to(torch_runtime.int64)
    return tensor


def tensor_length(tensor) -> int:
    ret = 1
    for num in list(tensor.size()):
        ret *= num
    return ret
