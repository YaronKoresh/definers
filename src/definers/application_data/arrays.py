class ArrayService:
    @staticmethod
    def runtime_numpy_modules():
        from definers.application_data.runtime_patches import init_cupy_numpy

        return init_cupy_numpy()

    @staticmethod
    def runtime_vectorizers():
        from definers.application_data.vectorizers import (
            create_vectorizer,
            vectorize,
        )

        return create_vectorizer, vectorize

    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            return None

    @staticmethod
    def log(*args) -> None:
        try:
            from definers.system import log as runtime_log

            runtime_log(*args)
        except Exception:
            return None

    @staticmethod
    def cupy_to_numpy(value: object) -> object:
        try:
            import cupy as cp

            return cp.asnumpy(value)
        except Exception:
            return value

    @staticmethod
    def numpy_to_cupy(value: object) -> object:
        try:
            import cupy as cp

            return cp.array(value)
        except Exception:
            return value

    @staticmethod
    def is_cupy_value(value: object) -> bool:
        return "cupy" in str(type(value))

    @classmethod
    def is_array_value(cls, value: object) -> bool:
        np_module, numpy_module = cls.runtime_numpy_modules()
        runtime_array_type = getattr(np_module, "ndarray", numpy_module.ndarray)
        if runtime_array_type is numpy_module.ndarray:
            return isinstance(value, numpy_module.ndarray)
        return isinstance(value, (numpy_module.ndarray, runtime_array_type))

    @staticmethod
    def numpy_to_str(value) -> str:
        return " ".join(value.flatten().astype(str).tolist())

    @classmethod
    def coerce_existing_array(cls, value):
        np_module, numpy_module = cls.runtime_numpy_modules()
        if numpy_module.issubdtype(
            value.dtype, numpy_module.str_
        ) or numpy_module.issubdtype(value.dtype, numpy_module.bytes_):
            return cls.cupy_to_numpy(cls.str_to_numpy(cls.numpy_to_str(value)))
        if not np_module.issubdtype(value.dtype, np_module.number):
            raise TypeError(
                f"CuPy array of dtype {value.dtype} is not supported."
            )
        return numpy_module.asarray(cls.cupy_to_numpy(value))

    @classmethod
    def coerce_numpy_array(cls, value):
        import torch

        np_module, numpy_module = cls.runtime_numpy_modules()
        if cls.is_cupy_value(value):
            value = cls.cupy_to_numpy(value)
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        if isinstance(value, str):
            return numpy_module.asarray(
                cls.cupy_to_numpy(cls.str_to_numpy(value))
            )
        if cls.is_array_value(value):
            return cls.coerce_existing_array(value)
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

    @classmethod
    def reshape_to_two_dimensions(cls, value):
        arr = cls.coerce_numpy_array(value)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, arr.shape[-1])

    @classmethod
    def reshape_to_three_dimensions(cls, value):
        arr = cls.coerce_numpy_array(value)
        if arr.ndim <= 2:
            return arr.reshape(-1, 1, 1)
        if arr.ndim == 3:
            return arr
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])

    @staticmethod
    def dtype(size: int = 16, is_float: bool = True):
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

    @classmethod
    def str_to_numpy(cls, txt):
        if isinstance(txt, (tuple, list)):
            txt = "".join(txt)
        create_vectorizer, vectorize = cls.runtime_vectorizers()
        vectorizer = create_vectorizer([txt])
        return cls.numpy_to_cupy(vectorize(vectorizer, [txt]))

    @classmethod
    def one_dim_numpy(cls, value):
        return cls.numpy_to_cupy(cls.two_dim_numpy(value).flatten())

    @classmethod
    def two_dim_numpy(cls, value):
        try:
            return cls.numpy_to_cupy(cls.reshape_to_two_dimensions(value))
        except ValueError as error:
            shape = getattr(cls.coerce_numpy_array(value), "shape", None)
            raise ValueError(
                f"Cannot reshape array of shape {shape} to 2D: {error}"
            )

    @classmethod
    def three_dim_numpy(cls, value):
        try:
            return cls.numpy_to_cupy(cls.reshape_to_three_dimensions(value))
        except ValueError as error:
            shape = getattr(cls.coerce_numpy_array(value), "shape", None)
            raise ValueError(
                f"Cannot reshape array of shape {shape} to 3D: {error}"
            )

    @classmethod
    def numpy_to_list(cls, np_arr):
        np_module, _ = cls.runtime_numpy_modules()
        return np_module.concatenate(np_arr, axis=None).ravel().tolist()

    @classmethod
    def guess_numpy_sample_rate(
        cls,
        audio_data,
        possible_sample_rates=None,
        window_type="hann",
        window_size=None,
        peak_prominence=0.01,
        peak_distance=10,
        frequency_threshold=0.05,
    ):
        from scipy import signal as scipy_signal

        _, numpy_module = cls.runtime_numpy_modules()
        audio_data = cls.cupy_to_numpy(audio_data)
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

    @classmethod
    def guess_numpy_type(cls, data) -> str:
        np_module, _ = cls.runtime_numpy_modules()
        cls.numpy_to_list(data)
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

    @classmethod
    def infer_data_type(cls, array):
        try:
            return cls.guess_numpy_type(array)
        except Exception:
            return None

    @staticmethod
    def get_max_shapes(*data) -> list[int]:
        lengths: list[int] = []
        shapes = [np_arr.shape for np_arr in data]
        for shape in shapes:
            while len(lengths) < len(shape):
                lengths.append(0)
            for index, dim in enumerate(shape):
                lengths[index] = max(lengths[index], dim)
        return lengths

    @classmethod
    def pad_nested(cls, nested_data, lengths, fill_value=0):
        _, numpy_module = cls.runtime_numpy_modules()
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
        ret = [
            cls.pad_nested(arr, lengths[1:], fill_value) for arr in nested_data
        ]
        diff = lengths[0] - len(ret)
        if diff > 0:
            ret.extend([cls.pad_nested([], lengths[1:], fill_value)] * diff)
        return ret

    @classmethod
    def reshape_numpy(cls, data, fill_value=0, lengths=None):
        _, numpy_module = cls.runtime_numpy_modules()
        if isinstance(data, numpy_module.ndarray):
            data = data.tolist()
        if not data:
            return numpy_module.array([])
        try:
            if lengths is None:
                lengths = cls.get_max_shapes(data)
            cls.log("Reshaping data", lengths)
            reshaped_data = cls.pad_nested(data, lengths, fill_value)
            cls.log("Reshaped data", lengths)
            return numpy_module.array(reshaped_data)
        except (TypeError, IndexError) as error:
            cls.catch(error)
            return numpy_module.array([])
        except Exception as error:
            cls.catch(error)
            return numpy_module.array([])

    @classmethod
    def pad_or_reshape(cls, arr_list):
        if not arr_list:
            return []
        max_lens = cls.get_max_shapes(*arr_list)
        return [cls.reshape_numpy(arr, lengths=max_lens) for arr in arr_list]

    @staticmethod
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

    @staticmethod
    def tensor_length(tensor) -> int:
        ret = 1
        for num in list(tensor.size()):
            ret *= num
        return ret


cupy_to_numpy = ArrayService.cupy_to_numpy
numpy_to_cupy = ArrayService.numpy_to_cupy
numpy_to_str = ArrayService.numpy_to_str
dtype = ArrayService.dtype
str_to_numpy = ArrayService.str_to_numpy
one_dim_numpy = ArrayService.one_dim_numpy
two_dim_numpy = ArrayService.two_dim_numpy
three_dim_numpy = ArrayService.three_dim_numpy
numpy_to_list = ArrayService.numpy_to_list
guess_numpy_sample_rate = ArrayService.guess_numpy_sample_rate
guess_numpy_type = ArrayService.guess_numpy_type
infer_data_type = ArrayService.infer_data_type
get_max_shapes = ArrayService.get_max_shapes
pad_nested = ArrayService.pad_nested
reshape_numpy = ArrayService.reshape_numpy
pad_or_reshape = ArrayService.pad_or_reshape
convert_tensor_dtype = ArrayService.convert_tensor_dtype
tensor_length = ArrayService.tensor_length
