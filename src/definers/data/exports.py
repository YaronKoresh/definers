def read_as_numpy(path: str):
    from definers.data.loaders import load_as_numpy

    return load_as_numpy(path)


def get_prediction_file_extension(pred_type):
    if pred_type is None:
        return "data"
    pred_type_lower = pred_type.lower().strip()
    if pred_type_lower == "video":
        return "mp4"
    if pred_type_lower == "image":
        return "png"
    if pred_type_lower == "audio":
        return "wav"
    if pred_type_lower == "text":
        return "txt"
    return "data"


def is_gpu():
    import torch

    return torch.cuda.is_available()


def check_onnx(path):
    import onnx

    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        return False
    return True


def pytorch_to_onnx(model_torch, input_dim, onnx_path="model.onnx"):
    import torch

    dummy_input = torch.randn(1, input_dim).cuda()
    torch.onnx.export(model_torch, dummy_input, onnx_path, verbose=True)
    return onnx_path
