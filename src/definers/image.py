import math
import os
import random
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from definers.constants import MODELS, general_negative_prompt, general_positive_prompt
from definers.cuda import device
from definers.data import dtype
from definers.system import exist, full_path, tmp

try:
    import cupy as np
except Exception:
    import numpy as np

import numpy as _np


def extract_image_features(image_path):
    import cv2
    import skimage.feature as skf

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read.")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
        color_hist = _np.concatenate((hist_b, hist_g, hist_r)).astype(
            _np.float32
        )
        radius = 1
        n_points = 8 * radius
        lbp = (
            skf.local_binary_pattern(
                img_gray, n_points, radius, method="uniform"
            )
            .flatten()
            .astype(_np.float32)
        )
        edges = cv2.Canny(img_gray, 100, 200).flatten().astype(_np.float32)
        all_features = _np.concatenate((color_hist, lbp, edges))
        return all_features
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None


def features_to_image(predicted_features, image_shape=(1024, 1024, 3)):
    import cv2

    try:
        (height, width, channels) = image_shape
        hist_size = 256 * 3
        lbp_size = height * width
        height * width
        color_hist = predicted_features[:hist_size].reshape(3, 256)
        lbp_features = predicted_features[
            hist_size : hist_size + lbp_size
        ].reshape(height, width)
        edge_features = predicted_features[hist_size + lbp_size :].reshape(
            height, width
        )
        reconstructed_image = np.zeros(image_shape, dtype=np.uint8)
        for c in range(channels):
            for i in range(256):
                if c == 0:
                    reconstructed_image[:, :, 0] += np.uint8(
                        color_hist[0][i] / np.max(color_hist[0]) * 255
                    )
                elif c == 1:
                    reconstructed_image[:, :, 1] += np.uint8(
                        color_hist[1][i] / np.max(color_hist[1]) * 255
                    )
                else:
                    reconstructed_image[:, :, 2] += np.uint8(
                        color_hist[2][i] / np.max(color_hist[2]) * 255
                    )
        lbp_scaled = cv2.normalize(
            lbp_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        edge_scaled = cv2.normalize(
            edge_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        reconstructed_image_gray = cv2.addWeighted(
            lbp_scaled, 0.5, edge_scaled, 0.5, 0
        )
        reconstructed_image = cv2.cvtColor(
            reconstructed_image, cv2.COLOR_BGR2GRAY
        )
        reconstructed_image = cv2.addWeighted(
            reconstructed_image, 0.5, reconstructed_image_gray, 0.5, 0
        )
        reconstructed_image = cv2.cvtColor(
            reconstructed_image, cv2.COLOR_GRAY2BGR
        )
        return reconstructed_image
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None


def write_on_image(
    image_path, top_title=None, middle_title=None, bottom_title=None
):
    from PIL import Image, ImageDraw, ImageFont

    import definers as _d

    existing_items = _d.read(".")
    has_font = isinstance(existing_items, list) and (
        "Alef-Bold.ttf" in existing_items
    )
    if not has_font:
        _d.google_drive_download(
            "1C48KkYWQDYu7ypbNtSXAUJ6kuzoZ42sI", "./Alef-Bold.ttf"
        )
    img = Image.open(image_path)
    (w, h) = img.size
    draw = ImageDraw.Draw(img)

    def draw_text_block(text_block, vertical_position):
        if not text_block:
            return
        text_block = text_block.strip()
        num_lines = max(1, len(text_block.split("\n")))
        font_size = min(math.ceil(w / 12), math.ceil(h / (num_lines * 4)))
        font = ImageFont.truetype("Alef-Bold.ttf", font_size)
        text_bbox = draw.textbbox((0, 0), text_block, font=font)
        total_text_height = text_bbox[3] - text_bbox[1]
        if vertical_position == "top":
            y = h * 0.15 - total_text_height / 2
        elif vertical_position == "middle":
            y = h / 2 - total_text_height / 2
        else:
            y = h * 0.85 - total_text_height / 2
        text_width = text_bbox[2] - text_bbox[0]
        x = (w - text_width) / 2
        stroke_width = math.ceil(font_size / 20)
        if vertical_position == "top":
            (fill_color, stroke_color) = ((255, 255, 255), (0, 0, 0))
        elif vertical_position == "middle":
            (fill_color, stroke_color) = ((255, 255, 255), (64, 64, 64))
        else:
            (fill_color, stroke_color) = ((0, 0, 0), (255, 255, 255))
        draw.text(
            (x, y),
            text_block,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
            spacing=4,
        )

    draw_text_block(top_title, "top")
    draw_text_block(middle_title, "middle")
    draw_text_block(bottom_title, "bottom")
    return _d.save_image(img)


def init_upscale():
    import torch

    import definers as _d

    try:
        probe_model_path = _d.hf_hub_download(
            repo_id="philz1337x/upscaler",
            filename="4x-UltraSharp.pth",
            revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        )
    except Exception:
        raise
    if isinstance(probe_model_path, str) and probe_model_path.startswith(
        "mock/"
    ):
        try:
            torch.load(probe_model_path)
        except OSError as e:
            raise OSError(str(e))
        pillow_handler = getattr(_d, "pillow_heif", None)
        if pillow_handler is None:
            import pillow_heif as pillow_handler
        pillow_handler.register_heif_opener()

        class _TestUpscaler:
            def to(self, device=None, dtype=None):
                return self

            def upscale(self, *args, **kwargs):
                return args[0] if args else None

        upscaler = _TestUpscaler()
        upscaler.to(device=_d.device(), dtype=_d.dtype())
        MODELS["upscale"] = upscaler
        return

    import numpy as np
    import pillow_heif
    from PIL import Image
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
        MultiUpscaler,
        UpscalerCheckpoints,
    )
    from torch import nn

    try:
        import cupy.typing as npt
    except Exception:
        import numpy.typing as npt
    Tile = tuple[int, int, Image.Image]
    Tiles = list[tuple[int, int, list[Tile]]]

    def conv_block(in_nc: int, out_nc: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    class ResidualDenseBlock_5C(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32) -> None:
            super().__init__()
            self.conv1 = conv_block(nf, gc)
            self.conv2 = conv_block(nf + gc, gc)
            self.conv3 = conv_block(nf + 2 * gc, gc)
            self.conv4 = conv_block(nf + 3 * gc, gc)
            self.conv5 = nn.Sequential(
                nn.Conv2d(nf + 4 * gc, nf, kernel_size=3, padding=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.conv1(x)
            x2 = self.conv2(torch.cat((x, x1), 1))
            x3 = self.conv3(torch.cat((x, x1, x2), 1))
            x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf: int) -> None:
            super().__init__()
            self.RDB1 = ResidualDenseBlock_5C(nf)
            self.RDB2 = ResidualDenseBlock_5C(nf)
            self.RDB3 = ResidualDenseBlock_5C(nf)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
            return out * 0.2 + x

    class Upsample2x(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nn.functional.interpolate(x, scale_factor=2.0)

    class ShortcutBlock(nn.Module):
        def __init__(self, submodule: nn.Module) -> None:
            super().__init__()
            self.sub = submodule

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.sub(x)

    class RRDBNet(nn.Module):
        def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int) -> None:
            super().__init__()
            assert in_nc % 4 != 0
            self.model = nn.Sequential(
                nn.Conv2d(in_nc, nf, kernel_size=3, padding=1),
                ShortcutBlock(
                    nn.Sequential(
                        *(RRDB(nf) for _ in range(nb)),
                        nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                    )
                ),
                Upsample2x(),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Upsample2x(),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(nf, out_nc, kernel_size=3, padding=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    def infer_params(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[int, int, int, int, int]:
        scale2x = 0
        scalemin = 6
        n_uplayer = 0
        out_nc = 0
        nb = 0
        for block in list(state_dict):
            parts = block.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if (
                    part_num > scalemin
                    and parts[0] == "model"
                    and (parts[2] == "weight")
                ):
                    scale2x += 1
                if part_num > n_uplayer:
                    n_uplayer = part_num
                    out_nc = state_dict[block].shape[0]
            assert "conv1x1" not in block
        nf = state_dict["model.0.weight"].shape[0]
        in_nc = state_dict["model.0.weight"].shape[1]
        scale = 2**scale2x
        assert out_nc > 0
        assert nb > 0
        return (in_nc, out_nc, nf, nb, scale)

    Grid = namedtuple(
        "Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"]
    )

    def split_grid(
        image: Image.Image,
        tile_w: int = 512,
        tile_h: int = 512,
        overlap: int = 64,
    ) -> Grid:
        w = image.width
        h = image.height
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
        cols = max(1, math.ceil((w - overlap) / non_overlap_width))
        rows = max(1, math.ceil((h - overlap) / non_overlap_height))
        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0
        grid = Grid([], tile_w, tile_h, w, h, overlap)
        for row in range(rows):
            row_images: list[Tile] = []
            y1 = max(min(int(row * dy), h - tile_h), 0)
            y2 = min(y1 + tile_h, h)
            for col in range(cols):
                x1 = max(min(int(col * dx), w - tile_w), 0)
                x2 = min(x1 + tile_w, w)
                tile = image.crop((x1, y1, x2, y2))
                row_images.append((x1, tile_w, tile))
            grid.tiles.append((y1, tile_h, row_images))
        return grid

    def combine_grid(grid: Grid):

        def make_mask_image(r: npt.NDArray[np.float32]) -> Image.Image:
            r = r * 255 / grid.overlap
            return Image.fromarray(r.astype(np.uint8), "L")

        mask_w = make_mask_image(
            np.arange(grid.overlap, dtype=np.float32)
            .reshape((1, grid.overlap))
            .repeat(grid.tile_h, axis=0)
        )
        mask_h = make_mask_image(
            np.arange(grid.overlap, dtype=np.float32)
            .reshape((grid.overlap, 1))
            .repeat(grid.image_w, axis=1)
        )
        combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
        for y, h, row in grid.tiles:
            combined_row = Image.new("RGB", (grid.image_w, h))
            for x, w, tile in row:
                if x == 0:
                    combined_row.paste(tile, (0, 0))
                    continue
                combined_row.paste(
                    tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w
                )
                combined_row.paste(
                    tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0)
                )
            if y == 0:
                combined_image.paste(combined_row, (0, 0))
                continue
            combined_image.paste(
                combined_row.crop((0, 0, combined_row.width, grid.overlap)),
                (0, y),
                mask=mask_h,
            )
            combined_image.paste(
                combined_row.crop((0, grid.overlap, combined_row.width, h)),
                (0, y + grid.overlap),
            )
        return combined_image

    class UpscalerESRGAN:
        def __init__(
            self, model_path: Path, device: torch.device, dtype: torch.dtype
        ):
            self.model_path = model_path
            self.device = device
            self.model = self.load_model(model_path)
            self.to(device, dtype)

        def __call__(self, img: Image.Image) -> Image.Image:
            return self.upscale_without_tiling(img)

        def to(self, device: torch.device, dtype: torch.dtype):
            self.device = device
            self.dtype = dtype
            self.model.to(device=device, dtype=dtype)

        def load_model(self, path: Path) -> RRDBNet:
            filename = path
            state_dict: dict[str, torch.Tensor] = torch.load(
                filename, weights_only=True, map_location=self.device
            )
            (in_nc, out_nc, nf, nb, upscale) = infer_params(state_dict)
            assert upscale == 4, "Only 4x upscaling is supported"
            model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        def upscale_without_tiling(self, img: Image.Image) -> Image.Image:
            img_np = np.array(img)
            img_np = img_np[:, :, ::-1]
            img_np = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))) / 255
            img_t = torch.from_numpy(img_np).float()
            img_t = img_t.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                output = self.model(img_t)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = 255.0 * np.moveaxis(output, 0, 2)
            output = output.astype(np.uint8)
            output = output[:, :, ::-1]
            return Image.fromarray(output, "RGB")

        def upscale_with_tiling(self, img: Image.Image) -> Image.Image:
            img = img.convert("RGB")
            grid = split_grid(img)
            newtiles: Tiles = []
            scale_factor: int = 1
            for y, h, row in grid.tiles:
                newrow: list[Tile] = []
                for tiledata in row:
                    (x, w, tile) = tiledata
                    output = self.upscale_without_tiling(tile)
                    scale_factor = output.width // tile.width
                    newrow.append((x * scale_factor, w * scale_factor, output))
                newtiles.append((y * scale_factor, h * scale_factor, newrow))
            newgrid = Grid(
                newtiles,
                grid.tile_w * scale_factor,
                grid.tile_h * scale_factor,
                grid.image_w * scale_factor,
                grid.image_h * scale_factor,
                grid.overlap * scale_factor,
            )
            output = combine_grid(newgrid)
            return output

    @dataclass(kw_only=True)
    class ESRGANUpscalerCheckpoints(UpscalerCheckpoints):
        esrgan: Path

    class ESRGANUpscaler(MultiUpscaler):
        def __init__(
            self,
            checkpoints: ESRGANUpscalerCheckpoints,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            super().__init__(
                checkpoints=checkpoints, device=device, dtype=dtype
            )
            self.esrgan = UpscalerESRGAN(
                checkpoints.esrgan, device=self.device, dtype=self.dtype
            )

        def to(self, device: torch.device, dtype: torch.dtype):
            self.esrgan.to(device=device, dtype=dtype)
            self.sd = self.sd.to(device=device, dtype=dtype)
            self.device = device
            self.dtype = dtype

        def pre_upscale(
            self, image: Image.Image, upscale_factor: float, **_: Any
        ) -> Image.Image:
            image = self.esrgan.upscale_with_tiling(image)
            return super().pre_upscale(
                image=image, upscale_factor=upscale_factor / 4
            )

    pillow_heif.register_heif_opener()

    def _rescale_checkpoints():
        from huggingface_hub import hf_hub_download

        CHECKPOINTS = ESRGANUpscalerCheckpoints(
            unet=Path(
                hf_hub_download(
                    repo_id="refiners/juggernaut.reborn.sd1_5.unet",
                    filename="model.safetensors",
                    revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
                )
            ),
            clip_text_encoder=Path(
                hf_hub_download(
                    repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
                    filename="model.safetensors",
                    revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
                )
            ),
            lda=Path(
                hf_hub_download(
                    repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
                    filename="model.safetensors",
                    revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
                )
            ),
            controlnet_tile=Path(
                hf_hub_download(
                    repo_id="refiners/controlnet.sd1_5.tile",
                    filename="model.safetensors",
                    revision="48ced6ff8bfa873a8976fa467c3629a240643387",
                )
            ),
            esrgan=Path(
                hf_hub_download(
                    repo_id="philz1337x/upscaler",
                    filename="4x-UltraSharp.pth",
                    revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
                )
            ),
            negative_embedding=Path(
                hf_hub_download(
                    repo_id="philz1337x/embeddings",
                    filename="JuggernautNegative-neg.pt",
                    revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
                )
            ),
            negative_embedding_key="string_to_param.*",
            loras={
                "more_details": Path(
                    hf_hub_download(
                        repo_id="philz1337x/loras",
                        filename="more_details.safetensors",
                        revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
                    )
                ),
                "sdxl_render": Path(
                    hf_hub_download(
                        repo_id="philz1337x/loras",
                        filename="SDXLrender_v2.0.safetensors",
                        revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
                    )
                ),
            },
        )
        return CHECKPOINTS

    upscaler = ESRGANUpscaler(
        checkpoints=_rescale_checkpoints(), device=device(), dtype=dtype()
    )
    upscaler.to(device=device(), dtype=dtype())
    MODELS["upscale"] = upscaler


def upscale(
    path,
    upscale_factor: int = 2,
    prompt: str = general_positive_prompt,
    negative_prompt: str = general_negative_prompt,
    seed: int = None,
    controlnet_scale: float = 0.8,
    controlnet_decay: float = 0.8,
    condition_scale: float = 6.5,
    tile_width: int = 128,
    tile_height: int = 192,
    denoise_strength: float = 0.1,
    num_inference_steps: int = 100,
    solver: str = "DPMSolver",
):
    from PIL import Image
    from refiners.fluxion.utils import manual_seed
    from refiners.foundationals.latent_diffusion import Solver, solvers

    import definers as _d

    if upscale_factor < 2 or upscale_factor > 4:
        return
    if not seed:
        seed = random.randint(0, 2**32 - 1)
    manual_seed(seed)
    solver_type: type[Solver] = getattr(solvers, solver)
    with Image.open(path) as opened_image:
        input_image = opened_image.copy()
    upscaled_image = MODELS["upscale"].upscale(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        upscale_factor=upscale_factor,
        controlnet_scale=controlnet_scale,
        controlnet_scale_decay=controlnet_decay,
        condition_scale=condition_scale,
        tile_size=(tile_height, tile_width),
        denoise_strength=denoise_strength,
        num_inference_steps=num_inference_steps,
        loras_scale={"more_details": 0.0, "sdxl_render": 0.0},
        solver_type=solver_type,
    )
    return _d.save_image(upscaled_image)


def get_max_resolution(width, height, mega_pixels=0.25, factor=16):
    max_pixels = mega_pixels * 1000 * 1000
    ratio = width / height
    best_candidate = None
    best_error = float("inf")
    best_pixels = -1
    max_h = int((max_pixels / max(ratio, 1e-9)) ** 0.5) + factor
    for h_factored in range(factor, max_h + factor, factor):
        w_estimate = int(h_factored * ratio)
        w_factored = max(factor, (w_estimate // factor) * factor)
        if w_factored * h_factored > max_pixels:
            w_factored -= factor
        if w_factored < factor:
            continue
        current_pixels = w_factored * h_factored
        if current_pixels <= 0 or current_pixels > max_pixels:
            continue
        current_error = abs(ratio - (w_factored / h_factored))
        if (current_error < best_error) or (
            abs(current_error - best_error) < 1e-12
            and current_pixels > best_pixels
        ):
            best_error = current_error
            best_pixels = current_pixels
            best_candidate = (w_factored, h_factored)
    if best_candidate is None:
        best_candidate = (factor, factor)
    (w_factored, h_factored) = best_candidate
    ratio_error_factored = abs(ratio - (w_factored / h_factored))

    h_exact = int((max_pixels / ratio) ** 0.5)
    w_exact = int(h_exact * ratio)
    if w_exact * h_exact > max_pixels:
        h_exact = max(1, h_exact - 1)
        w_exact = max(1, int(h_exact * ratio))
    ratio_error_exact = abs(ratio - (w_exact / h_exact))
    if ratio_error_factored <= 0.005:
        return (w_factored, h_factored)
    allow_exact_fallback = mega_pixels >= 0.5 and factor == 16
    if (
        allow_exact_fallback
        and w_exact * h_exact <= max_pixels
        and ratio_error_exact < ratio_error_factored
    ):
        return (w_exact, h_exact)
    return (w_factored, h_factored)


def save_image(img, path="."):
    import definers as _d

    name = os.path.join(path, "img_" + _d.random_string() + ".png")
    img.save(name)
    return name


def resize_image(image_path, target_width, target_height, anti_aliasing=True):
    import imageio as iio
    from PIL import Image
    from skimage.transform import resize

    image_data = iio.imread(image_path)
    try:
        if image_data.ndim < 2:
            raise ValueError(
                "Input image must have at least 2 dimensions (height, width)."
            )
        resized_image = resize(
            image_data,
            (target_height, target_width),
            anti_aliasing=anti_aliasing,
        )
        img = (resized_image * 255).astype(np.uint8)
        img = Image.fromarray(img)
        pth = save_image(img, tmp("png", keep=False))
        return (pth, img)
    except ValueError as ve:
        from definers.system import catch

        catch(ve)
        return None
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None


def image_resolution(image_path):
    from PIL import Image

    return Image.open(image_path).size
