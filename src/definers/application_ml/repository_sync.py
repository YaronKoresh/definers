_runtime_os = __import__("os")
_runtime_os.environ.setdefault("USE_TORCH", "1")
_runtime_os.environ.setdefault("USE_TF", "0")


class RepositorySyncService:
    SUPPORTED_MODEL_TYPES = (
        "bin",
        "joblib",
        "onnx",
        "pkl",
        "pt",
        "pth",
        "safetensors",
    )
    HUGGINGFACE_HOSTS = {"huggingface.co", "www.huggingface.co"}
    HUGGINGFACE_ROUTE_SEGMENTS = {"blob", "raw", "resolve", "tree"}
    MODEL_INDEX_SUFFIXES = {
        "bin": ".bin.index.json",
        "safetensors": ".safetensors.index.json",
    }
    REMOTE_PROBE_TIMEOUT_SECONDS = 10
    SHARD_NAME_PATTERN = __import__("re").compile(
        r"^(?P<prefix>.*?)(?P<index>\d+)(?P<suffix>.*?)(?P<extension>(?:\.[^.]+)+)$"
    )
    PREFERRED_MODEL_BASENAMES = (
        "model",
        "adapter_model",
        "diffusion_pytorch_model",
        "pytorch_model",
    )
    TOKENIZER_METADATA_FILES = (
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    )
    TEXT_GENERATION_ALLOW_PATTERNS = (
        "*.bin",
        "*.json",
        "*.model",
        "*.py",
        "*.safetensors",
        "*.vocab",
    )

    class ResolvedModelSource:
        def __init__(
            self,
            local_path: str,
            model_type: str,
            shard_paths=(),
            trusted_directories=(),
            loader_kind: str = "file",
        ):
            self.local_path = local_path
            self.model_type = model_type
            self.shard_paths = tuple(shard_paths)
            self.trusted_directories = tuple(trusted_directories)
            self.loader_kind = loader_kind

    class HuggingFaceReference:
        def __init__(
            self,
            repo_id: str,
            revision: str | None = None,
            file_path: str | None = None,
        ):
            self.repo_id = repo_id
            self.revision = revision
            self.file_path = file_path

    class ShardNameParts:
        def __init__(
            self,
            prefix: str,
            index: int,
            width: int,
            suffix: str,
            extension: str,
        ):
            self.prefix = prefix
            self.index = index
            self.width = width
            self.suffix = suffix
            self.extension = extension

    class TokenizerProcessorAdapter:
        def __init__(self, tokenizer):
            self._tokenizer = tokenizer

        @property
        def tokenizer(self):
            return self._tokenizer

        def __call__(
            self,
            *,
            text: str,
            images=None,
            audios=None,
            return_tensors: str = "pt",
        ):
            del images
            del audios
            return self._tokenizer(text, return_tensors=return_tensors)

        def batch_decode(
            self,
            output_ids,
            *,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = False,
        ):
            return self._tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

    class TextGenerationModelAdapter:
        def __init__(self, model, processor):
            self._model = model
            self._processor = processor

        def __getattr__(self, name: str):
            return getattr(self._model, name)

        def generate(self, *args, **kwargs):
            if args or "prompt" not in kwargs:
                return self._model.generate(*args, **kwargs)
            prompt = kwargs.pop("prompt")
            beam_width = kwargs.pop("beam_width", 1)
            kwargs.pop("images", None)
            kwargs.pop("audios", None)
            if "max_new_tokens" not in kwargs and "max_length" not in kwargs:
                kwargs["max_new_tokens"] = 200
            inputs = _tokenize_generation_prompt(self._processor, prompt)
            if hasattr(inputs, "to"):
                inputs = inputs.to(device())
            generated = self._model.generate(
                **inputs,
                num_beams=max(1, beam_width),
                **kwargs,
            )
            input_ids = inputs["input_ids"]
            output_ids = generated[:, input_ids.shape[1] :]
            decoded = self._processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0] if decoded else ""

    @staticmethod
    def constant_registries():
        from definers.constants import MODELS, PROCESSORS, TOKENIZERS, tasks

        return MODELS, PROCESSORS, TOKENIZERS, tasks

    @staticmethod
    def cuda_functions():
        from definers.cuda import device, free

        return device, free

    @staticmethod
    def system_functions():
        from definers.system import get_ext, tmp

        return get_ext, tmp

    @staticmethod
    def logger_instance():
        from definers.logger import init_logger

        return init_logger()

    @staticmethod
    def download_function():
        from definers.web import download_file

        return download_file

    @staticmethod
    def init_model_file(
        task: str, turbo: bool = True, model_type: str | None = None
    ):
        from definers.system import secure_path

        free()
        model_path = str(tasks.get(task, task)).strip()
        try:
            resolved_source = _resolve_model_source(model_path, model_type)
            trusted_directories = (
                resolved_source.trusted_directories
                if resolved_source.trusted_directories
                else None
            )
            local_model_path = secure_path(
                resolved_source.local_path,
                trust=trusted_directories,
            )
            shard_paths = tuple(
                secure_path(shard_path, trust=trusted_directories)
                for shard_path in resolved_source.shard_paths
            )
            model = _load_model(
                local_model_path,
                resolved_source.model_type,
                shard_paths=shard_paths,
                loader_kind=resolved_source.loader_kind,
                task_key=task,
            )
            if model is None:
                raise RuntimeError(
                    f"Model loader returned None for task '{task}' from '{model_path}'"
                )
            if turbo:
                _apply_turbo_optimizations(model)
            logger.info("✅ Model loaded successfully.")
            MODELS[task] = model
            return model
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Model source not found for task '{task}': {model_path}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"Definers failed to load task '{task}' from '{model_path}': {error}"
            ) from error
        finally:
            free()

    @staticmethod
    def load_model(
        model_path: str,
        model_type: str,
        *,
        shard_paths=(),
        loader_kind: str = "file",
        task_key: str | None = None,
    ):
        from definers.application_ml.safe_deserialization import (
            load_serialized_model,
        )

        if loader_kind == "hf-text-generation":
            return _load_huggingface_text_generation_model(
                model_path,
                task_key=task_key,
            )
        if model_type not in SUPPORTED_MODEL_TYPES:
            logger.error(
                f'Error: Model type "{model_type}" is not supported. Must be one of {list(SUPPORTED_MODEL_TYPES)}'
            )
            return None
        load_paths = shard_paths or (model_path,)
        logger.info(
            f"Attempting to load a {model_type.upper()} model from: {model_path}"
        )
        if len(load_paths) > 1:
            logger.info(f"Detected {len(load_paths)} model shards.")
        if model_type == "joblib":
            return load_serialized_model(model_path, model_type)
        if model_type == "onnx":
            import onnxruntime

            return onnxruntime.InferenceSession(model_path)
        if model_type == "pkl":
            return load_serialized_model(model_path, model_type)
        if model_type in {"bin", "pt", "pth"}:
            import torch

            if len(load_paths) == 1:
                model = torch.load(
                    model_path,
                    map_location=device(),
                    weights_only=True,
                )
            else:
                model = _merge_mapping_parts(
                    torch.load(
                        shard_path,
                        map_location=device(),
                        weights_only=True,
                    )
                    for shard_path in load_paths
                )
        else:
            from safetensors.torch import load_file

            if len(load_paths) == 1:
                model = load_file(model_path, device=device())
            else:
                model = _merge_mapping_parts(
                    load_file(shard_path, device=device())
                    for shard_path in load_paths
                )
        if hasattr(model, "eval"):
            model.eval()
            logger.info("Model set to evaluation mode.")
        return model

    @staticmethod
    def resolve_model_source(
        model_reference: str, requested_model_type: str | None
    ):
        normalized_reference = str(model_reference).strip()
        normalized_model_type = _normalize_model_type(requested_model_type)
        if _is_huggingface_reference(normalized_reference):
            return _resolve_huggingface_model_source(
                normalized_reference,
                normalized_model_type,
            )
        if _is_http_url(normalized_reference):
            return _resolve_generic_remote_source(
                normalized_reference,
                normalized_model_type,
            )
        local_model_type = _resolve_model_type(
            normalized_reference,
            normalized_model_type,
        )
        return ResolvedModelSource(
            local_path=normalized_reference,
            model_type=local_model_type,
        )

    @staticmethod
    def resolve_huggingface_model_source(
        value: str, requested_model_type: str | None
    ):
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

        reference = _parse_huggingface_reference(value)
        repo_files = tuple(
            sorted(
                HfApi().list_repo_files(
                    reference.repo_id,
                    revision=reference.revision,
                )
            )
        )
        selected_files = _select_huggingface_files(
            reference,
            repo_files,
            reference.file_path,
            requested_model_type,
        )
        if not selected_files:
            raise FileNotFoundError(
                f"No compatible model files found in Hugging Face repository '{reference.repo_id}'."
            )
        if _is_transformers_text_generation_repo(repo_files, selected_files):
            snapshot_dir = snapshot_download(
                repo_id=reference.repo_id,
                revision=reference.revision,
                allow_patterns=list(TEXT_GENERATION_ALLOW_PATTERNS),
            )
            return ResolvedModelSource(
                local_path=snapshot_dir,
                model_type=_resolve_model_type(
                    selected_files[0],
                    requested_model_type,
                ),
                trusted_directories=_trusted_directories_for_paths(
                    (snapshot_dir,)
                ),
                loader_kind="hf-text-generation",
            )
        local_files = []
        for repo_file in selected_files:
            local_files.append(
                hf_hub_download(
                    repo_id=reference.repo_id,
                    filename=repo_file,
                    revision=reference.revision,
                )
            )
        resolved_model_type = _resolve_model_type(
            selected_files[0],
            requested_model_type,
        )
        if len(local_files) == 1:
            return ResolvedModelSource(
                local_path=local_files[0],
                model_type=resolved_model_type,
                trusted_directories=_trusted_directories_for_paths(local_files),
            )
        return ResolvedModelSource(
            local_path=local_files[0],
            model_type=resolved_model_type,
            shard_paths=tuple(local_files),
            trusted_directories=_trusted_directories_for_paths(local_files),
        )

    @staticmethod
    def resolve_generic_remote_source(
        url: str, requested_model_type: str | None
    ):
        from pathlib import Path

        remote_urls = _discover_remote_shard_urls(url)
        resolved_model_type = _resolve_model_type(url, requested_model_type)
        if len(remote_urls) == 1:
            target_path = tmp(resolved_model_type, keep=False)
            downloaded_path = download_file(remote_urls[0], target_path)
            if downloaded_path is None:
                raise FileNotFoundError(
                    f"Could not download model from '{remote_urls[0]}'."
                )
            _validate_downloaded_model_file(
                downloaded_path, resolved_model_type
            )
            return ResolvedModelSource(
                local_path=downloaded_path,
                model_type=resolved_model_type,
                trusted_directories=_trusted_directories_for_paths(
                    (downloaded_path,)
                ),
            )
        temp_directory = Path(tmp(dir=True))
        local_files = []
        for remote_url in remote_urls:
            target_path = str(temp_directory / _remote_file_name(remote_url))
            downloaded_path = download_file(remote_url, target_path)
            if downloaded_path is None:
                raise FileNotFoundError(
                    f"Could not download model shard from '{remote_url}'."
                )
            _validate_downloaded_model_file(
                downloaded_path, resolved_model_type
            )
            local_files.append(downloaded_path)
        return ResolvedModelSource(
            local_path=local_files[0],
            model_type=resolved_model_type,
            shard_paths=tuple(local_files),
            trusted_directories=_trusted_directories_for_paths(local_files),
        )

    @staticmethod
    def parse_huggingface_reference(value: str):
        from urllib.parse import urlparse

        normalized_value = str(value).strip()
        if not _is_http_url(normalized_value):
            return HuggingFaceReference(repo_id=normalized_value)
        parsed = urlparse(normalized_value)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(
                f"Invalid Hugging Face reference: '{normalized_value}'"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        revision = None
        file_path = None
        if len(parts) >= 4 and parts[2] in HUGGINGFACE_ROUTE_SEGMENTS:
            revision = parts[3]
            remaining_parts = parts[4:]
            if remaining_parts:
                file_path = "/".join(remaining_parts)
        return HuggingFaceReference(
            repo_id=repo_id,
            revision=revision,
            file_path=file_path,
        )

    @staticmethod
    def is_huggingface_reference(value: str) -> bool:
        from urllib.parse import urlparse

        normalized_value = str(value).strip()
        if _is_http_url(normalized_value):
            return (
                urlparse(normalized_value).netloc.lower() in HUGGINGFACE_HOSTS
            )
        return _is_huggingface_repo_id(normalized_value)

    @staticmethod
    def is_huggingface_repo_id(value: str) -> bool:
        if not isinstance(value, str) or not value or "/" not in value:
            return False
        user, name = value.split("/", 1)
        if not user or not name:
            return False
        allowed_characters = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        return all(
            character in allowed_characters for character in user
        ) and all(character in allowed_characters for character in name)

    @staticmethod
    def select_huggingface_files(
        reference, repo_files, requested_file, requested_model_type
    ):
        if requested_file:
            exact_match = _match_requested_repo_file(
                reference,
                repo_files,
                requested_file,
                requested_model_type,
            )
            if exact_match:
                return exact_match
        candidates = _supported_repo_files(repo_files, requested_model_type)
        if not candidates:
            return ()
        indexed_candidates = [
            candidate for candidate in candidates if _is_index_file(candidate)
        ]
        if indexed_candidates:
            return _expand_indexed_repo_file(
                reference,
                repo_files,
                indexed_candidates[0],
            )
        shard_group = _largest_repo_shard_group(candidates)
        if shard_group:
            return shard_group
        ranked_candidates = sorted(candidates, key=_repo_file_priority)
        return (ranked_candidates[0],)

    @staticmethod
    def match_requested_repo_file(
        reference, repo_files, requested_file, requested_model_type
    ):
        normalized_request = requested_file.strip("/")
        if normalized_request in repo_files:
            if _is_index_file(normalized_request):
                return _expand_indexed_repo_file(
                    reference,
                    repo_files,
                    normalized_request,
                )
            companion_index = _companion_index_path(normalized_request)
            if companion_index and companion_index in repo_files:
                return _expand_indexed_repo_file(
                    reference,
                    repo_files,
                    companion_index,
                )
            shard_group = _find_repo_shard_group(repo_files, normalized_request)
            if shard_group:
                return shard_group
            return (normalized_request,)
        matching_candidates = [
            repo_file
            for repo_file in repo_files
            if _repo_file_matches_hint(repo_file, normalized_request)
        ]
        matching_candidates = _supported_repo_files(
            tuple(matching_candidates),
            requested_model_type,
        )
        if not matching_candidates:
            return ()
        shard_group = _largest_repo_shard_group(matching_candidates)
        if shard_group:
            return shard_group
        return (sorted(matching_candidates, key=_repo_file_priority)[0],)

    @staticmethod
    def supported_repo_files(repo_files, requested_model_type):
        supported_files = []
        for repo_file in repo_files:
            try:
                repo_model_type = _resolve_model_type(
                    repo_file, requested_model_type
                )
            except ValueError:
                continue
            if requested_model_type and repo_model_type != requested_model_type:
                continue
            supported_files.append(repo_file)
        return supported_files

    @staticmethod
    def expand_indexed_repo_file(reference, repo_files, index_file):
        try:
            shard_files = _read_index_shard_files(reference, index_file)
        except Exception:
            shard_files = ()
        if shard_files:
            available_files = [
                shard_file
                for shard_file in shard_files
                if shard_file in repo_files
            ]
            if available_files:
                return tuple(available_files)
        shard_group = _find_repo_shard_group(
            repo_files,
            index_file.replace(".index.json", ""),
        )
        if shard_group:
            return shard_group
        raise FileNotFoundError(
            f"No shards found for index file '{index_file}'."
        )

    @staticmethod
    def read_index_shard_files(reference, index_file):
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        local_index_path = hf_hub_download(
            repo_id=reference.repo_id,
            filename=index_file,
            revision=reference.revision,
        )
        with open(local_index_path, encoding="utf-8") as file_obj:
            index_data = json.load(file_obj)
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return ()
        parent_directory = str(Path(index_file).parent)
        normalized_parent = "" if parent_directory == "." else parent_directory
        shard_files = []
        for shard_name in weight_map.values():
            shard_path = str(Path(normalized_parent) / shard_name).replace(
                "\\", "/"
            )
            shard_files.append(shard_path.lstrip("./"))
        return tuple(sorted(dict.fromkeys(shard_files)))

    @staticmethod
    def find_repo_shard_group(repo_files, requested_file):
        groups = _group_repo_shards(repo_files)
        normalized_request = requested_file.strip("/")
        for group in groups:
            if normalized_request in group:
                return group
            for candidate in group:
                if _repo_file_matches_hint(candidate, normalized_request):
                    return group
        return ()

    @staticmethod
    def largest_repo_shard_group(repo_files):
        groups = _group_repo_shards(tuple(repo_files))
        if not groups:
            return ()
        return max(groups, key=lambda group: (len(group), group[0]))

    @staticmethod
    def group_repo_shards(repo_files):
        from pathlib import Path

        grouped_files = {}
        for repo_file in repo_files:
            parsed_name = _parse_shard_name(Path(repo_file).name)
            if parsed_name is None:
                continue
            group_key = (
                str(Path(repo_file).parent),
                parsed_name.prefix,
                parsed_name.suffix,
                parsed_name.extension,
            )
            grouped_files.setdefault(group_key, []).append(
                (parsed_name.index, repo_file)
            )
        groups = []
        for indexed_files in grouped_files.values():
            if len(indexed_files) < 2:
                continue
            ordered_files = tuple(
                repo_file
                for _, repo_file in sorted(
                    indexed_files, key=lambda item: item[0]
                )
            )
            groups.append(ordered_files)
        return groups

    @staticmethod
    def discover_remote_shard_urls(url: str):
        import re

        shard_name = _parse_shard_name(_remote_file_name(url))
        if shard_name is None:
            return (url,)
        if "-of-" in shard_name.suffix:
            total_match = re.search(r"-of-(\d+)", shard_name.suffix)
            if total_match is not None:
                total_shards = int(total_match.group(1))
                return tuple(
                    _replace_remote_file_name(
                        url, _format_shard_name(shard_name, index)
                    )
                    for index in range(1, total_shards + 1)
                )
        discovered_urls = {shard_name.index: url}
        lower_index = shard_name.index - 1
        while lower_index >= 0:
            candidate_url = _replace_remote_file_name(
                url,
                _format_shard_name(shard_name, lower_index),
            )
            if not _remote_file_exists(candidate_url):
                break
            discovered_urls[lower_index] = candidate_url
            lower_index -= 1
        upper_index = shard_name.index + 1
        while upper_index - shard_name.index <= 256:
            candidate_url = _replace_remote_file_name(
                url,
                _format_shard_name(shard_name, upper_index),
            )
            if not _remote_file_exists(candidate_url):
                break
            discovered_urls[upper_index] = candidate_url
            upper_index += 1
        return tuple(
            url_value for _, url_value in sorted(discovered_urls.items())
        )

    @staticmethod
    def remote_file_exists(url: str) -> bool:
        import requests

        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=REMOTE_PROBE_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return True
            if response.status_code not in {403, 405}:
                return False
            fallback_response = requests.get(
                url,
                allow_redirects=True,
                stream=True,
                timeout=REMOTE_PROBE_TIMEOUT_SECONDS,
            )
            fallback_response.close()
            return fallback_response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def remote_file_name(url: str) -> str:
        from pathlib import Path
        from urllib.parse import urlparse

        path_name = Path(urlparse(url).path).name
        if path_name:
            return path_name
        raise ValueError(f"Could not determine remote file name for '{url}'.")

    @staticmethod
    def replace_remote_file_name(url: str, new_file_name: str) -> str:
        from pathlib import Path
        from urllib.parse import urlparse

        parsed = urlparse(url)
        updated_path = str(Path(parsed.path).parent / new_file_name).replace(
            "\\", "/"
        )
        return parsed._replace(path=updated_path).geturl()

    @staticmethod
    def parse_shard_name(file_name: str):
        match = SHARD_NAME_PATTERN.match(file_name)
        if match is None:
            return None
        index_text = match.group("index")
        prefix = match.group("prefix")
        suffix = match.group("suffix")
        extension = match.group("extension").lstrip(".").lower()
        if extension not in SUPPORTED_MODEL_TYPES:
            return None
        if not prefix and not suffix:
            return None
        return ShardNameParts(
            prefix=prefix,
            index=int(index_text),
            width=len(index_text),
            suffix=suffix,
            extension=extension,
        )

    @staticmethod
    def format_shard_name(shard_name, index: int) -> str:
        return (
            f"{shard_name.prefix}{index:0{shard_name.width}d}"
            f"{shard_name.suffix}.{shard_name.extension}"
        )

    @staticmethod
    def resolve_model_type(value: str, requested_model_type: str | None):
        normalized_model_type = _normalize_model_type(requested_model_type)
        if normalized_model_type is not None:
            return normalized_model_type
        normalized_value = str(value).strip().lower()
        for model_type, suffix in MODEL_INDEX_SUFFIXES.items():
            if normalized_value.endswith(suffix):
                return model_type
        try:
            resolved_extension = get_ext(value).lower()
        except Exception as error:
            raise ValueError(
                f"Could not determine model type for '{value}'."
            ) from error
        normalized_extension = _normalize_model_type(resolved_extension)
        if normalized_extension is None:
            raise ValueError(f"Unsupported model type: '{resolved_extension}'.")
        return normalized_extension

    @staticmethod
    def normalize_model_type(model_type: str | None):
        if model_type is None:
            return None
        normalized_model_type = str(model_type).strip().lower().strip(".")
        if normalized_model_type in SUPPORTED_MODEL_TYPES:
            return normalized_model_type
        return None

    @staticmethod
    def is_http_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    @staticmethod
    def is_index_file(path: str) -> bool:
        normalized_path = str(path).strip().lower()
        return any(
            normalized_path.endswith(index_suffix)
            for index_suffix in MODEL_INDEX_SUFFIXES.values()
        )

    @staticmethod
    def companion_index_path(path: str):
        from pathlib import Path

        normalized_path = str(path).strip()
        normalized_model_type = _normalize_model_type(
            Path(normalized_path).suffix
        )
        if normalized_model_type is None:
            return None
        index_suffix = MODEL_INDEX_SUFFIXES.get(normalized_model_type)
        if index_suffix is None:
            return None
        return f"{normalized_path}{index_suffix[len('.' + normalized_model_type) :]}"

    @staticmethod
    def repo_file_matches_hint(repo_file: str, hint: str) -> bool:
        from pathlib import Path

        normalized_hint = Path(hint).name
        normalized_file_name = Path(repo_file).name
        if not normalized_hint:
            return False
        if normalized_file_name == normalized_hint:
            return True
        file_stem = Path(normalized_file_name).stem
        hint_stem = Path(normalized_hint).stem
        return file_stem.startswith(hint_stem) or file_stem.endswith(hint_stem)

    @staticmethod
    def repo_file_priority(repo_file: str):
        from pathlib import Path

        base_name = Path(repo_file).stem
        try:
            preferred_rank = PREFERRED_MODEL_BASENAMES.index(base_name)
        except ValueError:
            preferred_rank = len(PREFERRED_MODEL_BASENAMES)
        return preferred_rank, len(repo_file), repo_file

    @staticmethod
    def is_transformers_text_generation_repo(
        repo_files, selected_files
    ) -> bool:
        from pathlib import Path

        repo_file_names = {Path(repo_file).name for repo_file in repo_files}
        if "config.json" not in repo_file_names:
            return False
        if "generation_config.json" not in repo_file_names:
            return False
        if not any(
            tokenizer_file in repo_file_names
            for tokenizer_file in TOKENIZER_METADATA_FILES
        ):
            return False
        return any(
            Path(selected_file).name.startswith(PREFERRED_MODEL_BASENAMES)
            or Path(selected_file).name.startswith("model-")
            or Path(selected_file).name.startswith("pytorch_model-")
            for selected_file in selected_files
        )

    @staticmethod
    def read_json_if_exists(path):
        import json

        if not path.exists() or not path.is_file():
            return {}
        try:
            with open(path, encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def should_try_auto_processor(snapshot_dir: str) -> bool:
        from pathlib import Path

        snapshot_path = Path(snapshot_dir)
        for file_name in (
            "processor_config.json",
            "preprocessor_config.json",
            "feature_extractor_config.json",
            "image_processor_config.json",
            "image_processing_config.json",
        ):
            if (snapshot_path / file_name).exists():
                return True
        config = _read_json_if_exists(snapshot_path / "config.json")
        if not config:
            return False
        for key in (
            "processor_class",
            "feature_extractor_type",
            "image_processor_type",
        ):
            value = config.get(key)
            if isinstance(value, str) and value.strip():
                return True
        auto_map = config.get("auto_map")
        if isinstance(auto_map, dict):
            for key in (
                "AutoProcessor",
                "AutoImageProcessor",
                "AutoFeatureExtractor",
            ):
                if key in auto_map:
                    return True
        for key in (
            "vision_config",
            "audio_config",
            "image_token_index",
            "video_token_index",
            "vision_feature_layer",
            "vision_feature_select_strategy",
            "image_seq_length",
            "audio_seq_length",
        ):
            if key in config:
                return True
        architectures = config.get("architectures")
        if isinstance(architectures, list):
            architecture_text = " ".join(
                value for value in architectures if isinstance(value, str)
            ).lower()
            if any(
                token in architecture_text
                for token in (
                    "vision",
                    "audio",
                    "vl",
                    "multimodal",
                    "multi_modal",
                )
            ):
                return True
        return False

    @staticmethod
    def load_huggingface_text_generation_model(
        snapshot_dir: str, *, task_key: str | None
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = None
        processor = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot_dir,
                trust_remote_code=True,
            )
        except Exception as error:
            logger.warning(
                "Tokenizer load failed for '%s': %s",
                snapshot_dir,
                error,
            )
            tokenizer = None
        try:
            from transformers import AutoProcessor
        except Exception as error:
            logger.warning(
                "AutoProcessor import failed for '%s': %s",
                snapshot_dir,
                error,
            )
        else:
            try:
                processor = AutoProcessor.from_pretrained(
                    snapshot_dir,
                    trust_remote_code=True,
                )
            except Exception as error:
                logger.warning(
                    "Processor load failed for '%s': %s",
                    snapshot_dir,
                    error,
                )
                processor = None
        if processor is not None and not hasattr(processor, "tokenizer"):
            tokenizer = tokenizer or processor
            processor = TokenizerProcessorAdapter(tokenizer)
        elif processor is None and tokenizer is not None:
            processor = TokenizerProcessorAdapter(tokenizer)
        if (
            tokenizer is None
            and processor is not None
            and hasattr(processor, "tokenizer")
        ):
            tokenizer = processor.tokenizer
        try:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_dir,
                trust_remote_code=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"AutoModelForCausalLM.from_pretrained failed for '{snapshot_dir}': {error}"
            ) from error
        if hasattr(model, "to"):
            model = model.to(device())
        if task_key is not None:
            if processor is not None:
                PROCESSORS[task_key] = processor
            if tokenizer is not None:
                TOKENIZERS[task_key] = tokenizer
        if processor is None:
            return model
        return TextGenerationModelAdapter(model, processor)

    @staticmethod
    def tokenize_generation_prompt(processor, prompt: str):
        try:
            return processor(
                text=prompt,
                images=None,
                audios=None,
                return_tensors="pt",
            )
        except TypeError:
            return processor(text=prompt, return_tensors="pt")

    @staticmethod
    def merge_mapping_parts(parts):
        merged_parts = {}
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError(
                    "Split model loading requires each shard to deserialize into a mapping."
                )
            overlapping_keys = merged_parts.keys() & part.keys()
            if overlapping_keys:
                duplicate_key = next(iter(overlapping_keys))
                raise ValueError(
                    f"Duplicate parameter '{duplicate_key}' detected while merging shards."
                )
            merged_parts.update(part)
        return merged_parts

    @staticmethod
    def trusted_directories_for_paths(paths):
        import os
        from pathlib import Path

        normalized_paths = [str(Path(path).resolve()) for path in paths if path]
        if not normalized_paths:
            return ()
        parent_directories = [
            str(Path(path).resolve().parent) for path in normalized_paths
        ]
        common_directory = os.path.commonpath(parent_directories)
        trusted_directories = {common_directory}
        trusted_directories.update(parent_directories)
        return tuple(sorted(trusted_directories))

    @staticmethod
    def validate_downloaded_model_file(
        model_path: str, model_type: str
    ) -> None:
        from definers.application_ml.safe_deserialization import (
            validate_serialized_model_file,
        )

        if model_type in {"joblib", "pkl"}:
            validate_serialized_model_file(model_path, model_type)
            return
        if model_type not in {"bin", "onnx", "pt", "pth", "safetensors"}:
            return
        with open(model_path, "rb") as file_obj:
            header = file_obj.read(512)
        lowered_header = header.lower()
        if lowered_header.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            raise ValueError(
                "Downloaded a Git LFS pointer instead of model bytes. Use a raw artifact source or a Hugging Face repo-aware source."
            )
        if lowered_header.startswith(
            b"<!doctype html"
        ) or lowered_header.startswith(b"<html"):
            raise ValueError(
                "Downloaded HTML instead of model bytes. Use a raw artifact URL or a Hugging Face repo-aware source."
            )

    @staticmethod
    def apply_turbo_optimizations(model) -> None:
        for attr_name, args in [
            ("enable_vae_slicing", ()),
            ("enable_vae_tiling", ()),
            ("enable_model_cpu_offload", ()),
            ("enable_sequential_cpu_offload", ()),
            ("enable_attention_slicing", (1,)),
        ]:
            try:
                getattr(model, attr_name)(*args)
            except AttributeError:
                continue
            except Exception as error:
                logger.warning(
                    f"Could not apply optimization {attr_name}: {error}"
                )


MODELS, PROCESSORS, TOKENIZERS, tasks = (
    RepositorySyncService.constant_registries()
)
device, free = RepositorySyncService.cuda_functions()
get_ext, tmp = RepositorySyncService.system_functions()
download_file = RepositorySyncService.download_function()
logger = RepositorySyncService.logger_instance()
SUPPORTED_MODEL_TYPES = RepositorySyncService.SUPPORTED_MODEL_TYPES
HUGGINGFACE_HOSTS = RepositorySyncService.HUGGINGFACE_HOSTS
HUGGINGFACE_ROUTE_SEGMENTS = RepositorySyncService.HUGGINGFACE_ROUTE_SEGMENTS
MODEL_INDEX_SUFFIXES = RepositorySyncService.MODEL_INDEX_SUFFIXES
REMOTE_PROBE_TIMEOUT_SECONDS = (
    RepositorySyncService.REMOTE_PROBE_TIMEOUT_SECONDS
)
SHARD_NAME_PATTERN = RepositorySyncService.SHARD_NAME_PATTERN
PREFERRED_MODEL_BASENAMES = RepositorySyncService.PREFERRED_MODEL_BASENAMES
TOKENIZER_METADATA_FILES = RepositorySyncService.TOKENIZER_METADATA_FILES
TEXT_GENERATION_ALLOW_PATTERNS = (
    RepositorySyncService.TEXT_GENERATION_ALLOW_PATTERNS
)
ResolvedModelSource = RepositorySyncService.ResolvedModelSource
HuggingFaceReference = RepositorySyncService.HuggingFaceReference
ShardNameParts = RepositorySyncService.ShardNameParts
TokenizerProcessorAdapter = RepositorySyncService.TokenizerProcessorAdapter
TextGenerationModelAdapter = RepositorySyncService.TextGenerationModelAdapter
init_model_file = RepositorySyncService.init_model_file
_load_model = RepositorySyncService.load_model
_resolve_model_source = RepositorySyncService.resolve_model_source
_resolve_huggingface_model_source = (
    RepositorySyncService.resolve_huggingface_model_source
)
_resolve_generic_remote_source = (
    RepositorySyncService.resolve_generic_remote_source
)
_parse_huggingface_reference = RepositorySyncService.parse_huggingface_reference
_is_huggingface_reference = RepositorySyncService.is_huggingface_reference
_is_huggingface_repo_id = RepositorySyncService.is_huggingface_repo_id
_select_huggingface_files = RepositorySyncService.select_huggingface_files
_match_requested_repo_file = RepositorySyncService.match_requested_repo_file
_supported_repo_files = RepositorySyncService.supported_repo_files
_expand_indexed_repo_file = RepositorySyncService.expand_indexed_repo_file
_read_index_shard_files = RepositorySyncService.read_index_shard_files
_find_repo_shard_group = RepositorySyncService.find_repo_shard_group
_largest_repo_shard_group = RepositorySyncService.largest_repo_shard_group
_group_repo_shards = RepositorySyncService.group_repo_shards
_discover_remote_shard_urls = RepositorySyncService.discover_remote_shard_urls
_remote_file_exists = RepositorySyncService.remote_file_exists
_remote_file_name = RepositorySyncService.remote_file_name
_replace_remote_file_name = RepositorySyncService.replace_remote_file_name
_parse_shard_name = RepositorySyncService.parse_shard_name
_format_shard_name = RepositorySyncService.format_shard_name
_resolve_model_type = RepositorySyncService.resolve_model_type
_normalize_model_type = RepositorySyncService.normalize_model_type
_is_http_url = RepositorySyncService.is_http_url
_is_index_file = RepositorySyncService.is_index_file
_companion_index_path = RepositorySyncService.companion_index_path
_repo_file_matches_hint = RepositorySyncService.repo_file_matches_hint
_repo_file_priority = RepositorySyncService.repo_file_priority
_is_transformers_text_generation_repo = (
    RepositorySyncService.is_transformers_text_generation_repo
)
_read_json_if_exists = RepositorySyncService.read_json_if_exists
_should_try_auto_processor = RepositorySyncService.should_try_auto_processor
_load_huggingface_text_generation_model = (
    RepositorySyncService.load_huggingface_text_generation_model
)
_tokenize_generation_prompt = RepositorySyncService.tokenize_generation_prompt
_merge_mapping_parts = RepositorySyncService.merge_mapping_parts
_trusted_directories_for_paths = (
    RepositorySyncService.trusted_directories_for_paths
)
_validate_downloaded_model_file = (
    RepositorySyncService.validate_downloaded_model_file
)
_apply_turbo_optimizations = RepositorySyncService.apply_turbo_optimizations
