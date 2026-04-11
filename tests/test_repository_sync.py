import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from definers.ml import repository_sync


class TestRepositorySyncHelpers(unittest.TestCase):
    def test_parse_huggingface_blob_reference(self):
        reference = repository_sync._parse_huggingface_reference(
            "https://huggingface.co/ibm-granite/granite-4.0-h-350m/blob/3b17b717b8f2f5d305b0a92c1491e239aeda19c8/model.safetensors"
        )

        self.assertEqual(reference.repo_id, "ibm-granite/granite-4.0-h-350m")
        self.assertEqual(
            reference.revision,
            "3b17b717b8f2f5d305b0a92c1491e239aeda19c8",
        )
        self.assertEqual(reference.file_path, "model.safetensors")

    def test_repo_file_hint_matches_suffix(self):
        self.assertTrue(
            repository_sync._repo_file_matches_hint(
                "subdir/001-my-model.safetensors",
                "my-model.safetensors",
            )
        )

    def test_find_repo_shard_group_supports_suffix(self):
        repo_files = (
            "subdir/001-my-model.safetensors",
            "subdir/002-my-model.safetensors",
            "subdir/readme.md",
        )

        self.assertEqual(
            repository_sync._find_repo_shard_group(
                repo_files,
                "my-model.safetensors",
            ),
            (
                "subdir/001-my-model.safetensors",
                "subdir/002-my-model.safetensors",
            ),
        )

    def test_discover_remote_shard_urls_supports_suffix(self):
        base_url = "https://example.com/002-my-model.safetensors"
        existing_urls = {
            "https://example.com/001-my-model.safetensors",
            "https://example.com/003-my-model.safetensors",
        }

        with patch.object(
            repository_sync,
            "_remote_file_exists",
            side_effect=lambda url: url in existing_urls,
        ):
            discovered_urls = repository_sync._discover_remote_shard_urls(
                base_url
            )

        self.assertEqual(
            discovered_urls,
            (
                "https://example.com/001-my-model.safetensors",
                "https://example.com/002-my-model.safetensors",
                "https://example.com/003-my-model.safetensors",
            ),
        )

    def test_validate_downloaded_model_file_rejects_html(self):
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(
            suffix=".safetensors"
        )
        os.close(temp_file_descriptor)
        Path(temp_file_path).write_bytes(
            b"<!DOCTYPE html><html><body>not a model</body></html>"
        )
        try:
            with self.assertRaisesRegex(ValueError, "Downloaded HTML"):
                repository_sync._validate_downloaded_model_file(
                    temp_file_path,
                    "safetensors",
                )
        finally:
            Path(temp_file_path).unlink(missing_ok=True)

    def test_validate_downloaded_model_file_secures_path_before_opening(self):
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(
            suffix=".safetensors"
        )
        os.close(temp_file_descriptor)
        Path(temp_file_path).write_bytes(b"safe-bytes")
        try:
            with patch(
                "definers.system.secure_path",
                side_effect=lambda path, trust=None: path,
            ) as mock_secure_path:
                repository_sync._validate_downloaded_model_file(
                    temp_file_path,
                    "safetensors",
                )

            mock_secure_path.assert_called_once()
            self.assertEqual(mock_secure_path.call_args.args[0], temp_file_path)
        finally:
            Path(temp_file_path).unlink(missing_ok=True)

    def test_trusted_directories_include_common_parent(self):
        trusted_directories = repository_sync._trusted_directories_for_paths(
            (
                "C:/Users/User/.cache/huggingface/hub/model-a/file1.safetensors",
                "C:/Users/User/.cache/huggingface/hub/model-a/file2.safetensors",
            )
        )

        self.assertIn(
            "C:\\Users\\User\\.cache\\huggingface\\hub\\model-a",
            trusted_directories,
        )

    def test_trusted_directories_ignore_relative_paths(self):
        trusted_directories = repository_sync._trusted_directories_for_paths(
            (
                "relative/model/file1.safetensors",
                "C:/Users/User/.cache/huggingface/hub/model-a/file2.safetensors",
            )
        )

        self.assertEqual(
            trusted_directories,
            ("C:\\Users\\User\\.cache\\huggingface\\hub\\model-a",),
        )

    def test_transformers_text_generation_repo_detection(self):
        repo_files = (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        )

        self.assertTrue(
            repository_sync._is_transformers_text_generation_repo(
                repo_files,
                ("model.safetensors",),
            )
        )

    def test_read_index_shard_files_uses_fast_file_download(self):
        reference = repository_sync.HuggingFaceReference(
            "owner/repo",
            revision="rev-1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = Path(temp_dir) / "model.index.json"
            local_index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0": "0001-of-0002.safetensors",
                            "layer.1": "0002-of-0002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "definers.model_installation.hf_file_download",
                return_value=str(local_index_path),
            ) as mock_download:
                shard_files = repository_sync._read_index_shard_files(
                    reference,
                    "subdir/model.safetensors.index.json",
                )

        self.assertEqual(
            shard_files,
            (
                "subdir/0001-of-0002.safetensors",
                "subdir/0002-of-0002.safetensors",
            ),
        )
        mock_download.assert_called_once_with(
            repo_id="owner/repo",
            filename="subdir/model.safetensors.index.json",
            revision="rev-1",
            item_label="subdir/model.safetensors.index.json",
            detail="Downloading shard index from owner/repo.",
        )


class TestRepositorySyncHuggingFaceRouting(unittest.TestCase):
    def test_init_model_file_uses_model_installation_file_download_for_blob_url(
        self,
    ):
        blob_url = (
            "https://huggingface.co/ibm-granite/granite-4.0-h-350m/blob/"
            "3b17b717b8f2f5d305b0a92c1491e239aeda19c8/model.safetensors"
        )
        fake_download = MagicMock(return_value="C:/tmp/model.safetensors")

        secure_path_calls = []

        def fake_secure_path(value, trust=None, **kwargs):
            secure_path_calls.append((value, trust, kwargs))
            return value

        with (
            patch(
                "definers.model_installation._huggingface_repo_files",
                return_value=("model.safetensors",),
            ),
            patch(
                "definers.model_installation.hf_file_download",
                fake_download,
            ),
            patch(
                "definers.model_installation.hf_snapshot_download"
            ) as mock_snapshot,
            patch.object(
                repository_sync, "_load_model", return_value={}
            ) as mock_load,
            patch.object(
                repository_sync, "_apply_turbo_optimizations"
            ) as mock_turbo,
            patch.object(repository_sync, "download_file") as mock_download,
            patch.object(repository_sync, "free"),
            patch("definers.system.secure_path", side_effect=fake_secure_path),
            patch.dict(repository_sync.MODELS, {}, clear=True),
        ):
            repository_sync.init_model_file(blob_url, turbo=False)

        mock_download.assert_not_called()
        mock_turbo.assert_not_called()
        mock_snapshot.assert_not_called()
        fake_download.assert_called_once_with(
            repo_id="ibm-granite/granite-4.0-h-350m",
            filename="model.safetensors",
            revision="3b17b717b8f2f5d305b0a92c1491e239aeda19c8",
            item_label="model.safetensors",
            detail="Downloading file from ibm-granite/granite-4.0-h-350m.",
            completed=1,
            total=1,
        )
        mock_load.assert_called_once_with(
            "C:/tmp/model.safetensors",
            "safetensors",
            shard_paths=(),
            loader_kind="file",
            task_key=blob_url,
            trusted_directories=("C:\\tmp",),
        )
        self.assertEqual(len(secure_path_calls), 1)
        self.assertEqual(secure_path_calls[0][0], "C:/tmp/model.safetensors")
        self.assertEqual(
            secure_path_calls[0][1],
            ("C:\\tmp",),
        )

    def test_init_model_file_trusts_huggingface_cache_directory(self):
        blob_url = (
            "https://huggingface.co/ibm-granite/granite-4.0-h-350m/blob/"
            "3b17b717b8f2f5d305b0a92c1491e239aeda19c8/model.safetensors"
        )
        cached_path = (
            "C:/Users/User/.cache/huggingface/hub/models--ibm-granite--"
            "granite-4.0-h-350m/snapshots/3b17b717b8f2f5d305b0a92c1491e239aeda19c8/"
            "model.safetensors"
        )
        secure_path_calls = []

        def fake_secure_path(value, trust=None, **kwargs):
            secure_path_calls.append((value, trust, kwargs))
            return value

        with (
            patch(
                "definers.model_installation._huggingface_repo_files",
                return_value=("model.safetensors",),
            ),
            patch(
                "definers.model_installation.hf_file_download",
                MagicMock(return_value=cached_path),
            ),
            patch.object(
                repository_sync, "_load_model", return_value={}
            ) as mock_load,
            patch.object(repository_sync, "_apply_turbo_optimizations"),
            patch.object(repository_sync, "free"),
            patch("definers.system.secure_path", side_effect=fake_secure_path),
            patch.dict(repository_sync.MODELS, {}, clear=True),
        ):
            repository_sync.init_model_file(blob_url, turbo=False)

        self.assertEqual(len(secure_path_calls), 1)
        self.assertEqual(secure_path_calls[0][0], cached_path)
        self.assertEqual(
            secure_path_calls[0][1],
            (
                "C:\\Users\\User\\.cache\\huggingface\\hub\\models--ibm-granite--granite-4.0-h-350m\\snapshots\\3b17b717b8f2f5d305b0a92c1491e239aeda19c8",
            ),
        )
        mock_load.assert_called_once_with(
            cached_path,
            "safetensors",
            shard_paths=(),
            loader_kind="file",
            task_key=blob_url,
            trusted_directories=(
                "C:\\Users\\User\\.cache\\huggingface\\hub\\models--ibm-granite--granite-4.0-h-350m\\snapshots\\3b17b717b8f2f5d305b0a92c1491e239aeda19c8",
            ),
        )

    def test_select_huggingface_files_accepts_suffix_hint(self):
        reference = repository_sync.HuggingFaceReference("user/model")
        repo_files = (
            "weights/001-my-model.safetensors",
            "weights/002-my-model.safetensors",
        )

        selected_files = repository_sync._select_huggingface_files(
            reference,
            repo_files,
            "my-model.safetensors",
            "safetensors",
        )

        self.assertEqual(
            selected_files,
            (
                "weights/001-my-model.safetensors",
                "weights/002-my-model.safetensors",
            ),
        )

    def test_init_model_file_uses_snapshot_for_transformers_repo(self):
        repo_id = "ibm-granite/granite-4.0-h-350m"
        snapshot_dir = "C:/Users/User/.cache/huggingface/hub/models--ibm-granite--granite-4.0-h-350m/snapshots/abcdef"
        fake_transformers_module = types.ModuleType("transformers")

        mock_snapshot_download = MagicMock(return_value=snapshot_dir)
        mock_hf_download = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.batch_decode.return_value = ["decoded"]
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_processor = MagicMock()
        mock_processor.tokenizer = mock_tokenizer
        fake_transformers_module.AutoModelForCausalLM = MagicMock()
        fake_transformers_module.AutoModelForCausalLM.from_pretrained = (
            MagicMock(return_value=mock_model)
        )
        fake_transformers_module.AutoProcessor = MagicMock()
        fake_transformers_module.AutoProcessor.from_pretrained = MagicMock(
            return_value=mock_processor
        )
        fake_transformers_module.AutoTokenizer = MagicMock()
        fake_transformers_module.AutoTokenizer.from_pretrained = MagicMock(
            return_value=mock_tokenizer
        )

        local_models = {}
        local_processors = {}
        local_tokenizers = {}
        secure_path_calls = []

        def fake_secure_path(value, trust=None, **kwargs):
            secure_path_calls.append((value, trust, kwargs))
            return value

        with (
            patch.dict(
                sys.modules,
                {
                    "transformers": fake_transformers_module,
                },
            ),
            patch(
                "definers.model_installation._huggingface_repo_files",
                return_value=(
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                ),
            ),
            patch(
                "definers.model_installation.hf_snapshot_download",
                mock_snapshot_download,
            ),
            patch(
                "definers.model_installation.hf_file_download",
                mock_hf_download,
            ),
            patch.object(repository_sync, "_apply_turbo_optimizations"),
            patch.object(repository_sync, "device", return_value="cpu"),
            patch.object(repository_sync, "free"),
            patch.object(repository_sync, "MODELS", local_models),
            patch.object(repository_sync, "PROCESSORS", local_processors),
            patch.object(repository_sync, "TOKENIZERS", local_tokenizers),
            patch("definers.system.secure_path", side_effect=fake_secure_path),
        ):
            repository_sync.init_model_file(repo_id, turbo=False)

        mock_hf_download.assert_not_called()
        mock_snapshot_download.assert_called_once()
        self.assertEqual(
            mock_snapshot_download.call_args.kwargs["repo_id"],
            repo_id,
        )
        self.assertIsNone(mock_snapshot_download.call_args.kwargs["revision"])
        self.assertEqual(
            mock_snapshot_download.call_args.kwargs["allow_patterns"],
            [
                "*.bin",
                "*.json",
                "*.model",
                "*.py",
                "*.safetensors",
                "*.vocab",
            ],
        )
        self.assertEqual(
            mock_snapshot_download.call_args.kwargs["item_label"],
            repo_id,
        )
        self.assertEqual(
            mock_snapshot_download.call_args.kwargs["detail"],
            "Downloading text-generation repository.",
        )
        fake_transformers_module.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            snapshot_dir,
            trust_remote_code=True,
        )
        fake_transformers_module.AutoProcessor.from_pretrained.assert_called_once_with(
            snapshot_dir,
            trust_remote_code=True,
        )
        fake_transformers_module.AutoTokenizer.from_pretrained.assert_called_once_with(
            snapshot_dir,
            trust_remote_code=True,
        )
        self.assertIs(local_processors[repo_id], mock_processor)
        self.assertIs(local_tokenizers[repo_id], mock_tokenizer)
        self.assertIsInstance(
            local_models[repo_id],
            repository_sync.TextGenerationModelAdapter,
        )
        self.assertEqual(len(secure_path_calls), 1)
        self.assertEqual(secure_path_calls[0][0], snapshot_dir)


if __name__ == "__main__":
    unittest.main()
