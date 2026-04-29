# standard
import json
from pathlib import Path
from typing import Any

# extern
from pydantic import BaseModel, ConfigDict, Field

# intern
from ..utils.hash import md5sum, file_md5sum
from ..defines import EMBEDDING_MODEL, DEFAULT_VLLM, MANIFEST_PATH
from .models import CachedFile


class Manifest(BaseModel):
    """Track indexed files and vector store membership."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(0, ge=0)
    llm_model: str = EMBEDDING_MODEL
    extensions: list[str] = Field(default_factory=list)
    vllm: str = DEFAULT_VLLM
    files_by_extensions: dict[str, dict[str, CachedFile]] = Field(
        default_factory=dict
    )

    @staticmethod
    def existing_manifest_data() -> dict[str, Any]:
        """Load existing manifest JSON data if it exists."""

        if not MANIFEST_PATH.exists():
            return {}

        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(
                f"Manifest path is not a file: {MANIFEST_PATH}"
            )

        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid manifest JSON file: {MANIFEST_PATH}"
            ) from e

        if not isinstance(manifest_data, dict):
            raise ValueError(
                f"Invalid manifest format: expected object in {MANIFEST_PATH}"
            )

        return manifest_data

    def _remove_extensions(self, extensions: set[str]) -> list[str]:
        """Remove files for extensions outside the active set."""

        delete_chunks_ids: list[str] = []

        for ext in list(self.files_by_extensions.keys()):
            if "*" not in extensions and ext not in extensions:
                for cached_file in self.files_by_extensions[ext].values():
                    delete_chunks_ids.extend(cached_file.chunks_ids)

                del self.files_by_extensions[ext]

                if ext in self.extensions:
                    self.extensions.remove(ext)

        return delete_chunks_ids

    def _remove_missing_files(self) -> list[str]:
        """Remove manifest files that no longer exist on disk."""

        delete_chunks_ids: list[str] = []

        for ext in list(self.files_by_extensions.keys()):
            for file_id in list(self.files_by_extensions[ext].keys()):
                cached_file = self.files_by_extensions[ext][file_id]
                path = Path(cached_file.file_path)

                if not path.exists():
                    delete_chunks_ids.extend(cached_file.chunks_ids)
                    del self.files_by_extensions[ext][file_id]

            if not self.files_by_extensions[ext]:
                del self.files_by_extensions[ext]

                if ext in self.extensions:
                    self.extensions.remove(ext)

        return delete_chunks_ids

    @classmethod
    def load(
        cls, chunk_size: int, extensions: set[str]
    ) -> tuple["Manifest", list[str]]:
        """Load or create a manifest for the current index settings."""

        if not MANIFEST_PATH.exists():
            return cls(chunk_size=chunk_size, llm_model=EMBEDDING_MODEL), []

        manifest = cls(**cls.existing_manifest_data())
        delete_chunks_ids: list[str] = []

        if (
            manifest.chunk_size != chunk_size
            or manifest.llm_model != EMBEDDING_MODEL
            or manifest.vllm != DEFAULT_VLLM
        ):
            for files_by_id in manifest.files_by_extensions.values():
                for cached_file in files_by_id.values():
                    delete_chunks_ids.extend(cached_file.chunks_ids)

            return (
                cls(chunk_size=chunk_size, llm_model=EMBEDDING_MODEL),
                delete_chunks_ids
            )

        delete_chunks_ids.extend(manifest._remove_extensions(extensions))
        delete_chunks_ids.extend(manifest._remove_missing_files())

        return manifest, delete_chunks_ids

    def add_store(
        self, chunks_metadata: dict[str, dict[str, Any]],
        chunks_ids: list[str], store: str,
    ) -> None:
        """Mark chunks as present in an index store."""

        for chunk_id in chunks_ids:
            metadata = chunks_metadata[chunk_id]

            file_path = Path(metadata["file_path"])
            file_id: str = md5sum(str(file_path))
            file_suffix: str = file_path.suffix.removeprefix(".").lower()

            manifest_files = self.files_by_extensions.get(file_suffix, {})
            manifest_file = manifest_files.get(file_id)

            if manifest_file is None:
                continue

            manifest_file.chunks_ids.add(chunk_id)
            manifest_file.stores.add(store)

    def sync_files(
        self, files: list[Path]
    ) -> tuple[list[str], set[str], set[str]]:
        """Sync manifest entries with files collected from disk."""

        delete_chunks_ids: list[str] = []
        updated_files_ids: set[str] = set()
        new_files_ids: set[str] = set()

        for file in files:
            file_id: str = md5sum(str(file))
            file_hash: str = file_md5sum(file)
            file_suffix: str = file.suffix.removeprefix(".").lower()

            manifest_files = self.files_by_extensions.setdefault(
                file_suffix, {}
            )
            manifest_file = manifest_files.get(file_id)

            if manifest_file is None:
                manifest_file = CachedFile(
                    file_path=str(file),
                    file_hash=file_hash,
                    chunks_ids=set(),
                )
                manifest_files[file_id] = manifest_file
                new_files_ids.add(file_id)

            elif manifest_file.file_hash != file_hash:
                delete_chunks_ids.extend(manifest_file.chunks_ids)
                manifest_file.chunks_ids = set()
                manifest_file.stores.clear()
                updated_files_ids.add(file_id)

            manifest_file.file_path = str(file)
            manifest_file.file_hash = file_hash

        return delete_chunks_ids, updated_files_ids, new_files_ids
