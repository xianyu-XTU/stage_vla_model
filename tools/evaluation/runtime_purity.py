"""Inspect whether the formal V7 process has exposed or loaded V5 Python."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from types import ModuleType

from .bootstrap import V7_ROOT


V5_PACKAGE = "stage_vla"
VENDORED_V5_ROOT = V7_ROOT / "vendor" / "stage_vla_v5"


def is_v5_module_name(name: str) -> bool:
    """Match V5 exactly without treating ``stage_vla_v7`` as V5."""
    return name == V5_PACKAGE or name.startswith(f"{V5_PACKAGE}.")


class V5ImportBlocker:
    """Reject any attempt to load the V5 package during physical evaluation."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if is_v5_module_name(fullname):
            raise ImportError(f"V5 Python imports are forbidden in V7 evaluation: {fullname}")
        return None


def install_v5_import_blocker() -> None:
    """Require a clean startup and prevent later V5 imports in this process."""
    purity = audit_runtime_purity()
    if not purity.verified:
        raise RuntimeError(f"V7 runtime is already V5-exposed: {purity.as_dict()}")
    if not purity.import_blocker_enabled:
        sys.meta_path.insert(0, V5ImportBlocker())


def _normalized_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expandvars(os.fspath(value)))).resolve()


def _is_same_or_child(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RuntimePurity:
    """Serializable evidence for both required runtime-isolation dimensions."""

    vendor_path_exposed: bool
    loaded_v5_modules: tuple[str, ...]
    exposed_vendor_paths: tuple[str, ...]
    v5_root_env_set: bool
    import_blocker_enabled: bool

    @property
    def loaded_v5_module_count(self) -> int:
        return len(self.loaded_v5_modules)

    @property
    def verified(self) -> bool:
        return not self.vendor_path_exposed and self.loaded_v5_module_count == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "vendor_path_exposed": self.vendor_path_exposed,
            "loaded_v5_module_count": self.loaded_v5_module_count,
            "loaded_v5_modules": list(self.loaded_v5_modules),
            "exposed_vendor_paths": list(self.exposed_vendor_paths),
            "v5_root_env_set": self.v5_root_env_set,
            "import_blocker_enabled": self.import_blocker_enabled,
            "verified": self.verified,
        }


def audit_runtime_purity(
    *,
    modules: Mapping[str, ModuleType | object] | None = None,
    search_path: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    vendor_root: str | Path | None = None,
) -> RuntimePurity:
    """Return a deterministic snapshot of V5 module and path exposure."""
    module_map = sys.modules if modules is None else modules
    path_entries = sys.path if search_path is None else search_path
    environment = os.environ if environ is None else environ
    roots = {_normalized_path(vendor_root or VENDORED_V5_ROOT)}
    configured_root = environment.get("STAGE_VLA_V5_ROOT")
    if configured_root:
        roots.add(_normalized_path(configured_root))

    exposed: list[str] = []
    for entry in path_entries:
        if not entry:
            continue
        candidate = _normalized_path(entry)
        exposes_package = (candidate / V5_PACKAGE / "__init__.py").is_file()
        if exposes_package or any(
            _is_same_or_child(candidate, root) for root in roots
        ):
            exposed.append(str(candidate))

    loaded = tuple(sorted(name for name in module_map if is_v5_module_name(name)))
    return RuntimePurity(
        vendor_path_exposed=bool(exposed),
        loaded_v5_modules=loaded,
        exposed_vendor_paths=tuple(dict.fromkeys(exposed)),
        v5_root_env_set=bool(configured_root),
        import_blocker_enabled=any(
            isinstance(finder, V5ImportBlocker) for finder in sys.meta_path
        ),
    )


__all__ = [
    "RuntimePurity",
    "V5ImportBlocker",
    "V5_PACKAGE",
    "VENDORED_V5_ROOT",
    "audit_runtime_purity",
    "install_v5_import_blocker",
    "is_v5_module_name",
]
