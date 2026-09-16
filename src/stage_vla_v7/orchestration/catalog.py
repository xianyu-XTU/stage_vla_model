"""Application-owned physical metadata catalog."""

from __future__ import annotations

from collections.abc import Iterable

from stage_vla_v7.interfaces import ObjectProfile


class ObjectCatalog:
    """Resolve semantic labels to physical profiles without using a model."""

    def __init__(self, profiles: Iterable[ObjectProfile] = ()) -> None:
        self._profiles: dict[str, ObjectProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: ObjectProfile, *, replace: bool = False) -> None:
        if profile.label in self._profiles and not replace:
            raise ValueError(f"object profile {profile.label!r} is already registered")
        self._profiles[profile.label] = profile

    def resolve(self, label: str) -> ObjectProfile:
        try:
            return self._profiles[label]
        except KeyError as exc:
            raise LookupError(f"no physical profile is registered for {label!r}") from exc

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._profiles)


def rigid_cube_profile(
    label: str,
    *,
    size_m: float = 0.04,
    mass_kg: float = 0.05,
) -> ObjectProfile:
    """Create a flat-supported rigid cube profile."""
    return ObjectProfile(
        label=label,
        geometry="box",
        size_xyz_m=(size_m, size_m, size_m),
        mass_kg=mass_kg,
    )


def default_cube_catalog() -> ObjectCatalog:
    """Return labels needed by the initial V5 cube migration."""
    labels = (
        "red_cube",
        "blue_cube",
        "green_cube",
        "yellow_cube",
        *(f"cube_{index}" for index in range(1, 9)),
    )
    return ObjectCatalog(rigid_cube_profile(label) for label in labels)
