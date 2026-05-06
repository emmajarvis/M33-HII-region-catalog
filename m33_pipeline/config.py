from dataclasses import dataclass, field


FIELDS = ["NW", "NE", "SW", "SE", "F5", "F6", "F7", "F8", "F9"]


@dataclass(frozen=True)
class FieldConfig:
    field: str
    max_zoi_pc: int = 100
    signoi_threshold: float = 3.0
    has_oiii_amplitude: bool = True


@dataclass(frozen=True)
class PhotometryConfig:
    max_zoi_pc: int = 100
    edge_ring_iterations: int = 1
    clip_negative_after_bg: bool = False
    dig_clip_sigma: float = 3.0
    dig_clip_iterations: int = 2


@dataclass(frozen=True)
class DerivedConfig:
    catalog_dir: str = "CATALOGS/flux_catalogs"
    logu_n_mc: int = 2000
    density_n_mc: int = 200
    metallicity_n_mc: int = 500


@dataclass(frozen=True)
class PipelineConfig:
    fields: tuple[str, ...] = tuple(FIELDS)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    derived: DerivedConfig = field(default_factory=DerivedConfig)


FIELD_OVERRIDES = {
    "F5": {"signoi_threshold": 4.5},
    "F9": {"has_oiii_amplitude": False},
}


def get_field_config(field_name: str) -> FieldConfig:
    if field_name not in FIELDS and field_name != "NW":
        raise ValueError(f"Unknown field: {field_name}")
    overrides = FIELD_OVERRIDES.get(field_name, {})
    return FieldConfig(field=field_name, **overrides)


def get_photometry_config() -> PhotometryConfig:
    return PhotometryConfig()


def get_derived_config() -> DerivedConfig:
    return DerivedConfig()


def get_pipeline_config() -> PipelineConfig:
    return PipelineConfig()
