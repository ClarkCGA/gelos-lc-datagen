from dataclasses import dataclass, field
from typing import List, Optional, Union
import yaml
import numpy as np


@dataclass
class PlatformConfig:
    """A base class for satellite platform configurations."""
    collection: str
    resolution: int
    native_crs: bool
    bands: List[str] = field(default_factory=list)
    fill_na: bool
    na_value: Union[int, float]    
    dtype: np.dtype

    def __post_init__(self):
        """Converts dtype string from YAML to a numpy.dtype object."""
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype)


@dataclass
class Sentinel2Config(PlatformConfig):
    time_ranges: List[str]
    nodata_pixel_percentage: int
    cloud_cover: int
    cloud_band: str

@dataclass
class Sentinel1Config(PlatformConfig):
    nodata_pixel_percentage: int
    delta_days: int

@dataclass
class LandsatConfig(PlatformConfig):
    platforms: List[str]
    cloud_cover: int
    cloud_band: str
    delta_days: int

@dataclass
class DemConfig(PlatformConfig):
    year: str

@dataclass
class LandCoverConfig(PlatformConfig):
    year: str
    sampling_factor: Optional[int] = None

@dataclass
class ChipConfig:
    sample_size: int
    chip_size: int

@dataclass
class DatasetConfig:
    version: str

@dataclass
class AoiConfig:
    version: str

@dataclass
class MetadataConfig:
    file: str

@dataclass
class DirectoryConfig:
    working_dir: str
    output_dir: str

@dataclass
class GELOSConfig:
    """The main container for all configuration."""
    dataset: DatasetConfig
    aoi: AoiConfig
    directory: DirectoryConfig
    metadata: MetadataConfig
    log_errors: bool
    sentinel_2: Sentinel2Config
    sentinel_1: Sentinel1Config
    landsat: LandsatConfig
    dem: DemConfig
    land_cover: LandCoverConfig
    chips: ChipConfig

    @classmethod
    def from_yaml(cls, path: str):
        """Loads and parses the config from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            dataset=DatasetConfig(**config_dict['dataset']),
            aoi=AoiConfig(**config_dict['aoi']),
            directory=DirectoryConfig(**config_dict['directory']),
            metadata=MetadataConfig(**config_dict['metadata']),
            log_errors=config_dict['log_errors'],
            sentinel_2=Sentinel2Config(**config_dict['sentinel_2']),
            sentinel_1=Sentinel1Config(**config_dict['sentinel_1']),
            landsat=LandsatConfig(**config_dict['landsat']),
            dem=DemConfig(**config_dict.get('dem', {})),
            land_cover=LandCoverConfig(**config_dict.get('land_cover', {})),
            chips=ChipConfig(**config_dict['chips'])
        )
