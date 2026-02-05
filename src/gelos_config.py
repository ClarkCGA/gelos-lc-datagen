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
    fill_na: bool
    na_value: Union[int, float]    
    dtype: np.dtype
    def __post_init__(self):
        """Converts dtype string from YAML to a numpy.dtype object."""
        if isinstance(self.dtype, str):
            self.dtype = np.dtype(self.dtype)


@dataclass
class S2L2AConfig(PlatformConfig):
    time_ranges: List[str]
    nodata_pixel_percentage: int
    cloud_cover: int
    cloud_band: str
    bands: List[str]

@dataclass
class S1RTCConfig(PlatformConfig):
    nodata_pixel_percentage: int
    delta_days: int
    bands: List[str]

@dataclass
class LC2L2Config(PlatformConfig):
    platforms: List[str]
    cloud_cover: int
    cloud_band: str
    delta_days: int
    bands: List[str]

@dataclass
class DEMConfig(PlatformConfig):
    year: str

@dataclass
class LULCConfig(PlatformConfig):
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
    include_indices: Optional[List[int]]
    exclude_indices: Optional[List[int]]

@dataclass
class DirectoryConfig:
    working: str
    output: str
    zip_output: bool

@dataclass
class GELOSConfig:
    """The main container for all configuration."""
    dataset: DatasetConfig
    aoi: AoiConfig
    directory: DirectoryConfig
    log_errors: bool
    s2l2a: S2L2AConfig
    s1rtc: S1RTCConfig
    lc2l2: LC2L2Config
    dem: DEMConfig
    lulc: LULCConfig
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
            log_errors=config_dict['log_errors'],
            s2l2a=S2L2AConfig(**config_dict['s2l2a']),
            s1rtc=S1RTCConfig(**config_dict['s1rtc']),
            lc2l2=LC2L2Config(**config_dict['lc2l2']),
            dem=DEMConfig(**config_dict.get('dem', {})),
            lulc=LULCConfig(**config_dict.get('lulc', {})),
            chips=ChipConfig(**config_dict['chips'])
        )
