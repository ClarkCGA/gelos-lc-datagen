from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aoi_processor import AOI_Processor
from .utils.output import save_multitemporal_chips, save_thumbnails, save_fire_chips
from .utils.array import unique_class, process_array, missing_values, harmonize_to_old
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon, box, shape, mapping
import geopandas as gpd

class ChipGenerator:
    def __init__(self, processor: "AOI_Processor"):
        self.processor = processor
        self.chip_entries = []
        
    def gen_chips(self, index, arrays):
        """
        Saves chip data arrays to files.
        """
        lc_path = f"{self.processor.working_directory}/lc_{index:06}.tif"
        dem_path = f"{self.processor.working_directory}/dem_{index:06}.tif"
        sentinel_2_dates, sentinel_1_dates, landsat_dates = [], [], []
        sentinel_2_dates = save_multitemporal_chips(arrays['sentinel_2'], self.processor.working_directory, index)
        sentinel_1_dates = save_multitemporal_chips(arrays['sentinel_1'], self.processor.working_directory, index)
        landsat_dates = save_multitemporal_chips(arrays['landsat'], self.processor.working_directory, index)

        save_thumbnails(arrays['sentinel_2'], self.processor.working_directory, index)
        save_thumbnails(arrays['landsat'], self.processor.working_directory, index)
        save_thumbnails(arrays['sentinel_1'], self.processor.working_directory, index)
    
        arrays['land_cover'].rio.to_raster(lc_path)
        arrays['dem'].rio.to_raster(dem_path)
        return sentinel_2_dates, sentinel_1_dates, landsat_dates
 
    def generate_from_aoi(self):
        for name, stack in self.processor.stacks.items():
            # if name == "sentinel_2":
            #     continue
            print(f"loading {name} stack")
            self.processor.stacks[name] = stack.compute()

        land_cover_sample_size = int(self.processor.config.chips.sample_size / self.processor.config.land_cover.resolution)
        
        self.land_cover_min = self.processor.stacks['land_cover'].coarsen(x = land_cover_sample_size,
                                         y = land_cover_sample_size,
                                         boundary = "trim"
                                        ).min()
        self.land_cover_max = self.processor.stacks['land_cover'].coarsen(x = land_cover_sample_size,
                                         y = land_cover_sample_size,
                                         boundary = "trim"
                                        ).max()
        self.land_cover_uniqueness = (self.land_cover_min == self.land_cover_max) & (self.land_cover_min > 0)
        # self.land_cover_uniqueness[0:2, :] = False
        # self.land_cover_uniqueness[-2:, :] = False
        # self.land_cover_uniqueness[:, 0:2] = False
        # self.land_cover_uniqueness[:, -2:] = False

        ys, xs = np.where(self.land_cover_uniqueness)

        # Following indices are added to limit the number of rangeland, bareground, and water chips per tile
        land_cover_indices = {1: 0, 2: 0, 5: 0, 7: 0, 8: 0, 11: 0}

        for index in range(0, len(ys)):

            sentinel_2_dates, sentinel_1_dates, landsat_dates = [], [], []
            land_cover = None
            footprints = {}
            arrays = {}
            status = None

            try:
                x = xs[index]
                y = ys[index]


                # process the land cover stack first, to check land cover information
                arrays["land_cover"], footprints["land_cover"] = process_array(
                    stack = self.processor.stacks['land_cover'],
                    epsg = self.processor.epsg,
                    coords = (x, y),
                    array_name = "land_cover",
                    chip_size = self.processor.config.chips.chip_size,
                    sample_size = self.processor.config.chips.sample_size,
                    resolution = self.processor.config.land_cover.resolution,
                    fill_na = self.processor.config.land_cover.fill_na,
                    na_value = self.processor.config.land_cover.na_value,
                    dtype = self.processor.config.land_cover.dtype,
                )

                if (~np.isin(arrays['land_cover'], [1, 2, 4, 5, 7, 8, 11])).any():
                    raise ValueError("land_cover_values_wrong")

                if (np.isin(arrays['land_cover'], [4])).any():
                    raise ValueError("land_cover_values_flooded_vegetation")

                land_cover = int(np.unique(arrays['land_cover'])[0])
                
                if land_cover_indices[land_cover] > 400:
                    raise ValueError(f"land_cover_{land_cover}_limit")

                # process the rest of the stacks into arrays
                for name, stack in self.processor.stacks.items():
                    if name == 'land_cover':
                        continue
                    stack_config = getattr(self.processor.config, name)
                    arrays[name], footprints[name] = process_array(
                        stack = stack,
                        epsg = self.processor.epsg,
                        coords = (x, y),
                        array_name = name,
                        chip_size = self.processor.config.chips.chip_size,
                        sample_size = self.processor.config.chips.sample_size,
                        resolution = stack_config.resolution,
                        fill_na = stack_config.fill_na,
                        na_value = stack_config.na_value,
                        dtype = stack_config.dtype,
                    )

                # generate chips from arrays
                print(f"Generating Chips for chip {self.processor.chip_index}...")
                sentinel_2_dates, sentinel_1_dates, landsat_dates = self.gen_chips(self.processor.chip_index, arrays)
                status = 'success'
                land_cover_indices[land_cover] += 1

            except Exception as e:
                print(e)
                status = str(e)    

            finally:
                self.chip_entries.append({
                        'chip_index': self.processor.chip_index,
                        'aoi_index': self.processor.aoi_index,
                        'sentinel_2_dates': sentinel_2_dates,
                        'sentinel_1_dates': sentinel_1_dates,
                        'landsat_dates': landsat_dates,
                        'land_cover': land_cover,
                        'chip_footprint': footprints.get('land_cover'),
                        'epsg': self.processor.epsg,
                        'status': status,
                })
                self.processor.chip_index += 1

        chip_df = pd.DataFrame(self.chip_entries)
        return chip_df
    
    def gen_fire_chips(self,
                       stack, #chips per quarter per year
                       sensor_name,
                       chip_slice,
                       config
                       ):
        y0, y1, x0, x1 = chip_slice
        resolution = abs(stack[0].rio.resolution()[0])
        sample_size = int(config.chips.sample_size / resolution)
        chip_size = int(config.chips.chip_size / resolution)

        results = []
        for q_idx, q_stack in enumerate(stack):
            status = None
            chip = q_stack.isel(y=slice(y0, y1), x=slice(x0, x1), drop=False)
            try:
                chip = chip.compute()
            except Exception as e:
                print(f"{e} for {sensor_name} Q_{q_idx+1} data")
                status = str(e)
                continue
                
            if "time" not in chip.dims:
                chip = chip.expand_dims("time")
                
            if sensor_name == "sentinel_2":
                chip = harmonize_to_old(chip)
            chip = chip.fillna(-999)
            chip = chip.rio.write_nodata(-999)
            chip = chip.astype(np.dtype(np.int16))
            chip = chip.rename(f"{sensor_name}")

            if missing_values(chip, chip_size, sample_size):
                print(f"{sensor_name} slice Q{q_idx+1} skipped: missing values")
                # raise ValueError(f"{sensor_name} missing values")
                continue
            
            epsg = int(chip.rio.crs.to_epsg()) if chip.rio.crs else int(chip.coords.get("epsg", xr.DataArray([0])).values[0])

            if chip.rio.crs is None and epsg:
                chip = chip.rio.write_crs(f"EPSG:{epsg}")
            chip = chip.assign_coords(
                x=chip.x.astype(float),
                y=chip.y.astype(float),
            )
            chip = chip.sortby("x")
            chip = chip.sortby("y", ascending=False)
            
            native_footprint = gpd.GeoSeries([box(*chip.rio.bounds())], crs=chip.rio.crs)
            # Reproject the GeoSeries to EPSG:4326 and get the geometry
            footprint = native_footprint.to_crs("EPSG:4326").iloc[0]
            footprint = footprint.wkt

            status = "success" if status is None else status
            results.append((chip, footprint, epsg, status))

        return results

    def generate_time_series(self, time_series_type, metadata_df):
        start_len = len(metadata_df)
        chip_slices = (self.processor.event_chip_slices if time_series_type == "event" 
                                      else self.processor.event_chip_slices)
       
        for chip_index, chip_slice in enumerate(chip_slices):
            for sensor_name, stack in self.processor.stacks.items():
                print(f"Extracting burn-rich chip areas from {time_series_type} stack for {sensor_name}")
                try:
                    results = self.gen_fire_chips(
                        stack, sensor_name, chip_slice, self.processor.config
                    )
                except Exception as e:
                    print(f"{sensor_name} chip slice {chip_index} error {e}")
                    status = str(e)
                    continue
                for chip, footprint, epsg, status in results:
                    metadata_df = save_fire_chips(
                            chip,
                            self.processor.aoi_index,
                            self.processor.aoi,
                            chip_index,
                            time_series_type,
                            metadata_df,
                            epsg,
                            self.processor.config,
                            footprint,
                            sensor_name, 
                            status
                        )
        return metadata_df.iloc[start_len:].copy()

