from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aoi_processor import AOI_Processor
from .utils.output import save_multitemporal_chips, save_thumbnails, save_fire_chips
from .utils.array import unique_class, process_array, missing_values, harmonize_to_old, get_chip_slices
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import rioxarray 
from shapely.geometry import Polygon, box, shape, mapping
import geopandas as gpd

class ChipGenerator:
    def __init__(self, processor: "AOI_Processor"):
        self.processor = processor
        self.chip_entries = []
        
    def gen_fire_chips(self, stack, sensor_name, config):
        """Generate chips for all burn areas in this sensor's stack"""
        resolution = abs(stack[0].rio.resolution()[0])
        sample_size = int(config.chips.sample_size / resolution)
        chip_size = int(config.chips.chip_size / resolution)

        chip_slices = self.sensor_chip_slices[sensor_name]
        
        if not chip_slices:
            print(f"No valid chip slices found for {sensor_name}")
            return []
        
        all_results = []
        for slice_idx, (y0, y1, x0, x1) in enumerate(chip_slices):
            slice_results = []
            for q_idx, q_stack in enumerate(stack):
                status = None
                try:
                    # Use pixel indices directly - they're in this sensor's grid!
                    chip = q_stack.isel(y=slice(y0, y1), x=slice(x0, x1))
                    chip = chip.compute()

                    if np.isnan(chip).any():
                        print(f"WARNING: NaN found in {sensor_name} Q{q_idx+1}, filling with -999")
                        chip = chip.fillna(-999)
                    
                    if (chip == 0).all():
                        raise ValueError(f"All zeros in {sensor_name} quarter {q_idx+1}")
                    
                    if "time" not in chip.dims:
                        chip = chip.expand_dims("time")
                        
                    epsg = int(chip.rio.crs.to_epsg()) if chip.rio.crs else self.processor.epsg
                    if chip.rio.crs is None and epsg:
                        chip = chip.rio.write_crs(f"EPSG:{epsg}")
                        
                    chip = chip.assign_coords(
                        x=chip.x.astype(float),
                        y=chip.y.astype(float),
                    )
                    chip = chip.sortby("x")
                    chip = chip.sortby("y", ascending=False)

                    if sensor_name == "sentinel_2":
                        chip = harmonize_to_old(chip)
                    
                    chip = chip.fillna(-999)
                    chip = chip.rio.write_nodata(-999)
                    chip = chip.rename(f"{sensor_name}")

                    if np.isnan(chip.values).any():
                        raise ValueError(f"NaN values still present after filling")

                    if missing_values(chip, chip_size, sample_size):
                        raise ValueError(f"Missing values for {sensor_name} quarter {q_idx+1}")
                    
                    status = "success"
                    ts = pd.to_datetime(str(chip.time.values[0])) if "time" in chip.dims else pd.to_datetime(str(q_stack.time.values[0]))
                    quarter = int(pd.Timestamp(ts).quarter)

                    native_footprint = gpd.GeoSeries([box(*chip.rio.bounds())], crs=chip.rio.crs)
                    footprint = native_footprint.to_crs("EPSG:4326").iloc[0].wkt

                    slice_results.append((chip, footprint, epsg, status, ts, quarter, slice_idx))
                    
                except Exception as e:
                    status = str(e)
                    slice_results.append((None, None, None, f"error: {e}", None, q_idx+1, slice_idx))
            
            all_results.extend(slice_results)
        
        return all_results

    def generate_time_series(self, time_series_type, metadata_df):
        start_len = len(metadata_df)

        required_sensors = getattr(getattr(self.processor.config, "dataset", None), "fire", None)
        required_sensors = getattr(required_sensors, "required_event_sensors", ["sentinel_2", "sentinel_1", "landsat"])
        required_quarters = {1, 2, 3, 4}
        
        for name, stack in self.processor.stacks.items():
            self.processor.stacks[name] = stack.persist()

       


        print(f"Generating event chips for AOI {self.processor.aoi_index}")
        self.sensor_burn_masks = {}
        self.sensor_chip_slices = {}

        all_sensor_results = {}

        if not hasattr(self.processor, 'sensor_chip_slices') or self.processor.sensor_chip_slices is None:
            print("Computing chip slices (first time for this AOI)...")
            self.sensor_burn_masks = {}
            self.sensor_chip_slices = {}
            
            for sensor_name, stack in self.processor.stacks.items():
                print(f"Extracting burn-rich chip areas from stack for {sensor_name}")
                burn_mask = self.processor.burn_mask.rio.reproject_match(
                    stack[0][0],
                    resampling=rasterio.enums.Resampling.nearest
                )
                self.sensor_burn_masks[sensor_name] = burn_mask.compute()
                self.sensor_chip_slices[sensor_name] = get_chip_slices(
                    stack[0][0], burn_mask, self.processor.config
                )
                print(f"  {sensor_name}: {len(self.sensor_chip_slices[sensor_name])} chip locations")
            
            # cache on processor for reuse
            self.processor.sensor_chip_slices = self.sensor_chip_slices
            self.processor.sensor_burn_masks = self.sensor_burn_masks
        else:
            print("Reusing cached chip slices from previous run...")
            self.sensor_chip_slices = self.processor.sensor_chip_slices
            self.sensor_burn_masks = self.processor.sensor_burn_masks
            
        # for sensor_name, stack in self.processor.stacks.items():
        #     print(f"Extracting burn-rich chip areas from {time_series_type} stack for {sensor_name}")

        #     burn_mask = self.processor.burn_mask.rio.reproject_match(
        #         stack[0][0],
        #         resampling=rasterio.enums.Resampling.nearest
        #         )
        #     self.sensor_burn_masks[sensor_name] = burn_mask.compute()
        #     self.sensor_chip_slices[sensor_name] = get_chip_slices(stack[0][0], burn_mask, self.processor.config)
        #     print(f"  {sensor_name}: {len(self.sensor_chip_slices[sensor_name])} chip locations")

        for sensor_name, stack in self.processor.stacks.items():
            results = self.gen_fire_chips(stack, sensor_name, self.processor.config)
            all_sensor_results[sensor_name] = results
        
        # Group results by slice_idx
        # Assumes each sensor returns results with slice_idx as last element
        slice_groups = {}
        for sensor_name, results in all_sensor_results.items():
            for result in results:
                if len(result) >= 7:
                    chip, footprint, epsg, status, ts, quarter, slice_idx = result
                    if slice_idx not in slice_groups:
                        slice_groups[slice_idx] = {}
                    if sensor_name not in slice_groups[slice_idx]:
                        slice_groups[slice_idx][sensor_name] = []
                    slice_groups[slice_idx][sensor_name].append(result[:-1])  # Remove slice_idx
            
        # process each slice group
        for chip_index, sensor_results in slice_groups.items():
            coverage = {}
            for sensor_name, results in sensor_results.items():
                if sensor_name in required_sensors:
                    good_quarters = {q for (_, _, _, status, _, q) in results if status == "success"}
                    coverage[sensor_name] = good_quarters
            
            all_ok = all(coverage.get(s, set()) >= required_quarters for s in required_sensors)
            
            
            if time_series_type == "event" and not all_ok:
                print(f"[skip] Event chip {chip_index:02d}: Missing quarters "
                    f"(have: {coverage})")
                continue
            
            # Save all successful chips for this slice
            for sensor_name, results in sensor_results.items():
                for chip, footprint, epsg, status, ts, quarter in results:
                    if status == "success" and chip is not None:
                        metadata_df = save_fire_chips(
                             chip, self.processor.aoi_index, self.processor.aoi,
                            chip_index, time_series_type, metadata_df, epsg,
                            self.processor.config, footprint, sensor_name, status, ts
                        )
                
            self.processor.valid_event_chip_ids.add(chip_index)
            
        return metadata_df.iloc[start_len:].copy()