from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aoi_processor import AOI_Processor
from .utils.output import save_multitemporal_chips, save_thumbnails, save_fire_chips
from .utils.array import unique_class, process_array, missing_values, harmonize_to_old, get_chip_slices, rasterize_aoi
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import rioxarray 
from shapely.geometry import Polygon, box, shape, mapping
import geopandas as gpd
import time as systime
from pathlib import Path
import json

class ChipGenerator:
    def __init__(self, processor: "AOI_Processor"):
        self.processor = processor
        self.chip_entries = []
        self.sensor_burn_masks= {}
        self.sensor_chip_slices= {}
        self.allowed_slice_ids = None
        
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
            if self.allowed_slice_ids is not None and slice_idx not in self.allowed_slice_ids: # generate control chips only from event chip locations
                print("chips not available in the event chips")
                continue
            slice_results = []
            for q_idx, q_stack in enumerate(stack):
                status = None
                try:
                    # start_time = systime.perf_counter()
                    chip = q_stack.isel(y=slice(y0, y1), x=slice(x0, x1))
                    # chip = chip.compute()
                    # end_time = systime.perf_counter()
                    # elapsed_time = end_time - start_time
                    # print(f"Elapsed time for computing chip {slice_idx} for {q_idx}: {int(elapsed_time//60)} min {elapsed_time%60:.2f} sec")


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
                    
                    if missing_values(chip, chip_size, sample_size):
                        raise ValueError(f"Missing values for {sensor_name} quarter {q_idx+1}")
                    
                    chip = chip.fillna(-999)
                    chip = chip.rio.write_nodata(-999)
                    chip = chip.rename(f"{sensor_name}")

                    if np.isnan(chip.values).any():
                        raise ValueError(f"NaN values still present after filling")

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
        
        elapsed_time = {}
        working_directory = Path(self.processor.config.directory.working) / self.processor.config.dataset.version
        time_log_path = working_directory / 'logs/elapsed_time_log.json'
        for name, stack in self.processor.stacks.items():
            # self.processor.stacks[name] = stack.persist()
            # print(stack)
            start_time = systime.perf_counter()
            self.processor.stacks[name] = stack.compute()
            end_time = systime.perf_counter()
            duration = end_time - start_time
            print(f"Elapsed time for computing for {name}: {int(duration//60)} min {duration%60:.2f} sec")
            elapsed_time[name] = {
                "elapsed_seconds": round(duration, 2),
                "elapsed_minutes": round(duration / 60, 2)
                }
            
        if time_log_path.exists():
            with open(time_log_path, "r") as f:
                log = json.load(f)
        else:
            log = {}
        
        if self.processor.aoi_index not in log:
            log[self.processor.aoi_index] = {"runs": []}
        log[self.processor.aoi_index]["runs"].append(elapsed_time)

        with open(time_log_path, "w") as f:
            json.dump(log, f, indent=4)
        
        print(f"Generating event chips for AOI {self.processor.aoi_index}")
        self.sensor_burn_masks = {}
        self.sensor_chip_slices = {}
        all_sensor_results = {}

        if time_series_type == "control":
            if not self.processor.valid_event_chip_ids:
                print("[skip-controls] No event chips recorded for this AOI.")
                return metadata_df.iloc[start_len:].copy()
            self.allowed_slice_ids = set(self.processor.valid_event_chip_ids)
            self.sensor_chip_slices = self.processor.event_chip_slices
            print(f"Generating control chips from {len(self.allowed_slice_ids)} event chip locations.")
        else:
            for sensor_name, stack in self.processor.stacks.items():
                print(f"Extracting burn-rich chip areas from {time_series_type} stack for {sensor_name}")
                self.burn_mask = rasterize_aoi(self.processor.aoi, stack[0][0])
                burn_mask = self.burn_mask.rio.reproject_match(
                        stack[0][0],
                        resampling=rasterio.enums.Resampling.nearest
                        )
                self.sensor_burn_masks[sensor_name] = burn_mask.compute()
                self.sensor_chip_slices[sensor_name] = get_chip_slices(stack[0][0], burn_mask, self.processor.config)
                print(f"{sensor_name}: {len(self.sensor_chip_slices[sensor_name])} chip locations")
                
                self.processor.event_chip_slices[sensor_name] = self.sensor_chip_slices[sensor_name]
        
        for sensor_name, stack in self.processor.stacks.items():
            start_time = systime.perf_counter()
            results = self.gen_fire_chips(stack, sensor_name, self.processor.config)
            all_sensor_results[sensor_name] = results
            end_time = systime.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time for generating {sensor_name} chips: {int(elapsed_time//60)} min {elapsed_time%60:.2f} sec")

        
        # Group results by slice_idx: Assumes each sensor returns results with slice_idx as last element
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
                        del chip
            
            self.processor.valid_event_chip_ids.add(chip_index)            
            

        return metadata_df.iloc[start_len:].copy()