from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aoi_processor import AOI_Processor
from .utils.output import save_multitemporal_chips, save_thumbnails, save_fire_chips
from .utils.array import unique_class, process_array, missing_values, harmonize_to_old, extract_quarterly_fire_chips
import numpy as np
import pandas as pd

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
    
    def generate_fire_chips(self, 
                            stack,
                            chip_slice,
                            chip_id_num,
                            aoi,
                            aoi_index,
                            config,
                            time_series_type,
                            metadata_df
                            ):
        chip_quarter_data = extract_quarterly_fire_chips(stack, chip_slice)

        for chip, epsg, quarter_idx in chip_quarter_data:
            dt = pd.to_datetime(str(chip.time.values[0]))
            if time_series_type == "event":
                chip_index = f"{aoi_index:05d}_{chip_id_num:02d}_e_Q{quarter_idx+1}_{dt.strftime('%Y%m%d')}"
            else:
                chip_index = f"{aoi_index:05d}_{chip_id_num:02d}_c_Q{quarter_idx+1}_{dt.strftime("%Y%m%d")}"

            try:
                stack = stack.compute()
            except:
                print(f"skipping the AOI for no {stack.name} data")
        
            if stack.shape[2] != 224 or stack.shape[3] != 224:
                print(f"Skipping chip ID {chip_index} for mismatch dimensions")
                return metadata_df
            
            for crop_idx, stack_cropped in enumerate(stack):
                if "time" not in stack_cropped.dims:
                    if "time" in stack_cropped.coords:
                        stack_cropped = stack_cropped.expand_dims("time")
                    else:
                        print(f"Skipping chip ID {chip_index} — no time dimension present")
                        continue
                if missing_values(stack_cropped, config['chips']['chip_size'], config['chips']['chip_size']):
                    print(f"Skipping chip ID {chip_index} for missing values")
                    continue      
                
                if stack.name == "sentinel-2":
                    stack_cropped = harmonize_to_old(stack_cropped)
                stack_cropped = stack_cropped.fillna(-999)
                stack_cropped = stack_cropped.rio.write_nodata(-999)
                stack_cropped = stack_cropped.astype(np.dtype(np.int16))
                stack_cropped = stack_cropped.rename(f"{stack.name}") # TODO: Match configuration according to the platform

                out_path = f"{self.config.directory.output}_{stack_cropped.name}_{chip_index:08}.tif"
                metadata_df = save_fire_chips(stack_cropped, aoi_index, aoi, chip_index, crop_idx, time_series_type, metadata_df, epsg, out_path)
        return metadata_df
