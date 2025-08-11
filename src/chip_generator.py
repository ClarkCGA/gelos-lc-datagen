from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.aoi_processor import AOI_Processor
from src.utils.output import save_multitemporal_chips, save_thumbnails
from src.utils.array import unique_class, process_array
import numpy as np
import pandas as pd

class ChipGenerator:
    def __init__(self, processor: "AOI_Processor"):
        self.processor = processor
        self.chip_entries = {}
        
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
    
        arrays['land_cover'].rio.to_raster(lc_path)
        arrays['dem'].rio.to_raster(dem_path)
        return sentinel_2_dates, sentinel_1_dates, landsat_dates
 
    def generate_from_aoi(self):
        for name, stack in self.processor.stacks:
            print(f"loading {name} stack")
            stack = stack.compute()

        land_cover_sample_size = int(self.processor.config.chips.sample_size / self.processor.config.land_cover.resolution)
        
        self.land_cover_uniqueness = self.processor.stacks['land_cover'].coarsen(x = land_cover_sample_size,
                                         y = land_cover_sample_size,
                                         boundary = "trim"
                                        ).reduce(unique_class)
        self.land_cover_uniqueness[0:2, :] = False
        self.land_cover_uniqueness[-2:, :] = False
        self.land_cover_uniqueness[:, 0:2] = False
        self.land_cover_uniqueness[:, -2:] = False

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
                    stack = stack,
                    epsg = self.processor.epsg,
                    coords = (x, y),
                    array_name = "land_cover",
                    sample_size = self.processor.config.chips.sample_size,
                    resolution = self.processor.config.land_cover.resolution,
                )

                if (~np.isin(arrays['land_cover'], [1, 2, 4, 5, 7, 8, 11])).any():
                    raise ValueError("land_cover_values_wrong")

                if (np.isin(arrays['land_cover'], [4])).any():
                    raise ValueError("land_cover_values_flooded_vegetation")

                land_cover = int(np.unique(arrays['land_cover'])[0])
                
                if land_cover_indices[land_cover] > 400:
                    raise ValueError(f"land_cover_{land_cover}_limit")

                # process th rest of the stacks into arrays
                for name, stack in self.processor.stacks.items():
                    stack_config = getattr(self.processor.config, name)
                    arrays[name], footprints[name] = process_array(
                        stack = stack,
                        epsg = self.processor.epsg,
                        coords = (x, y),
                        array_name = name,
                        sample_size = self.processor.config.chips.sample_size,
                        resolution = stack_config.resolution,
                        fill_na = False,
                        na_value = stack_config.na_value,
                        dtype = stack_config.dtype,
                    )

                # generate chips from arrays
                print(f"Generating Chips for chip {self.processor.chip_index}...")
                sentinel_2_dates, sentinel_1_dates, landsat_dates = self.gen_chips(self.processor.chip_index, arrays)
                status = 'success'
                land_cover_indices[land_cover] += 1

            except Exception as e:
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