from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.aoi_processor import AOI_Processor
from src.utils.output import save_multitemporal_chips, save_thumbnails
from src.utils.array import process_array
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
        lulc_path = f"{self.processor.working_directory}/lc_{index:06}.tif"
        dem_path = f"{self.processor.working_directory}/dem_{index:06}.tif"
        s2l2a_dates, s1rtc_dates, lc2l2_dates = [], [], []
        s2l2a_dates = save_multitemporal_chips(arrays['s2l2a'], self.processor.working_directory, index)
        s1rtc_dates = save_multitemporal_chips(arrays['s1rtc'], self.processor.working_directory, index)
        lc2l2_dates = save_multitemporal_chips(arrays['lc2l2'], self.processor.working_directory, index)

        save_thumbnails(arrays['s2l2a'], self.processor.working_directory, index)
        save_thumbnails(arrays['lc2l2'], self.processor.working_directory, index)
        save_thumbnails(arrays['s1rtc'], self.processor.working_directory, index)
    
        arrays['lulc'].rio.to_raster(lulc_path)
        arrays['dem'].rio.to_raster(dem_path)
        return s2l2a_dates, s1rtc_dates, lc2l2_dates
 
    def generate_from_aoi(self):
        for name, stack in self.processor.stacks.items():
            print(f"loading {name} stack")
            self.processor.stacks[name] = stack.compute()

        lulc_sample_size = int(self.processor.config.chips.sample_size / self.processor.config.lulc.resolution)
        
        self.lulc_min = self.processor.stacks['lulc'].coarsen(x = lulc_sample_size,
                                         y = lulc_sample_size,
                                         boundary = "trim"
                                        ).min()
        self.lulc_max = self.processor.stacks['lulc'].coarsen(x = lulc_sample_size,
                                         y = lulc_sample_size,
                                         boundary = "trim"
                                        ).max()
        self.lulc_uniqueness = (self.lulc_min == self.lulc_max) & (self.lulc_min > 0)
        # self.lulc_uniqueness[0:2, :] = False
        # self.lulc_uniqueness[-2:, :] = False
        # self.lulc_uniqueness[:, 0:2] = False
        # self.lulc_uniqueness[:, -2:] = False

        ys, xs = np.where(self.lulc_uniqueness)

        # Following indices are added to limit the number of rangeland, bareground, and water chips per tile
        lulc_indices = {1: 0, 2: 0, 5: 0, 7: 0, 8: 0, 11: 0}

        for index in range(0, len(ys)):

            s2l2a_dates, s1rtc_dates, lc2l2_dates = [], [], []
            footprints = {}
            arrays = {}
            status = None

            try:
                x = xs[index]
                y = ys[index]


                # process the land cover stack first, to check land cover information
                arrays["lulc"], footprints["lulc"] = process_array(
                    stack = self.processor.stacks['lulc'],
                    epsg = self.processor.epsg,
                    coords = (x, y),
                    array_name = "lulc",
                    chip_size = self.processor.config.chips.chip_size,
                    sample_size = self.processor.config.chips.sample_size,
                    resolution = self.processor.config.lulc.resolution,
                    fill_na = self.processor.config.lulc.fill_na,
                    na_value = self.processor.config.lulc.na_value,
                    dtype = self.processor.config.lulc.dtype,
                )

                if (~np.isin(arrays['lulc'], [1, 2, 4, 5, 7, 8, 11])).any():
                    raise ValueError("lulc_values_wrong")

                if (np.isin(arrays['lulc'], [4])).any():
                    raise ValueError("lulc_values_flooded_vegetation")

                chip_lulc = int(np.unique(arrays['lulc'])[0])
                
                if lulc_indices[lulc] > 400:
                    raise ValueError(f"lulc_{chip_lulc}_limit")

                # process th rest of the stacks into arrays
                for name, stack in self.processor.stacks.items():
                    if name == 'lulc':
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
                s2l2a_dates, s1rtc_dates, lc2l2_dates = self.gen_chips(self.processor.chip_index, arrays)
                status = 'success'
                lulc_indices[chip_lulc] += 1

            except Exception as e:
                print(e)
                status = str(e)    

            finally:
                self.chip_entries.append({
                        'chip_index': self.processor.chip_index,
                        'aoi_index': self.processor.aoi_index,
                        's2l2a_dates': s2l2a_dates,
                        's1rtc_dates': s1rtc_dates,
                        'lc2l2_dates': lc2l2_dates,
                        'lulc': chip_lulc,
                        'chip_footprint': footprints.get('lulc'),
                        'epsg': self.processor.epsg,
                        'status': status,
                        **self.processor.scene_ids
                })
                self.processor.chip_index += 1

        chip_df = pd.DataFrame(self.chip_entries)
        return chip_df 