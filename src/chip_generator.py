from src.downloader import Downloader
from src.aoi_processor import AOI_Processor
class ChipGenerator:
    def __init__(self, processor: AOI_Processor):
        self.donwloader = processor
    def process_chip(self):
        #processing logic
        return