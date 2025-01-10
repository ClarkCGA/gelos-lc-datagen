# Generating Data for GFM-Bech Pilot

This repo contains the code to generate samples for the pilot version of GFM-Bench presented at AGU Fall Meeting 2024. 

To run the code:
```
docker run -it -p 888s8:8888 -p 8787:8787 -v <PATH_TO_REPO>:/home/benchuser/code/:rw -v <PATH_TO_DATA>:/home/benchuser/data/ gfm-benchsss
```

The repo contains areas of interest for sample generation in two separate GeoJSON files located under `data/`.

The main notebook to generate the data is `main.ipynb`, and the `data_cleaning.ipynb` is used to clean the data befor running through the model. 
