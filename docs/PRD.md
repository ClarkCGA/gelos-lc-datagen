# Product Requirements Document (PRD)

## 1. Executive Summary
The GELOS land-cover generator produces multi-sensor satellite chip datasets for generating embeddings using the core GELOS repository. A containerized Python pipeline ingests AOIs, queries Microsoft Planetary Computer STAC for Sentinel-2/1, lc2l2, dem, and lulc products, stacks imagery, extracts chips with homogenous land cover, and packages metadata and thumbnails. 

The core value proposition is a reproducible, scalable pipeline that yields balanced, geo-diverse chips with consistent metadata and ready-to-train rasters. The MVP delivers an automated end-to-end run (download → chip generation → cleaning → packaged release) for the LC track using provided configs.

## 2. Mission
- Deliver a reliable, repeatable chip-generation pipeline for GELOS land cover dataset.
- Make setup and execution turnkey via Docker/conda and clear configs.

Core principles:
- Reproducibility (pinned configs, deterministic outputs per version)
- Data quality (strict homogeneity, cloud/nodata checks, class filtering)
- Transparency (traceable metadata, scene provenance)
- Scalability (Dask-backed processing, resumable downloads)
- Portability (containerized runtime, volume mounts)

## 3. Target Users
- ML researchers building/benchmarking geospatial models; intermediate Python; comfortable with notebooks/Jupyter.
- Data engineers curating geospatial corpora; intermediate geospatial tooling (STAC, rioxarray, geopandas).
- Internal contributors maintaining dataset versions; advanced familiarity with pipeline code and configs.

Key needs/pain points:
- Consistent naming/metadata.
- Avoid manual STAC querying and ad-hoc preprocessing.
- Require predictable class balance and quality filtering.
- Desire resumable runs and visibility into failures per AOI.

## 4. MVP Scope
**In Scope (MVP)**
- ✅ Run pipeline via `main.py` using `config.yml` to generate LC dataset version (v0.50.1 default).
- ✅ AOI ingestion from versioned GeoJSON; include/exclude index filtering.
- ✅ STAC search against MPC for S2/S1/lc2l2/dem/LC with cloud/nodata thresholds and date windows.
- ✅ Stacking, homogeneity checks, chip extraction, rasters + thumbnails per modality.
- ✅ Metadata capture (scene IDs, footprints, epsg, status) to CSV/GeoJSON; resumable runs.
- ✅ Data cleaning: class filtering/balancing, URL enrichment, asset renaming/copying, zipping release.
- ✅ Containerized environment (Docker + conda) with volume mounts for external/raw/interim/processed data.

**Out of Scope (MVP)**
- ❌ Web UI or orchestration service (CLI/Jupyter only).
- ❌ Automated cloud deployment; assume local or self-managed compute.
- ❌ Active learning or model training; pipeline stops at packaged data.

## 5. Core Architecture & Patterns
- Batch pipeline orchestrated by `main.py`: load YAML config → set working dirs → Downloader (search/stack) → ChipGenerator → DataCleaner.
- Config-driven design via dataclasses mapping YAML to typed objects.
- STAC interactions through pystac-client with retry policy; MPC signing via planetary-computer SDK.
- Array ops with stackstac/xarray/rioxarray; cloud masking and CRS alignment handled per platform helper.
- Dask LocalCluster for parallel stacking/compute.
- Deterministic file conventions: `{platform}_{chip_id:06}_{date}.tif/png`, `dem_{chip_id:06}.tif`, metadata CSV/GeoJSON.
- Resumability: presence of `chip_metadata.csv` and `aoi_metadata.geojson` drives continuation logic.
- Directory structure: `data/` (AOIs), `src/` (pipeline modules), `notebooks/` (exploration), `docs/` (PRD), `config.yml` (run settings).

## 6. Tools/Features (Pipeline Breakdown)
- Downloader: STAC search for S2/S1/lc2l2/dem/LC with per-platform constraints; persists AOI status and chip metadata.
- AOI Processor: computes overlap bbox, enforces min scenes (4 dates each), determines EPSG/MGRS/orbit/WRS path, stacks data.
- Chip Generator: homogeneity check on land-cover window; per-chip raster export and thumbnails; records footprints/status.
- Data Cleaner: filters invalid classes, ensures 4 dates per modality, optional sampling_factor balancing, enriches metadata (ids, lat/lon, category/color, thumbnail URLs), copies/renames assets to processed dir, zips release.
- Utilities: search helpers (date clipping, orbit/WRS selection), stack helpers (cloud masking, bbox adjustments), array slicing with CRS/footprint handling, thumbnail generation (S1 composite, RGB scaling).

## 7. Technology Stack
- Language: Python 3 (conda env `gelos-lc-datagen`).
- Libraries: dask/distributed, stackstac, rioxarray, xarray, geopandas, pystac-client, planetary-computer, s3fs, tqdm, PIL.
- Runtime: Docker image + docker-compose for Jupyter/Dask; volume mounts for data paths.
- Dev: JupyterLab for notebooks; CLI via `python main.py -c config.yml`.

## 8. Security & Configuration
- Auth: MPC signing handled via planetary-computer SDK (anonymous keys); no user auth in pipeline.
- Config: YAML (`config.yml`) controls dataset version, AOI version, dirs (working/output), logging, per-platform parameters, chip sizes, sampling_factor.
- Secrets: none expected; avoid embedding credentials; if MPC key used, pass via env.
- Deployment: assume trusted local or cloud VM; ensure volumes map to appropriate storage; handle large I/O limits and disk space.

## 9. Success Criteria
- ✅ Pipeline runs to completion for LC v0.50.1 using default config without manual intervention.
- ✅ Each chip has 4 dates per modality and passes homogeneity/valid-class checks.
- ✅ Metadata GeoJSON/CSV includes scene IDs, footprints, EPSG, status, ids; thumbnails generated for all chips.
- ✅ Class balance within configured sampling_factor; no excluded classes present.
- ✅ Release zip produced in processed output with expected file naming and counts.

Quality indicators:
- Reproducible outputs when re-running with same config and data sources.
- Visual spot-checks show correct geolocation and cloud masking.

User experience goals:
- Single-command run; clear progress logs per AOI.
- Resumable behavior on interruption.

## 12. Implementation Phases
**Phase 1: Baseline PRD & Documentation (1-2 days)**
- ✅ Finalize PRD, document run steps and configs.
- Validation: PRD approved; docs reviewed by team.

**Phase 2: Pipeline Hardening (3-5 days)**
- ✅ Verify default config run, tighten error handling/logging, ensure resumability paths.
- ✅ Add sanity checks on class counts and scene minima.
- Validation: successful end-to-end LC run; logs capture AOI/chip failures.

**Phase 3: Packaging & Quality (2-4 days)**
- ✅ Validate outputs (counts, class balance, metadata fields), generate sample QA report.
- ✅ Confirm thumbnails and zips produced; storage layout consistent.
- Validation: checksum/count report; manual spot-check.

**Phase 4: Future Extensions (post-MVP)**
- Validation: scope-approved roadmap item.

## 13. Future Considerations
- Automated QC dashboards (tile coverage, class distribution, cloud stats).
- Parameter sweeps for sampling_factor and cloud thresholds.
- Caching of STAC searches and downloads to reduce retries.

## 14. Risks & Mitigations
- STAC availability/performance: add retries (already present), allow backoff tuning, support cached catalogs.
- Data gaps (insufficient scenes): clear errors, allow AOI skipping and resumability; configurable thresholds.
- Disk/IO constraints: document storage requirements; support streaming/copy throttling.
- Class imbalance persists: refine sampling_factor logic; add pre-run class estimation.
- CRS/bbox mismatches: maintain tests for bbox adjustments; enforce EPSG logging per stack.

## 15. Appendix
- Key files: `main.py`, `config.yml`, `src/gelos_config.py`, `src/downloader.py`, `src/aoi_processor.py`, `src/chip_generator.py`, `src/data_cleaner.py`, `src/utils/*.py`.
- Data inputs: AOI GeoJSONs in `data/map_*.geojson`; external/raw/interim/processed volume mounts via compose.
- Environment: `environment.yml`; Dockerfile/compose for runtime; notebooks for exploration/QA.
