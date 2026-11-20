# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-11-12

### Added

- Add streaming RSG labeling from camera video, with optional ego motion and meta-actions,
  producing reasoning graphs and time-aligned CoC descriptions.
- Support Physical AI AV inputs using existing trajectory caches, meta-action files and
  extracted camera videos.
- Add a lightweight, typed CoC planning API for applications that provide their own media,
  inference and storage.
- Add verified source export, reproducible prompt/component identity and conformance fixtures.

### Changed

- Require Python 3.11 or newer, with Python 3.12 in the Docker image.
- Keep NumPy, pandas and Pydantic in the core installation; provide `pipeline`, `streaming` and
  `qwen` extras for standalone labeling, streaming RSG and local model execution.
  Base installation no longer installs the standalone workflow dependencies.

## [1.0.0] - 2026-05-26

### Added

- Initial public code release of the Chain-of-Causation autolabeling pipeline.
