# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Placeholder for upcoming changes.

## [1.0.0] - 2026-04-09

### Added
- Rust backend with Axum + SQLx + PostgreSQL integration.
- Dataset APIs for labeling and live statistics.
- Admin-protected dataset operations (Basic Auth via environment credentials).
- Delete operations for all samples, by label, and one-shot undo by sample ID.
- React label management panel with on-demand admin login modal.
- Dedicated destructive confirmation modal for dataset delete actions.
- WebAssembly inference pipeline integrated with React canvas + network visualizer.
- DevContainer, Docker Compose stack, and CI workflow baseline.

### Changed
- `Clear Drawing` now resets canvas and neural state without hiding the network panel.
- Empty canvas UX now shows neutral prediction/confidence (`-`) and disables sample save.
- README updated with API quick reference and current auth behavior.

### Security
- Sensitive runtime values are sourced from environment variables (`.env` / `.env.example`).
- `.env` remains ignored by git to avoid accidental secret commits.
