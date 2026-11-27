# NeuroStudio: Open Source Product Foundation

**Status:** product proposal for the existing TRIBE v2 experiments repository  
**Working name:** NeuroStudio (already used by the prototype UI; name can change)

## Product

NeuroStudio is a local-first tool for exploring time-aligned model predictions about brain responses to media. A person can load a supported stimulus or an existing prediction bundle, inspect the timeline, compare regions or model runs, search for peaks and patterns, and export the analysis.

It should meet people at different levels:

- **Explore:** open the interface, choose a sample or local file, and inspect the timeline without writing code.
- **Run:** use a short CLI command to process a local stimulus and save a portable result bundle.
- **Research:** use documented Python APIs and notebooks to inspect event timing, model outputs, parameters, and exports.

The scope can grow across modalities and backends. “Any input” is a long-term adapter goal, not a claim that the current model supports every medium or produces ground-truth neural measurements.

## Scientific language and limits

The current TRIBE v2 backend predicts fMRI responses from stimulus features, using an average-subject representation. Its output is a model prediction, not a recording of the person currently reading or watching. Text timing generated from a word-per-minute setting is estimated timing, not measured reading behavior. UI labels and exports must preserve this distinction and include backend, model version, input modality, timing assumptions, atlas, temporal resolution, and preprocessing metadata.

Avoid treating a high prediction as proof that a passage is engaging, emotional, memorable, or good. These are questions users may investigate; the software should expose the signal and method so they can form and test hypotheses.

## Current repository facts

This repository currently combines upstream TRIBE v2 code and documentation with a prototype Streamlit viewer, custom inference and analysis scripts, notebooks, and generated results. The top-level README and package metadata still identify the project as TRIBE v2 and point to Meta's repository. The current package requirements also do not include all dependencies used by the viewer.

The repository's current license is CC BY-NC 4.0. That license is not an OSI-approved open source license and restricts commercial use. Do not replace it with MIT, Apache-2.0, or another permissive license for the whole repository: the repository includes upstream code and model components that are not ours to relicense. The TRIBE v2 backend and weights must remain clearly attributed and subject to their own terms. Before calling the full repository “open source,” separate original NeuroStudio code from upstream components and audit each dependency, model, dataset, and result artifact. This document is a product plan, not legal advice.

Do not add books, films, music, or other third-party media to the repository as example inputs unless redistribution rights are clear. Use a short, openly licensed or creator-provided fixture. Keep large local inputs and generated bundles out of Git.

## First usable release (0.1)

A newcomer should be able to complete this path:

1. Install the lightweight viewer without downloading a foundation model.
2. Open a clearly licensed example prediction bundle.
3. Move through a stimulus timeline and see aligned text or media context, region traces, and model metadata.
4. Find high and low intervals with a transparent, user-configurable score.
5. Export selected time ranges and analysis tables.
6. Follow an optional advanced path to run inference with a supported backend.

This makes the visualization and analysis useful even when a user has no compatible GPU or does not accept the backend's model license.

### 0.1 acceptance checklist

- One clean installation path for the viewer; core, GUI, and inference dependencies are separated.
- A tiny, openly redistributable example bundle and a visible “load your own bundle” path.
- A documented bundle schema with timestamps, aligned context, region identifiers, values, and provenance.
- The app starts from a documented command and reports missing files/dependencies with actionable messages.
- Peak/range results can be reproduced from documented settings and exported to CSV/JSON.
- UI, CLI, and exports call predictions “predicted responses” and display timing/model provenance.
- No credentials, private data, or large generated outputs are required for the basic walkthrough.

## Architecture direction

Keep the product small and make each layer replaceable:

- **Bundle format:** versioned, portable container for aligned stimulus metadata, prediction arrays, region labels, and provenance. Validate on load.
- **Analysis library:** pure functions for smoothing, region selection, weighted composites, interval detection, comparison, and export.
- **Backend adapters:** a narrow interface that turns supported inputs into a bundle. TRIBE v2 is one optional adapter; other models can be added without changing the viewer.
- **CLI:** discoverable commands for validating, viewing, analyzing, and exporting bundles; inference is an optional command group.
- **GUI:** a beginner-friendly front end over the same analysis functions, with controls that reveal their assumptions.
- **Research API:** documented Python functions and notebooks for custom studies and reproducible workflows.

Do not build accounts, hosted processing, collaboration systems, contracts, or a plugin marketplace for 0.1. Local files and an explicit backend are enough to prove the workflow.

## Suggested repository shape

~~~text
src/neurostudio/
  bundles/       # schema, loading, validation, provenance
  analysis/      # reusable signal and interval operations
  backends/      # optional model adapters
  cli.py
  ui/            # GUI entry point
examples/
  tiny-open-bundle/
docs/
  PRODUCT_PLAN.md
  SCIENTIFIC_SCOPE.md
  BUNDLE_FORMAT.md
  INSTALLATION.md
~~~

Keep the existing upstream TRIBE v2 package in a clearly attributed, separately identifiable area until its license and packaging boundaries are resolved. Do not copy its README or metadata as NeuroStudio product documentation.

## Build sequence

1. **Clean boundary:** inventory upstream-derived and original files; record licenses for code, models, data, and outputs. Identify third-party media and large artifacts that should not be part of a distributable example.
2. **Runnable viewer:** make the existing UI's dependencies explicit; load and validate one small demo bundle; add clear empty-state and error messages.
3. **Reusable core:** move analysis operations out of the UI into importable functions and add schema/version metadata to outputs.
4. **Simple onboarding:** document GUI use first, then CLI, then API/notebook workflows. Include a no-GPU path.
5. **Inference adapter:** add TRIBE v2 as an optional backend with explicit model/license requirements, supported modalities, memory expectations, and timing behavior.
6. **Broader research use:** add additional backends and formats only when they can provide provenance and reproducible validation.

## Out of scope until the first release works

- Training a new foundation model.
- Claiming universal modality support.
- Inferring an individual's actual brain state from an unmeasured stimulus exposure.
- Automatically ranking creative quality or audience reaction as objective truth.
- Uploading private media or predictions to a hosted service.
- Redistributing copyrighted media or model weights without clear permission.
