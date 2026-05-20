# Project-Specific Mandates

This file defines the foundational mandates and workflows for the `bubble_analyser` project. These instructions take absolute precedence over general defaults.

## Validation & Compliance Workflow

To prevent CI failures on GitHub, the following steps are MANDATORY before declaring any task "complete" or suggesting a push:

1.  **Compliance Fixing:** After any significant modification to the codebase, you MUST invoke the `compliance-fixer` subagent to ensure code adheres to Ruff and MyPy standards.
2.  **Linting Verification:** Manually run `ruff check .` to verify no style or logical errors remain.
3.  **Type Verification:** Manually run `mypy .` to verify type safety across the entire project.
4.  **Testing Verification:** Manually run `python -m pytest tests/` to ensure no regressions were introduced.
5.  **CI Pre-checks:** Ensure all files have a single newline at the end and no trailing whitespace.

## Technical Standards

- **Color Spaces:** Always prefer `cv2` for color space conversions and I/O.
- **Deep Learning:** Ensure all image arrays passed to the Mask R-CNN model are explicitly cast to `np.float32`.
- **Modularity:** Maintain the separation between the `gui`, `core` (domain/services), and `processing` layers.
