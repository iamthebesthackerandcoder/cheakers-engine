# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Tight coupling: GUI/training import legacy [`gameotherother.py`](gameotherother.py) directly.
- checkers/ facade over engine; avoid refactoring without updating imports.
- Neural eval assumes stateless caching; shared TT in multiprocessing.
- In-place weight updates risk races during parallel self-play/training.
- Monorepo: root scripts integrate checkers/ and neural components.
- No DB; forward-only data serialization for models/checkpoints.
