"""Create an explicit README catalog from the verified jbackk-lang repository set."""
from __future__ import annotations

import json
from pathlib import Path


REPOSITORIES = (
    "analizator-gieldowy", "analizator-gieldowy-2.0", "Analizator_Gieldowy_v3.0",
    "Architektura-Mapowania-Zmyslowego-TIMDR", "Boundary-Matter", "deliverable_timdr_finanse",
    "EasySound", "FIELDCORE", "FLIGHT-TRACKING-TIMDR", "genertor-fotonow", "GIA-TIMDR",
    "GSF", "Helix-Astro", "Helix-Lock", "jbackk-lang.github.io", "KHIPU", "KHIPU-NEURAL",
    "MAGE-IN-IMAGE-DECODER", "math-validator-3.0", "math-validator-v2.0", "PC_TIMDR",
    "phi-fiber-dsp", "phi-topology-filter", "probabilistic-timdr", "RADAR-TRACKING",
    "RADAR-TRACKING-TIMDR", "Senscore", "SYNOPTYK-ARCTIC", "synoptyk-v2.0", "Synoptyk-v3",
    "THE_TIMDR_Hyperflow_Engine", "TIMDR-AI-Core", "TIMDR-Aviation-Diagnostics",
    "TIMDR-Battery-Predict", "TIMDR-Bio-Signals", "TIMDR-Concept-Archive",
    "TIMDR-Cosmology-Filters", "TIMDR-Crypto-Graph", "TIMDR-Cyclone-RI", "TIMDR-DNA",
    "TIMDR-Earthquake-Core", "TIMDR-Echosonda-3D", "TIMDR-EV-Predict", "TIMDR-fusion-tools",
    "TIMDR-Geometry-Formalism", "TIMDR-Grid-Monitor", "TIMDR-Industrial-Predict",
    "TIMDR-Materials-Design", "TIMDR-Math-Formalism", "TIMDR-META-DYNAMICS",
    "TIMDR-Modal-Formalism", "TIMDR-Mold-Risk", "TIMDR-Multisensory-Meditation-Engine",
    "TIMDR-Radar-Module", "TIMDR-Robot", "TIMDR-Security-Module", "TIMDR-Solar-PV",
    "TIMDR-Sygnalizacja", "TIMDR-Time-Formalism", "TIMDR-Tornado-NEXRAD", "topologic",
    "universal-state-analyzer",
)


def build_catalog() -> dict:
    return {
        "items": [
            {
                "id": f"jbackk-lang-{name.lower()}",
                "url": f"https://raw.githubusercontent.com/jbackk-lang/{name}/main/README.md",
                "title": f"jbackk-lang/{name} README",
            }
            for name in REPOSITORIES
        ]
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "online_catalogs.personal_repos.json"
    output.write_text(json.dumps(build_catalog(), indent=2) + "\n", encoding="utf-8")
    print(f"Created {output.name} with {len(REPOSITORIES)} repositories.")
