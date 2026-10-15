# Running BarnacleAI

## Browser demo

**[Open BarnacleAI](https://jerelclark2-bit.github.io/Barnacle-A.I/)**

Upload a JPG or PNG cropped to the inside of the frame. The website shows proposed barnacle centers, a count, a threshold-sensitivity range, and some image features. You can also manually count 12 randomly selected tiles to estimate a total with an approximate sampling interval.

The website uses the **trained 32-epoch CNN**. `inference.js` loads its exported PyTorch weights from `model_weights.json` and runs inference in your browser using TensorFlow.js. The image stays on your computer. TensorFlow.js itself loads from a CDN, so you'll need an internet connection, at least on first load.

This is experimental. A green circle is a possible barnacle center, **not proof of a barnacle**. The number shown is not a scientifically validated population count. The threshold range is **not** a 95% confidence interval. For the actual study results and limitations, see [RESULTS.md](RESULTS.md).

## Hosting the browser page

This repository is set up for GitHub Pages from the `main` branch, **/(root)**. The site's starting point is `index.html`. Keep these files next to it:

- `inference.js`: browser inference code.
- `model_weights.json`: trained CNN weights.

If you're testing it locally, serve the repository folder instead of double-clicking `index.html`:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Full Python version

The browser demo and the Python app are two frontends for the project. The browser runs the CNN only; the Python code also includes a Ridge regression baseline, analysis scripts, and a Streamlit dashboard. GitHub Pages **cannot run** Streamlit or FastAPI.

To use the Python dashboard locally, with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The API can be started separately with `uvicorn api:app --reload`. The included checkpoints live in `results/`.

Automated tests run on GitHub Actions. The browser-assets test also checks that the exported weights aren't missing or malformed. Passing tests doesn't replace a real image-by-image accuracy evaluation.

## What still needs real research

The dataset has two labeled photographs, and the provided outlines are incomplete. The reported CNN and Ridge errors are against those imperfect annotations, **not independently verified barnacle counts**. To support an interval for a real count, a person must manually check the randomly selected tiles. See [RESULTS.md](RESULTS.md).
