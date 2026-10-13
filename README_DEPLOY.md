# BarnacleAI web deployment status

## Public browser prototype

The repository's `index.html` contains an interactive static demo that runs entirely in the browser. It supports image upload, hand-engineered COSC 3 image features, an exploratory *untrained* candidate detector, and manually entered counts with a sampling-based interval.

This browser site **does not run the trained PyTorch model**. It intentionally distinguishes the experimental heuristic from validated CNN inference.

To preview the site from GitHub, use the HTMLPreview tool with this URL:

https://htmlpreview.github.io/?https://raw.githubusercontent.com/jerelclark2-bit/Barnacle-A.I/main/index.html

## Full neural-network website

The complete Python project is packaged separately as `BarnacleAI_live_deploy_ready.zip` in the originating ChatGPT conversation. Its Streamlit entrypoint is `app.py` and its trained CNN and Ridge artifacts are in `results/`. The complete application still needs deployment on a Python host such as Render or Streamlit Community Cloud.

For actual publication, upload the extracted contents of `BarnacleAI/` to the repository and deploy using `render.yaml` or the Streamlit entrypoint. Ensure package dependencies and checkpoints are available on the server. Test an uploaded image and check server logs before distributing a public URL.

## Scientific qualification

CNN detections and tile MAE were evaluated against incomplete annotation outlines, **not** independently verified biological counts. A statistically defensible population count interval requires manual ground-truth review.