# Chat Reply Recommendation System

This repository contains an offline chat-reply recommendation system based on fine-tuning GPT-2.

Contents
- `ChatRec_Model.ipynb` — Jupyter notebook with preprocessing, tokenization, training, generation, and evaluation.
- `conversationfile.xlsx` — Combined conversation data (CSV-format) used for testing.
- `run_train.py` — Guarded script to run offline training (attempts local GPT-2; aborts if not available).
- `run_smoke_test.py` — Quick preprocessing smoke test that builds context→A pairs and checks environment.
- `create_model_joblib.py` — Creates a placeholder `Model.joblib` pipeline package.
- `Model.joblib` — Placeholder model package (committed for submission/testing).
- `ReadMe.txt` — Original project readme (legacy)
- `Report.pdf` — Technical report

Quick start
1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Edit paths in `ChatRec_Model.ipynb` or set `COMBINED_PATH`/`USE_COMBINED` to your combined data file.

3. For a fast check run:

```powershell
python run_smoke_test.py
```

4. To run training (offline only if GPT-2 exists locally):

```powershell
python run_train.py
```

If you want to push this repo to GitHub, follow the commands below (after creating a remote repository on GitHub):

```powershell
git remote add origin https://github.com/<your-username>/meetmux-live-coding-round.git
git branch -M main
git push -u origin main
```

Notes
- Offline training requires pretrained GPT-2 files in the local Hugging Face cache. If not available, the notebook will either fall back to downloading (if allowed) or the training script will abort.
- `DEV_RUN=True` runs a tiny subset for quick end-to-end validation.

Contact
For questions or help running the pipeline, reply in the issue/PR or contact me directly.
