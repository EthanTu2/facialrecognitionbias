# Facial recognition & algorithmic bias

Work through `tutorial.ipynb` after you finish the setup below.

## What you need

- **[Git](https://git-scm.com/)**
- **Python 3.10 or newer.** Install from [python.org](https://www.python.org/downloads/) or use [Anaconda](https://www.anaconda.com/) if you prefer a bundle that includes Python and Jupyter tooling.

---

## Tutorial setup
### 1. Clone the repository

Open a **terminal**. Go to whichever directory you use for code projects, then run:

```bash
git clone https://github.com/EthanTu2/facialrecognitionbias.git
cd facialrecognitionbias
```

This downloads the entire project to your local machine.

### 2. Create a virtual environment (`.venv`)

Isolates this project’s packages from other Python projects.

```bash
# macOS / Linux
python3 -m venv .venv

# Windows (PowerShell)
python -m venv .venv
```

That creates a folder named `.venv` inside the repo.

### 3. Activate the virtual environment

You should see `(.venv)` at the start of your shell prompt once this works.

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

If Windows blocks the script (`execution policy`), run PowerShell once as Administrator:

`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

then try activating again.

### 4. Upgrade `pip` and install dependencies

Still **with `.venv` active** (you should see (venv) at the beginning of your prompt in the command line):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Register a Jupyter kernel for this project

Still inside the activated `.venv`, make a kernelspec:

```bash
python -m ipykernel install --user --name facialrecognitionbias --display-name "Python (.venv facialrecognitionbias)"
```

- `**facialrecognitionbias**` — kernelspec ID.
- `**Python (.venv facialrecognitionbias)**` — label you see when choosing a kernel in your Integrated Development Environment (IDE).

`ipykernel install` records the full path to this `.venv`’s Python. If you move or copy the project to another folder (or recreate `.venv`), run the `python -m ipykernel install ...` command again from the new environment.

### 6. Use that kernel when you open `tutorial.ipynb`

Confirm that your notebook is tied to `Python (.venv facialrecognitionbias)`.


| Where you work | What to click                                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **VS Code**    | Click on “Select Kernel” at the top right -> pick Python (.venv facialrecognitionbias). Alternatively “Python: Select Interpreter” → `./.venv/bin/python` on Windows. |
| **JupyterLab** | From repo root with `.venv` active: run `jupyter lab`, open `tutorial.ipynb`, then Kernel -> Change kernel -> Python (.venv facialrecognitionbias).                   |


If `ModuleNotFoundError` appears on imports, you are probably running a different Python than the one where you ran `pip install -r requirements.txt`. Reactivate `.venv`, re-run pip if needed, and re-select the kernel / interpreter tied to `./.venv`.

### 7. Run the notebook

That's it! Open `tutorial.ipynb`, check your kernel (step 6), then run cells from top to bottom.