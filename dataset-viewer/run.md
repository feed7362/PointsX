# Run dataset viewer

```bash
cd dataset-viewer
source .venv/bin/activate
./run.sh
```

You should see **Submissions**, a **Refresh** button, and records in the list. Click a row to open details.

## Blank window on macOS?

The venv may be using **Xcode system Python**, whose Tk is broken/deprecated.

Recreate the venv with Homebrew Python:

```bash
brew install python@3.12 python-tk@3.12
cd dataset-viewer
rm -rf .venv
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```
