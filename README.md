Fish tracking
=============

Make a wheel:s
```bash
pip install build
python -m build
```

Install the package:
```bash
pip install --find-links ~/dev/tracking/dist fish_tracking -r ~/dev/tracking/requirements.txt
```

Install only requirements:
```bash
pip install -r requirements.txt