"""Resolve relocated source files without changing historical experiment code or hashes."""
from pathlib import Path
import hashlib
SCRIPTS = Path(__file__).resolve().parents[1]
LOCATIONS = {'download_assets.py': 'setup/download_assets.py', 'step0_smoke.py': 'step0/step0_smoke.py', 'step1_extract.py': 'step1/step1_extract.py', 'step1_probe.py': 'step1/step1_probe.py', 'step1_common.py': 'common/step1_common.py', 'step1_finish.py': 'step1/step1_finish.py', 'step2_train.py': 'step2/step2_train.py', 'step2_evaluate.py': 'step2/step2_evaluate.py', 'step2_compare.py': 'step2/step2_compare.py', 'step3_train.py': 'step3/step3_train.py', 'step3_compare.py': 'step3/step3_compare.py'}

def script_path(path):
    path = Path(path)
    if path.is_file():
        return path
    # Only relocate explicitly registered scripts within the scripts tree.
    if SCRIPTS in path.absolute().parents and path.name in LOCATIONS:
        return SCRIPTS / LOCATIONS[path.name]
    return path

def configure():
    import step1_common
    original = step1_common.fingerprint
    def relocated_fingerprint(path):
        return original(script_path(path))
    step1_common.fingerprint = relocated_fingerprint
