# scripts/package_manifest.py
import json, os, hashlib, shutil, sys

root = "reports"
os.makedirs(root, exist_ok=True)
manifest = {"files": []}

for fn in sorted(os.listdir(root)):
    p = os.path.join(root, fn)
    if os.path.isfile(p):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        manifest["files"].append({"name": fn, "sha256": h})

with open(os.path.join(root, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

shutil.make_archive("submission_envelope", "zip", ".")
print("Created submission_envelope.zip")
