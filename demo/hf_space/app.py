# Hugging Face Space bootstrap: fetch the ACRS code from GitHub and launch its demo.
import io, sys, tarfile, urllib.request
from pathlib import Path

REF = "main"  # pin a commit hash here for an archived, reproducible demo
dest = Path("/tmp/acrs")
if not (dest / "demo/space_app.py").exists():
    data = urllib.request.urlopen(f"https://codeload.github.com/joyjeni/acrs-appworld/tar.gz/{REF}").read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        root = tf.getnames()[0].split("/")[0]
        tf.extractall("/tmp")
    Path(f"/tmp/{root}").rename(dest)
sys.path.insert(0, str(dest / "demo"))
import space_app  # noqa: E402

space_app.build_ui().launch()
