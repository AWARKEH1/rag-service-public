"""Download the Service-Public.gouv.fr fiches (DILA, Licence Ouverte 2.0) and unpack the XML files.

    python -m rag_sp.download            # -> data/raw/*.xml
"""

import argparse
import io
import os
import shutil
import urllib.request
import zipfile

from .config import get_settings


def unpack(payload: bytes, out_dir: str) -> int:
    """Write the XML files contained in `payload` (a zip archive or a single XML file). Returns the count."""
    os.makedirs(out_dir, exist_ok=True)
    if payload[:2] == b"PK":
        n = 0
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if not base.lower().endswith(".xml"):
                    continue
                with zf.open(name) as src, open(os.path.join(out_dir, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
        return n
    with open(os.path.join(out_dir, "fiches.xml"), "wb") as f:
        f.write(payload)
    return 1


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=s.source_url)
    ap.add_argument("--out", default=os.path.join(s.data_dir, "raw"))
    args = ap.parse_args()
    req = urllib.request.Request(args.url, headers={"User-Agent": "rag-service-public (portfolio project)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = r.read()
    print(f"{unpack(payload, args.out)} XML file(s) written to {args.out}")


if __name__ == "__main__":
    main()
