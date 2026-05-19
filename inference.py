"""Compatibility wrapper.

Do not edit/use this file directly. The real PUMA Docker inference entrypoint is
infer_wsi.py, and inference.sh calls infer_wsi.py.
"""

from infer_wsi import main


if __name__ == "__main__":
    main()
