"""CLI for the ComfyUI LoRA convert path.

Implementation lives in ``conceptmod.convert``. This file is a thin wrapper so
existing invocations keep working. Do not add a second converter here.
"""
from conceptmod.convert import *  # noqa: F403
from conceptmod.convert import main

if __name__ == "__main__":
    main()
