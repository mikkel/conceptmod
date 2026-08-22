from conceptmod.backends.base import Backend

# One process, one backend. Switching --backend must not change the others.
BACKENDS = ("sana", "zimage", "anima", "krea", "qwen")


def load_backend(name: str, device: str, **kwargs) -> Backend:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if name == "sana":
        from conceptmod.backends.sana import SanaBackend

        return SanaBackend(device=device, **kwargs)
    if name == "zimage":
        from conceptmod.backends.zimage import ZImageBackend

        return ZImageBackend(device=device, **kwargs)
    if name == "anima":
        from conceptmod.backends.anima import AnimaBackend

        return AnimaBackend(device=device, **kwargs)
    if name == "krea":
        from conceptmod.backends.krea import KreaBackend

        return KreaBackend(device=device, **kwargs)
    if name == "qwen":
        from conceptmod.backends.qwen import QwenBackend

        return QwenBackend(device=device, **kwargs)
    if name == "sdxl":
        raise ValueError(
            "SDXL is conceptmod 1.x (UNet + CLIP). "
            "2.0 backends are flow-matching DiTs: "
            + ", ".join(BACKENDS)
        )
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
