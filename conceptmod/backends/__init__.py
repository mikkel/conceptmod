from conceptmod.backends.base import Backend

BACKENDS = ("sana", "zimage", "anima", "krea")


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
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
