from conceptmod.backends.base import Backend


def load_backend(name: str, device: str, **kwargs) -> Backend:
    if name == "sana":
        from conceptmod.backends.sana import SanaBackend

        return SanaBackend(device=device, **kwargs)
    if name == "zimage":
        from conceptmod.backends.zimage import ZImageBackend

        return ZImageBackend(device=device, **kwargs)
    raise ValueError(f"unknown backend {name!r}")
