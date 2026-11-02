"""Download an image from a URL safely (scheme check, SSRF guard, size cap)."""

from __future__ import annotations

import ipaddress
import mimetypes
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from .config import get_settings


class FetchError(RuntimeError):
    pass


def _check_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FetchError("Sono accettati solo URL http(s) completi")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise FetchError(f"Host non risolvibile: {parsed.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchError("Indirizzi di rete locali/privati non consentiti")


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_image(url: str, dest_dir: str | Path | None = None) -> Path:
    settings = get_settings()
    url = (url or "").strip()
    _check_host(url)
    limit = settings.max_image_mb * 1024 * 1024
    request = Request(url, headers={"User-Agent": settings.user_agent, "Accept": "image/*,*/*;q=0.5"})
    opener = build_opener(_SafeRedirect)
    try:
        with opener.open(request, timeout=settings.http_timeout_s * 2) as response:
            data = response.read(limit + 1)
            content_type = response.headers.get_content_type()
    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"Download fallito: {exc}") from exc
    if len(data) > limit:
        raise FetchError(f"File oltre il limite di {settings.max_image_mb} MB")

    name = Path(urlparse(url).path).name or "immagine"
    suffix = Path(name).suffix.lower() or mimetypes.guess_extension(content_type or "") or ".img"
    target_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="geoosint_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / (Path(name).stem[:60] + suffix)
    path.write_bytes(data)
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise FetchError("Il contenuto scaricato non è un'immagine valida") from exc
    return path
