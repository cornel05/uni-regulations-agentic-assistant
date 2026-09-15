"""PDF downloader with Google Drive support.

The legacy downloader hit ``drive.google.com/uc?export=download`` and treated any
200 response as PDF bytes. For a large or permission-gated file that response is
an HTML virus-scan interstitial, which then reached PyMuPDF as a "PDF". Here the
current ``drive.usercontent.google.com`` endpoint is used, the confirm token is
followed when one appears, and the payload is verified to actually be a PDF.
"""

from __future__ import annotations

import logging
import re

import httpx

from config.settings import Settings
from domain.exceptions import PDFDownloadError

logger = logging.getLogger(__name__)

_DRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")
_DRIVE_FILE_ID = re.compile(r"/d/([a-zA-Z0-9_-]{10,})")
_DRIVE_QUERY_ID = re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})")
_CONFIRM_TOKEN = re.compile(r"name=\"confirm\"\s+value=\"([^\"]+)\"")

_PDF_MAGIC = b"%PDF"
# Some generators emit a few junk bytes before the header, so scan a small window.
_MAGIC_WINDOW = 1024
_MAX_BYTES = 100 * 1024 * 1024


class PdfDownloader:
    """Concrete PDFDownloaderPort over httpx."""

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._timeout = settings.pdf_download_timeout
        # Injectable so the confirm-token and not-a-PDF paths can be tested
        # offline; production passes nothing and gets the default transport.
        self._transport = transport

    async def download(self, url: str) -> bytes:
        target = _direct_url(url)
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True, transport=self._transport
            ) as client:
                response = await client.get(target)
                data = _payload(response, url)

                if not _looks_like_pdf(data):
                    token = _confirm_token(data)
                    if token:
                        logger.debug("Following Drive confirm token for %s", url)
                        response = await client.get(target, params={"confirm": token})
                        data = _payload(response, url)
        except httpx.HTTPError as exc:
            raise PDFDownloadError(f"Could not fetch {url}: {exc}") from exc

        if not _looks_like_pdf(data):
            raise PDFDownloadError(
                f"{url} did not return a PDF (got {len(data)} bytes of "
                f"{response.headers.get('content-type', 'unknown type')}) — "
                "the file may be private or require sign-in"
            )
        logger.debug("Downloaded %d bytes from %s", len(data), url)
        return data


def _payload(response: httpx.Response, url: str) -> bytes:
    if response.status_code != 200:
        raise PDFDownloadError(f"{url} returned HTTP {response.status_code}")
    data = response.content
    if len(data) > _MAX_BYTES:
        raise PDFDownloadError(f"{url} is larger than {_MAX_BYTES // (1024 * 1024)} MB")
    return data


def _direct_url(url: str) -> str:
    """Rewrite a Drive share link into its download endpoint; pass others through."""
    if not any(host in url for host in _DRIVE_HOSTS):
        return url
    file_id = _drive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _drive_file_id(url: str) -> str:
    for pattern in (_DRIVE_FILE_ID, _DRIVE_QUERY_ID):
        match = pattern.search(url)
        if match:
            return match.group(1)
    return ""


def _looks_like_pdf(data: bytes) -> bool:
    return _PDF_MAGIC in data[:_MAGIC_WINDOW]


def _confirm_token(data: bytes) -> str:
    match = _CONFIRM_TOKEN.search(data[:100_000].decode("utf-8", "ignore"))
    return match.group(1) if match else ""
