"""PdfDownloader: Drive URL rewriting, confirm tokens, and PDF verification.

The last of those is the legacy bug this guards: an HTML interstitial returned
with HTTP 200 used to be handed downstream as "PDF bytes".
"""

from __future__ import annotations

import httpx
import pytest

from config.settings import Settings
from domain.exceptions import PDFDownloadError
from infrastructure.pdf.downloader import (
    PdfDownloader,
    _confirm_token,
    _direct_url,
    _looks_like_pdf,
)

PDF_BYTES = b"%PDF-1.7\n%%EOF\n"

INTERSTITIAL = b"""<!DOCTYPE html><html><body>
<form id="download-form" action="https://drive.usercontent.google.com/download">
<input type="hidden" name="confirm" value="t-abc123">
</form></body></html>"""


def settings() -> Settings:
    return Settings(
        gemini_api_key="test",
        zilliz_cloud_endpoint="https://example.zillizcloud.com",
        zilliz_cloud_api_key="test",
    )


class TestDirectUrl:
    def test_rewrites_a_drive_share_link(self):
        target = _direct_url("https://drive.google.com/file/d/1zM208Xqjvmxh1HF/view")
        assert target.startswith("https://drive.usercontent.google.com/download?id=1zM208Xqjvmxh1HF")

    def test_rewrites_a_query_style_drive_link(self):
        target = _direct_url("https://drive.google.com/uc?export=download&id=1zM208Xqjvmxh1HF")
        assert "id=1zM208Xqjvmxh1HF" in target
        assert "drive.usercontent.google.com" in target

    def test_passes_through_a_plain_url(self):
        url = "https://hcmut.edu.vn/files/quy-che.pdf"
        assert _direct_url(url) == url

    def test_passes_through_a_drive_url_with_no_usable_id(self):
        url = "https://drive.google.com/drive/folders"
        assert _direct_url(url) == url


class TestPdfDetection:
    def test_accepts_a_pdf_header(self):
        assert _looks_like_pdf(PDF_BYTES)

    def test_accepts_a_header_after_leading_junk(self):
        assert _looks_like_pdf(b"\n\n\x00" + PDF_BYTES)

    def test_rejects_html(self):
        assert not _looks_like_pdf(INTERSTITIAL)

    def test_rejects_empty(self):
        assert not _looks_like_pdf(b"")


class TestConfirmToken:
    def test_extracts_the_token(self):
        assert _confirm_token(INTERSTITIAL) == "t-abc123"

    def test_absent_token_is_empty(self):
        assert _confirm_token(b"<html>no form here</html>") == ""


class TestDownload:
    async def test_returns_pdf_bytes(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
        downloader = PdfDownloader(settings(), transport=transport)
        assert await downloader.download("https://example.com/a.pdf") == PDF_BYTES

    async def test_follows_the_confirm_token(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.params.get("confirm", ""))
            if len(seen) == 1:
                return httpx.Response(200, content=INTERSTITIAL)
            return httpx.Response(200, content=PDF_BYTES)

        downloader = PdfDownloader(settings(), transport=httpx.MockTransport(handler))
        data = await downloader.download("https://drive.google.com/file/d/1zM208Xqjvmxh1HF/view")
        assert data == PDF_BYTES
        assert seen[-1] == "t-abc123"

    async def test_rejects_html_masquerading_as_a_pdf(self):
        """HTTP 200 is not proof of a PDF."""
        body = b"<html><body>Sign in to continue</body></html>"
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
        downloader = PdfDownloader(settings(), transport=transport)
        with pytest.raises(PDFDownloadError, match="did not return a PDF"):
            await downloader.download("https://drive.google.com/file/d/1zM208Xqjvmxh1HF/view")

    async def test_raises_on_http_error(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(403))
        downloader = PdfDownloader(settings(), transport=transport)
        with pytest.raises(PDFDownloadError, match="403"):
            await downloader.download("https://example.com/a.pdf")

    async def test_raises_on_transport_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out")

        downloader = PdfDownloader(settings(), transport=httpx.MockTransport(handler))
        with pytest.raises(PDFDownloadError):
            await downloader.download("https://example.com/a.pdf")
