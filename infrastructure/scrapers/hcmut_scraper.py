"""Selenium crawler for the university regulation listing page.

The listing is rendered client-side, so a headless browser is still the pragmatic
way in. Two changes from the legacy crawler: this is the single copy (the same
driver setup and selector were duplicated in three modules), and it waits for the
links to appear instead of sleeping a fixed eight seconds and hoping.
"""

from __future__ import annotations

import asyncio
import logging

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from config.settings import Settings
from domain.exceptions import ScraperError
from domain.models import RegulationSource

logger = logging.getLogger(__name__)

_LINK_SELECTOR = "div.sub-link a"
_DOCUMENT_HOSTS = ("drive.google.com", "docs.google.com")


class HcmutScraper:
    """Concrete ScraperPort over headless Chrome."""

    def __init__(self, settings: Settings) -> None:
        self._url = settings.regulation_page_url
        self._headless = settings.headless_mode
        self._wait = settings.wait_time

    async def scrape(self) -> list[RegulationSource]:
        """Selenium is blocking, so it runs off the event loop."""
        return await asyncio.to_thread(self._scrape_blocking)

    def _scrape_blocking(self) -> list[RegulationSource]:
        driver = self._start_driver()
        try:
            driver.get(self._url)
            WebDriverWait(driver, self._wait).until(
                expected_conditions.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, _LINK_SELECTOR)
                )
            )
            elements = driver.find_elements(By.CSS_SELECTOR, _LINK_SELECTOR)
        except TimeoutException as exc:
            raise ScraperError(
                f"No links matched {_LINK_SELECTOR!r} on {self._url} within "
                f"{self._wait}s — the page layout may have changed"
            ) from exc
        except WebDriverException as exc:
            raise ScraperError(f"Could not load {self._url}: {exc}") from exc
        else:
            sources = _collect(elements)
            logger.info("Crawled %d regulation links from %s", len(sources), self._url)
            return sources
        finally:
            driver.quit()

    def _start_driver(self) -> webdriver.Chrome:
        options = Options()
        if self._headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        try:
            return webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )
        except WebDriverException as exc:
            raise ScraperError(f"Could not start Chrome: {exc}") from exc


def _collect(elements) -> list[RegulationSource]:
    """Keep document links only, first title wins for a repeated URL."""
    found: dict[str, RegulationSource] = {}
    for element in elements:
        try:
            href = element.get_attribute("href") or ""
            title = (element.text or "").strip() or (
                element.get_attribute("textContent") or ""
            ).strip()
        except WebDriverException:
            # A stale element mid-scrape costs one link, not the whole crawl.
            continue

        if not href or not any(host in href for host in _DOCUMENT_HOSTS):
            continue
        found.setdefault(href, RegulationSource(title=title or href, link=href))
    return list(found.values())
