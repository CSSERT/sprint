import requests
from typing import Literal, TypedDict

LookbackDays = Literal["day", "week", "month", "year"]


class Prediction(TypedDict):
    start_index: int
    end_index: int
    text: str
    aspect: str
    attribute: str
    sentiment: str
    scope: str


class Paragraph(TypedDict):
    paragraph: str
    prediction: list[Prediction]


class ParagraphList(TypedDict):
    url: str
    title: str
    paragraph_results: list[Paragraph]


class CrawlAndAnalyzeResponse(TypedDict):
    request_at: str
    ticker: str
    lookback_days: str
    results: list[ParagraphList]


class ViBankACSAPI:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def crawl_and_analyze(
        self,
        ticker: str,
        *,
        lookback_days: LookbackDays = "month",
    ) -> CrawlAndAnalyzeResponse:
        response = self._session.post(
            f"{self._base_url}/crawl_and_analyze",
            json={
                "ticker": ticker,
                "lookback_days": lookback_days,
            },
        )
        response.raise_for_status()
        return response.json()
