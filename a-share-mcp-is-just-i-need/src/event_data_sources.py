"""Official announcement and financial-news public data adapters."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.market_data_sources import (
    DEFAULT_HEADERS,
    PublicDataSourceError,
    normalize_a_share_code,
)


class HTTPSession(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...


def _iso_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
    text = str(value).strip().replace("/", "-")
    match = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", text)
    if match:
        year, month, day = map(int, match.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return None


def _json(response: Any, provider: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
        response.encoding = "utf-8"
        payload = response.json()
    except Exception as error:
        raise PublicDataSourceError(provider, f"{provider} returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise PublicDataSourceError(provider, f"{provider} returned a non-object payload")
    return payload


def _normalize_title(value: Any) -> str:
    return BeautifulSoup(html.unescape(str(value or "")), "html.parser").get_text(
        " ", strip=True
    )


@dataclass
class OfficialAnnouncementSource:
    """Cross-check announcements through CNINFO and the listing exchange."""

    session: HTTPSession | None = None
    timeout: float = 15.0

    CNINFO_SEARCH = "https://www.cninfo.com.cn/new/information/topSearch/query"
    CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    CNINFO_FILES = "https://static.cninfo.com.cn/"
    SSE_QUERY = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
    SSE_FILES = "https://www.sse.com.cn/"
    SZSE_QUERY = "https://www.szse.cn/api/disc/announcement/annList"
    SZSE_FILES = "https://disc.static.szse.cn/download"

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _cninfo_token(self, digits: str) -> str:
        try:
            response = self.session.post(
                self.CNINFO_SEARCH,
                data={"keyWord": digits, "maxNum": 10},
                headers={**DEFAULT_HEADERS, "Referer": "https://www.cninfo.com.cn/"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            response.encoding = "utf-8"
            payload = response.json()
        except Exception as error:
            raise PublicDataSourceError("cninfo", "cninfo security lookup failed") from error
        candidates = (
            payload
            if isinstance(payload, list)
            else payload.get("data") or []
            if isinstance(payload, dict)
            else []
        )
        for item in candidates:
            if str(item.get("code") or item.get("zwjc") or "") == digits:
                org_id = item.get("orgId") or item.get("orgid")
                if org_id:
                    return f"{digits},{org_id}"
        # CNINFO accepts a bare stock code on some deployments.  Keep it as a
        # deterministic fallback and surface failures in the aggregate result.
        return digits

    def _cninfo(
        self, code: str, digits: str, exchange: str, start_date: str, end_date: str, top_k: int
    ) -> list[dict[str, Any]]:
        token = self._cninfo_token(digits)
        payload = _json(
            self.session.post(
                self.CNINFO_QUERY,
                data={
                    "pageNum": 1,
                    "pageSize": top_k,
                    "column": "sse" if exchange == "sh" else "szse",
                    "tabName": "fulltext",
                    "plate": "sh" if exchange == "sh" else "sz",
                    "stock": token,
                    "searchkey": "",
                    "secid": "",
                    "category": "",
                    "trade": "",
                    "seDate": f"{start_date}~{end_date}",
                    "sortName": "time",
                    "sortType": "desc",
                    "isHLtitle": "true",
                },
                headers={
                    **DEFAULT_HEADERS,
                    "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=self.timeout,
            ),
            "cninfo",
        )
        records = []
        for item in payload.get("announcements") or []:
            records.append(
                {
                    "symbol": code,
                    "date": _iso_date(item.get("announcementTime")),
                    "title": _normalize_title(item.get("announcementTitle")),
                    "source": "cninfo",
                    "url": urljoin(self.CNINFO_FILES, str(item.get("adjunctUrl") or "")),
                    "announcement_id": str(item.get("announcementId") or ""),
                }
            )
        return records

    def _sse(
        self, code: str, digits: str, start_date: str, end_date: str, top_k: int
    ) -> list[dict[str, Any]]:
        payload = _json(
            self.session.get(
                self.SSE_QUERY,
                params={
                    "isPagination": "true",
                    "productId": digits,
                    "keyWord": "",
                    "securityType": "0101,120100,020100,020200,120200",
                    "reportType2": "DQGG",
                    "reportType": "ALL",
                    "beginDate": start_date,
                    "endDate": end_date,
                    "pageHelp.pageSize": top_k,
                    "pageHelp.pageCount": 1,
                    "pageHelp.pageNo": 1,
                    "pageHelp.beginPage": 1,
                    "pageHelp.cacheSize": 1,
                },
                headers={**DEFAULT_HEADERS, "Referer": "https://www.sse.com.cn/"},
                timeout=self.timeout,
            ),
            "sse",
        )
        records = []
        for item in payload.get("result") or []:
            path = item.get("URL") or item.get("url") or ""
            records.append(
                {
                    "symbol": code,
                    "date": _iso_date(item.get("SSEDATE") or item.get("publishDate")),
                    "title": _normalize_title(item.get("TITLE") or item.get("title")),
                    "source": "sse",
                    "url": urljoin(self.SSE_FILES, str(path)),
                    "announcement_id": str(item.get("BULLETIN_TYPE") or path),
                }
            )
        return records

    def _szse(
        self, code: str, digits: str, start_date: str, end_date: str, top_k: int
    ) -> list[dict[str, Any]]:
        payload = _json(
            self.session.post(
                self.SZSE_QUERY,
                json={
                    "seDate": [start_date, end_date],
                    "stock": [digits],
                    "channelCode": ["listedNotice_disc"],
                    "pageSize": top_k,
                    "pageNum": 1,
                },
                headers={
                    **DEFAULT_HEADERS,
                    "Content-Type": "application/json",
                    "Referer": "https://www.szse.cn/disclosure/listed/notice/index.html",
                },
                timeout=self.timeout,
            ),
            "szse",
        )
        records = []
        for item in payload.get("data") or payload.get("announceList") or []:
            path = item.get("attachPath") or item.get("attachUrl") or item.get("url") or ""
            url = str(path)
            if url and not url.startswith("http"):
                url = urljoin(self.SZSE_FILES + "/", url.lstrip("/"))
            records.append(
                {
                    "symbol": code,
                    "date": _iso_date(item.get("publishTime") or item.get("publishDate")),
                    "title": _normalize_title(item.get("title") or item.get("announcementTitle")),
                    "source": "szse",
                    "url": url,
                    "announcement_id": str(item.get("id") or path),
                }
            )
        return records

    def announcements(
        self, code: str, start_date: str, end_date: str, top_k: int = 20
    ) -> dict[str, Any]:
        canonical, digits, exchange = normalize_a_share_code(code)
        providers = [
            ("cninfo", lambda: self._cninfo(canonical, digits, exchange, start_date, end_date, top_k)),
            (
                "sse" if exchange == "sh" else "szse",
                lambda: self._sse(canonical, digits, start_date, end_date, top_k)
                if exchange == "sh"
                else self._szse(canonical, digits, start_date, end_date, top_k),
            ),
        ]
        items: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        succeeded: list[str] = []
        for provider, fetch in providers:
            try:
                records = fetch()
                items.extend(records)
                succeeded.append(provider)
            except Exception as error:
                failures.append(
                    {"provider": provider, "error": type(error).__name__, "message": str(error)}
                )

        deduped: list[dict[str, Any]] = []
        seen: dict[tuple[str | None, str], dict[str, Any]] = {}
        for item in sorted(
            items,
            key=lambda row: (str(row.get("date") or ""), str(row.get("title") or "")),
            reverse=True,
        ):
            key = (item.get("date"), re.sub(r"\s+", "", item.get("title") or ""))
            if not key[1]:
                continue
            if key in seen:
                seen[key].setdefault("corroborated_by", []).append(
                    {
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "announcement_id": item.get("announcement_id"),
                    }
                )
                continue
            item["corroborated_by"] = []
            seen[key] = item
            deduped.append(item)
        if not deduped:
            raise PublicDataSourceError(
                "official_announcements",
                "CNINFO and exchange sources returned no usable announcements",
            )
        return {
            "symbol": canonical,
            "start_date": start_date,
            "end_date": end_date,
            "source_chain": succeeded,
            "source_failures": failures,
            "items": deduped[:top_k],
        }


@dataclass
class FinancialNewsSource:
    """CLS-first financial news with Sina used only to fill a shortfall."""

    session: HTTPSession | None = None
    timeout: float = 15.0

    CLS_SEARCH = "https://www.cls.cn/searchPage"
    SINA_ROLL = "https://feed.mix.sina.com.cn/api/roll/get"
    _window_fingerprints: dict[tuple[str, str], tuple[tuple[str, str], frozenset[str]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _cls(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            response = self.session.get(
                self.CLS_SEARCH,
                params={"keyword": query},
                headers={**DEFAULT_HEADERS, "Referer": "https://www.cls.cn/"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            response.encoding = "utf-8"
        except Exception as error:
            raise PublicDataSourceError("cls", "failed to retrieve CLS search page") from error

        soup = BeautifulSoup(response.text, "html.parser")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchor in soup.select('a[href*="/detail/"], a[href*="/telegraph"]'):
            title = anchor.get_text(" ", strip=True)
            href = anchor.get("href") or ""
            if len(title) < 6 or href in seen:
                continue
            parent_text = anchor.parent.get_text(" ", strip=True) if anchor.parent else title
            date_match = re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", parent_text)
            records.append(
                {
                    "date": _iso_date(date_match.group(0)) if date_match else None,
                    "title": title,
                    "summary": parent_text[:500],
                    "source": "cls",
                    "url": urljoin("https://www.cls.cn/", href),
                }
            )
            seen.add(href)
            if len(records) >= top_k:
                break

        # Some CLS deployments embed server-rendered search results as JSON.
        if not records:
            script = soup.find("script", id="__NEXT_DATA__")
            if script and script.string:
                try:
                    payload = json.loads(script.string)
                except json.JSONDecodeError:
                    payload = {}

                def walk(value: Any) -> None:
                    if len(records) >= top_k:
                        return
                    if isinstance(value, dict):
                        title = value.get("title") or value.get("content")
                        item_id = value.get("id") or value.get("article_id")
                        if isinstance(title, str) and item_id and len(title) >= 6:
                            records.append(
                                {
                                    "date": _iso_date(value.get("ctime") or value.get("time")),
                                    "title": _normalize_title(title),
                                    "summary": _normalize_title(value.get("brief") or title)[:500],
                                    "source": "cls",
                                    "url": f"https://www.cls.cn/detail/{item_id}",
                                }
                            )
                        for nested in value.values():
                            walk(nested)
                    elif isinstance(value, list):
                        for nested in value:
                            walk(nested)

                walk(payload)
        return records[:top_k]

    def _sina(self, query: str, top_k: int) -> list[dict[str, Any]]:
        payload = _json(
            self.session.get(
                self.SINA_ROLL,
                params={
                    "pageid": 153,
                    "lid": 2516,
                    "k": query,
                    "num": top_k,
                    "page": 1,
                },
                headers={**DEFAULT_HEADERS, "Referer": "https://finance.sina.com.cn/"},
                timeout=self.timeout,
            ),
            "sina",
        )
        data = payload.get("result", {}).get("data") or payload.get("data") or []
        records = []
        for item in data:
            records.append(
                {
                    "date": _iso_date(item.get("ctime") or item.get("pubDate") or item.get("date")),
                    "title": _normalize_title(item.get("title")),
                    "summary": _normalize_title(
                        item.get("intro") or item.get("summary") or item.get("keywords") or ""
                    )[:500],
                    "source": "sina",
                    "url": str(item.get("url") or item.get("wapurl") or ""),
                }
            )
        return records

    @staticmethod
    def _company_aliases(company_name: str) -> set[str]:
        compact = re.sub(r"[\s·・,，。()（）\-]", "", company_name or "")
        aliases = {compact} if len(compact) >= 2 else set()
        for suffix in ("股份有限公司", "有限责任公司", "集团股份", "集团", "股份", "有限公司"):
            if compact.endswith(suffix):
                shortened = compact[: -len(suffix)]
                if len(shortened) >= 2:
                    aliases.add(shortened)
        return aliases

    @classmethod
    def _query_candidates(cls, company_name: str, digits: str) -> list[str]:
        """Build a small exact-query fallback chain without relaxing relevance."""
        compact = re.sub(r"[\s·・,，。()（）\-]", "", company_name or "")
        candidates = [compact]
        candidates.extend(
            sorted(
                (alias for alias in cls._company_aliases(company_name) if alias != compact),
                key=len,
            )
        )
        if digits:
            candidates.append(digits)
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))[:3]

    @classmethod
    def _collect_query_candidates(
        cls,
        fetch: Any,
        queries: list[str],
        *,
        company_name: str,
        canonical: str,
        digits: str,
        start_date: str,
        end_date: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Try aliases/ticker in order and keep only strict, deduplicated matches."""
        collected: list[dict[str, Any]] = []
        used_queries: list[str] = []
        seen: set[tuple[str, str]] = set()
        for query in queries:
            if len(collected) >= top_k:
                break
            used_queries.append(query)
            valid = cls._valid_records(
                fetch(query, max(1, top_k - len(collected))),
                company_name=company_name,
                canonical=canonical,
                digits=digits,
                start_date=start_date,
                end_date=end_date,
                top_k=top_k - len(collected),
            )
            for item in valid:
                content_key = re.sub(
                    r"\s+", "", str(item.get("title") or item.get("summary") or "")
                )
                key = (str(item.get("date") or ""), content_key)
                if not content_key or key in seen:
                    continue
                seen.add(key)
                collected.append(item)
                if len(collected) >= top_k:
                    break
        return collected, used_queries

    @classmethod
    def _is_target_news(
        cls,
        item: dict[str, Any],
        *,
        company_name: str,
        canonical: str,
        digits: str,
    ) -> bool:
        # Provider-added ``symbol`` metadata is deliberately excluded: relevance
        # must be demonstrated by editorial text or upstream entity recognition.
        editorial = re.sub(
            r"\s+",
            "",
            f"{item.get('title') or ''} {item.get('summary') or ''}",
        ).lower()
        if any(alias.lower() in editorial for alias in cls._company_aliases(company_name)):
            return True
        if digits in editorial or canonical.lower() in editorial:
            return True

        entity_values: list[str] = []
        for key in (
            "entities",
            "entity_names",
            "mentioned_companies",
            "recognized_entities",
        ):
            value = item.get(key)
            if isinstance(value, dict):
                entity_values.extend(str(nested) for nested in value.values())
            elif isinstance(value, (list, tuple, set)):
                entity_values.extend(str(nested) for nested in value)
            elif value:
                entity_values.append(str(value))
        recognized = re.sub(r"\s+", "", " ".join(entity_values)).lower()
        return bool(
            digits in recognized
            or canonical.lower() in recognized
            or any(
                alias.lower() in recognized
                for alias in cls._company_aliases(company_name)
            )
        )

    @classmethod
    def _valid_records(
        cls,
        records: list[dict[str, Any]],
        *,
        company_name: str,
        canonical: str,
        digits: str,
        start_date: str,
        end_date: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for original in records:
            item = dict(original)
            observed = _iso_date(item.get("date"))
            if not observed or not (start_date <= observed <= end_date):
                continue
            if not cls._is_target_news(
                item,
                company_name=company_name,
                canonical=canonical,
                digits=digits,
            ):
                continue
            title_key = re.sub(r"\s+", "", item.get("title") or "")
            summary_key = re.sub(r"\s+", "", item.get("summary") or "")
            content_key = title_key or summary_key
            key = (observed, content_key)
            if not content_key or key in seen:
                continue
            seen.add(key)
            item["date"] = observed
            item["symbol"] = canonical
            item["content_type"] = "news"
            filtered.append(item)
            if len(filtered) >= top_k:
                break
        return filtered

    def _check_window_filter(
        self,
        *,
        provider: str,
        canonical: str,
        start_date: str,
        end_date: str,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            return
        fingerprint = frozenset(
            "|".join(
                (
                    str(item.get("date") or ""),
                    re.sub(r"\s+", "", str(item.get("title") or item.get("summary") or "")),
                    str(item.get("url") or ""),
                )
            )
            for item in records
        )
        history = getattr(self, "_window_fingerprints", None)
        if history is None:
            history = {}
            self._window_fingerprints = history
        key = (provider, canonical)
        window = (start_date, end_date)
        previous = history.get(key)
        history[key] = (window, fingerprint)
        if previous and previous[0] != window and previous[1] == fingerprint:
            raise PublicDataSourceError(
                provider,
                f"{provider} returned an identical news set for different time windows",
                retryable=False,
                code="PROVIDER_FILTER_BROKEN",
            )

    def news(
        self,
        code: str,
        company_name: str,
        start_date: str,
        end_date: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        canonical, digits, _exchange = normalize_a_share_code(code)
        source_chain: list[str] = []
        failures: list[dict[str, Any]] = []
        query_chain: dict[str, list[str]] = {}
        queries = self._query_candidates(company_name, digits)
        try:
            cls_records, query_chain["cls"] = self._collect_query_candidates(
                self._cls,
                queries,
                company_name=company_name,
                canonical=canonical,
                digits=digits,
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
            )
            self._check_window_filter(
                provider="cls",
                canonical=canonical,
                start_date=start_date,
                end_date=end_date,
                records=cls_records,
            )
            records = cls_records
            source_chain.append("cls")
        except Exception as error:
            records = []
            failures.append(
                {
                    "provider": "cls",
                    "error": type(error).__name__,
                    "code": getattr(error, "code", "UPSTREAM_ERROR"),
                    "message": str(error),
                }
            )

        if len(records) < top_k:
            try:
                sina_records, query_chain["sina"] = self._collect_query_candidates(
                    self._sina,
                    queries,
                    company_name=company_name,
                    canonical=canonical,
                    digits=digits,
                    start_date=start_date,
                    end_date=end_date,
                    top_k=top_k - len(records),
                )
                self._check_window_filter(
                    provider="sina",
                    canonical=canonical,
                    start_date=start_date,
                    end_date=end_date,
                    records=sina_records,
                )
                records.extend(sina_records)
                source_chain.append("sina")
            except Exception as error:
                failures.append(
                    {
                        "provider": "sina",
                        "error": type(error).__name__,
                        "code": getattr(error, "code", "UPSTREAM_ERROR"),
                        "message": str(error),
                    }
                )

        filtered: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str]] = set()
        for item in records:
            observed = str(item["date"])
            title_key = re.sub(r"\s+", "", item.get("title") or "")
            summary_key = re.sub(r"\s+", "", item.get("summary") or "")
            key = (observed, title_key or summary_key)
            if not (title_key or summary_key) or key in seen:
                continue
            seen.add(key)
            filtered.append(item)
            if len(filtered) >= top_k:
                break
        if not filtered:
            broken_filter = next(
                (
                    failure
                    for failure in failures
                    if failure.get("code") == "PROVIDER_FILTER_BROKEN"
                ),
                None,
            )
            if broken_filter is not None:
                raise PublicDataSourceError(
                    str(broken_filter["provider"]),
                    str(broken_filter["message"]),
                    retryable=False,
                    code="PROVIDER_FILTER_BROKEN",
                )
            raise PublicDataSourceError(
                "financial_news",
                "CLS and Sina returned zero target-relevant dated news records",
                retryable=False,
                code="INVALID_DATA",
            )
        return {
            "symbol": canonical,
            "company_name": company_name,
            "start_date": start_date,
            "end_date": end_date,
            "source_chain": source_chain,
            "query_chain": query_chain,
            "source_failures": failures,
            "items": filtered,
        }
