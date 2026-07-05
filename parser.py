from collections import defaultdict
import datetime as dt
import io
import json
import logging
from pathlib import Path
from typing import NamedTuple, TypedDict
import zipfile

from langchain_core.documents import Document

from constants import EXPECTED_SN_VERSION

logger = logging.getLogger(__name__)

_TAG_MEMBER_PREFIX = "Items/Tag/"


class StandardNotesData(TypedDict):
    uuid: str
    title: str
    text: str
    date: dt.date
    tags: set[str]


class StandardNotesTag(TypedDict):
    title: str
    references: list[str]


class ParsedBackup(NamedTuple):
    """`skipped` counts Note items dropped for a violated data contract — an
    unreadable (encrypted) content payload or a title without the yyyy-mm-dd
    prefix. Deleted items and non-Note items are expected non-data, not skips."""

    notes: list[StandardNotesData]
    skipped: int


def parse_standard_notes(
    backup_zip_path: Path, notes_json: str = "Standard Notes Backup and Import File.txt"
) -> ParsedBackup:
    tag_data = []

    # Read just the needed members straight from the ZIP — backups can carry large
    # file attachments, and nothing needs extracting to disk. Explicit utf-8: SN
    # exports are utf-8 and the diary is Czech — the locale default (e.g. cp1250
    # on Windows) would crash or mojibake every entry.
    with zipfile.ZipFile(backup_zip_path) as zf:
        with zf.open(notes_json) as f:
            sn_data = json.load(io.TextIOWrapper(f, encoding="utf-8"))

        if (v := sn_data.get("version")) != EXPECTED_SN_VERSION:
            logger.warning(
                "Standard Notes backup version changed. Expected %r, found %r.",
                EXPECTED_SN_VERSION,
                v,
            )

        for name in zf.namelist():
            if name.startswith(_TAG_MEMBER_PREFIX) and name.endswith(".txt"):
                with zf.open(name) as f:
                    tag_file_data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
                    tag_data.append(
                        StandardNotesTag(
                            title=tag_file_data["title"],
                            references=[r["uuid"] for r in tag_file_data["references"]],
                        )
                    )

    id_tag_map = defaultdict(set)
    for item in tag_data:
        title = item["title"]
        for ref in item["references"]:
            id_tag_map[ref].add(title)

    iso_date_fmt = "yyyy-mm-dd"
    parsed_data = []
    skipped = 0

    for item in sn_data["items"]:
        if not item.get("deleted") and item.get("content_type") == "Note":
            content = item.get("content")
            if not isinstance(content, dict):
                # Encrypted backups carry content as an opaque string — skip; the
                # caller reports zero parsed notes with a "decrypted export?" hint.
                skipped += 1
                continue
            uuid = item["uuid"]
            tags = id_tag_map[uuid]
            text = content.get("text", "")
            title = content.get("title", "")

            try:
                date = dt.date.fromisoformat(title[: len(iso_date_fmt)])
                title = title[len(iso_date_fmt) :]
                parsed_data.append(
                    StandardNotesData(
                        uuid=uuid, title=title, text=text, date=date, tags=tags
                    )
                )
            except Exception:
                skipped += 1
                continue

    if skipped:
        logger.warning(
            "Skipped %d Note item(s): no yyyy-mm-dd title prefix or unreadable "
            "(encrypted?) content",
            skipped,
        )
    parsed_data.sort(key=lambda x: x["date"])
    return ParsedBackup(notes=parsed_data, skipped=skipped)


def documents_from_notes(parsed_notes: list[StandardNotesData]) -> list[Document]:
    """Map parsed notes to LangChain Documents — one Document per entry (entries are
    tiny; no chunking). Shared by the app's ingest and the retrieval tests so the
    metadata contract can't drift between them."""
    documents = []
    for note in parsed_notes:
        if not note["text"].strip():
            continue
        date = note["date"]
        metadata = {
            "uuid": note["uuid"],
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "date_str": date.isoformat(),
            # Numeric yyyymmdd — Chroma's $gte/$lte are numeric-only, so date
            # RANGES filter on this key.
            "date_int": date.year * 10000 + date.month * 100 + date.day,
            "title": note["title"].strip(),
        }
        # Chroma forbids empty lists; omit the key for untagged entries.
        if note["tags"]:
            metadata["tags"] = sorted(note["tags"])
        documents.append(Document(page_content=note["text"], metadata=metadata))
    return documents
