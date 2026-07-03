from collections import defaultdict
import datetime as dt
import json
import logging
from pathlib import Path
import tempfile
from typing import TypedDict
import zipfile

from constants import EXPECTED_SN_VERSION

logger = logging.getLogger(__name__)


class StandardNotesData(TypedDict):
    uuid: str
    title: str
    text: str
    date: dt.date
    tags: set[str]


class StandardNotesTag(TypedDict):
    title: str
    references: list[str]


def parse_standard_notes(
    backup_zip_path: Path, notes_json: str = "Standard Notes Backup and Import File.txt"
) -> list[StandardNotesData]:
    tag_data = []
    sn_data = None
    tags_path = Path("Items/Tag")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        with zipfile.ZipFile(backup_zip_path) as zf:
            zf.extractall(tmp_dir)
        with open(tmp_dir / notes_json) as f:
            sn_data = json.load(f)

        if (v := sn_data.get("version")) != EXPECTED_SN_VERSION:
            logger.warning(
                "Standard Notes backup version changed. Expected %r, found %r.",
                EXPECTED_SN_VERSION,
                v,
            )

        tag_file_paths = (tmp_dir / tags_path).glob("*.txt")
        for tag_file_path in tag_file_paths:
            with open(tag_file_path) as f:
                tag_file_data = json.load(f)
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

    for item in sn_data["items"]:
        if not item.get("deleted") and item.get("content_type") == "Note":
            content = item["content"]
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
                continue

    parsed_data.sort(key=lambda x: x["date"])
    return parsed_data
