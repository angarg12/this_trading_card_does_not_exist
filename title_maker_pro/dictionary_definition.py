from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    word: str
    text_body: str
