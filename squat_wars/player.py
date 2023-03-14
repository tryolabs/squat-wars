import time
from typing import Optional


class Player:
    def __init__(self, name: str, email: Optional[str] = None, timestamp: Optional[int] = None):
        self.name = name
        self.email = email
        self.timestamp = timestamp if timestamp else int(time.time())

    def __repr__(self):
        return self.email

    def __eq__(self, other: "Player") -> bool:
        return self.email == other.email

    def __str__(self) -> str:
        return self.name
