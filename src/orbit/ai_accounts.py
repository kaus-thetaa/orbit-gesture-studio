"""Local, credential-free state for the AI Hub dashboard."""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import paths
from .fsutil import atomic_write_json, quarantine_corrupt

log = logging.getLogger(__name__)

PROVIDERS = {
    "Claude": "https://claude.ai",
    "ChatGPT": "https://chatgpt.com",
    "Gemini": "https://gemini.google.com",
}
STATUSES = {"available", "cooldown", "input_needed"}


@dataclass
class AIAccount:
    id: str
    provider: str
    name: str
    status: str = "available"
    cooldown_until: Optional[float] = None


class AIAccountStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else paths.ai_accounts_path()
        try:
            self.accounts = self._load()
        except (OSError, ValueError) as exc:
            log.error("AI Hub data unreadable (%s); preserving a backup and using defaults", exc)
            quarantine_corrupt(self.path)
            self.accounts = []
        if not self.path.exists():
            self.accounts = [
                AIAccount(str(uuid.uuid4()), provider, f"My {provider}")
                for provider in PROVIDERS
            ]
            self._save()

    def add(self, provider: str, name: str) -> AIAccount:
        name = name.strip()
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        if not name or len(name) > 48:
            raise ValueError("Account label must be between 1 and 48 characters.")
        if any(account.name.casefold() == name.casefold() for account in self.accounts):
            raise ValueError(f"An account named {name!r} already exists.")
        account = AIAccount(str(uuid.uuid4()), provider, name)
        self.accounts.append(account)
        self._save()
        return account

    def remove(self, account_id: str) -> None:
        self.accounts = [account for account in self.accounts if account.id != account_id]
        self._save()

    def set_status(self, account_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Unsupported account status: {status}")
        account = self._get(account_id)
        account.status = status
        account.cooldown_until = None
        self._save()

    def start_cooldown(self, account_id: str, hours: int, minutes: int) -> None:
        if not (0 <= hours <= 168 and 0 <= minutes <= 59) or hours * 60 + minutes <= 0:
            raise ValueError("Cooldown must be from 1 minute to 7 days.")
        account = self._get(account_id)
        account.status = "cooldown"
        account.cooldown_until = time.time() + (hours * 60 + minutes) * 60
        self._save()

    def expire_cooldowns(self, now: Optional[float] = None) -> list[AIAccount]:
        now = time.time() if now is None else now
        expired = []
        for account in self.accounts:
            if account.status == "cooldown" and account.cooldown_until is not None and account.cooldown_until <= now:
                account.status = "available"
                account.cooldown_until = None
                expired.append(account)
        if expired:
            self._save()
        return expired

    def _get(self, account_id: str) -> AIAccount:
        for account in self.accounts:
            if account.id == account_id:
                return account
        raise ValueError("AI account no longer exists.")

    def _load(self) -> list[AIAccount]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("accounts"), list):
            raise ValueError(f"Invalid AI Hub data in {self.path}")
        accounts = []
        for raw in data["accounts"]:
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid AI Hub account in {self.path}")
            account = AIAccount(
                id=str(raw.get("id", "")),
                provider=raw.get("provider", ""),
                name=raw.get("name", ""),
                status=raw.get("status", "available"),
                cooldown_until=raw.get("cooldown_until"),
            )
            if (
                not account.id
                or account.provider not in PROVIDERS
                or not isinstance(account.name, str)
                or not account.name.strip()
                or len(account.name) > 48
                or account.status not in STATUSES
                or (
                    account.status == "cooldown"
                    and (
                        not isinstance(account.cooldown_until, (int, float))
                        or not math.isfinite(account.cooldown_until)
                    )
                )
            ):
                raise ValueError(f"Invalid AI Hub account in {self.path}")
            accounts.append(account)
        return accounts

    def _save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "version": 1,
                "accounts": [
                    {
                        "id": account.id,
                        "provider": account.provider,
                        "name": account.name,
                        "status": account.status,
                        "cooldown_until": account.cooldown_until,
                    }
                    for account in self.accounts
                ],
            },
        )
