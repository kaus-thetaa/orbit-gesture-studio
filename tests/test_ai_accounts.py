import json

import pytest

from orbit.ai_accounts import AIAccountStore, PROVIDERS


def test_store_seeds_provider_cards_without_credentials(tmp_path):
    path = tmp_path / "ai_accounts.json"
    store = AIAccountStore(path)

    assert [account.provider for account in store.accounts] == list(PROVIDERS)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert all("password" not in account and "token" not in account for account in persisted["accounts"])


def test_add_remove_and_persist_multiple_accounts(tmp_path):
    path = tmp_path / "ai_accounts.json"
    store = AIAccountStore(path)
    added = store.add(
        "Claude",
        "Claude work",
        "https://claude.ai/chat/abc123",
        "Profile 2",
    )

    reloaded = AIAccountStore(path)
    persisted = next(account for account in reloaded.accounts if account.id == added.id)
    assert persisted.name == "Claude work"
    assert persisted.chat_url == "https://claude.ai/chat/abc123"
    assert persisted.chrome_profile == "Profile 2"
    reloaded.remove(added.id)
    assert all(account.id != added.id for account in AIAccountStore(path).accounts)


def test_account_update_changes_chat_and_profile(tmp_path):
    store = AIAccountStore(tmp_path / "ai_accounts.json")
    account = store.accounts[0]
    store.update(account.id, "Claude", "Claude work", "https://claude.ai/chat/abc", "Profile 3")

    assert account.name == "Claude work"
    assert account.chat_url == "https://claude.ai/chat/abc"
    assert account.chrome_profile == "Profile 3"


@pytest.mark.parametrize(
    "provider,name,url,profile,error",
    [
        ("Claude", "Work", "http://claude.ai/chat/abc", "Default", "HTTPS"),
        ("Claude", "Work", "https://evil.example/chat/abc", "Default", "HTTPS"),
        ("ChatGPT", "Work", "https://claude.ai/chat/abc", "Default", "ChatGPT"),
        ("Gemini", "Work", "https://user:pass@gemini.google.com/app/abc", "Default", "credentials"),
        ("Claude", "Work", "https://claude.ai/chat/abc", "/tmp/profile", "Chrome profile"),
    ],
)
def test_rejects_unsafe_chat_links_and_profiles(tmp_path, provider, name, url, profile, error):
    store = AIAccountStore(tmp_path / "ai_accounts.json")
    with pytest.raises(ValueError, match=error):
        store.add(provider, name, url, profile)


def test_duplicate_labels_and_invalid_providers_rejected(tmp_path):
    store = AIAccountStore(tmp_path / "ai_accounts.json")
    with pytest.raises(ValueError, match="already exists"):
        store.add("Claude", "My Claude")
    with pytest.raises(ValueError, match="Unsupported provider"):
        store.add("Unknown", "My account")


def test_cooldown_expiry_and_input_status_are_persisted(tmp_path):
    store = AIAccountStore(tmp_path / "ai_accounts.json")
    account = store.accounts[0]
    store.start_cooldown(account.id, 0, 1)
    assert account.status == "cooldown"
    assert store.expire_cooldowns(account.cooldown_until + 1) == [account]
    assert account.status == "available"
    store.set_status(account.id, "input_needed")

    reloaded = AIAccountStore(tmp_path / "ai_accounts.json")
    assert next(item for item in reloaded.accounts if item.id == account.id).status == "input_needed"


def test_cooldown_duration_must_be_valid(tmp_path):
    store = AIAccountStore(tmp_path / "ai_accounts.json")
    for hours, minutes in ((0, 0), (169, 0), (0, 60), (-1, 0)):
        with pytest.raises(ValueError):
            store.start_cooldown(store.accounts[0].id, hours, minutes)


def test_invalid_store_data_is_preserved_and_restored(tmp_path):
    path = tmp_path / "ai_accounts.json"
    path.write_text('{"version": 1, "accounts": [{"provider": "Unknown"}]}', encoding="utf-8")
    store = AIAccountStore(path)

    assert len(store.accounts) == len(PROVIDERS)
    assert (tmp_path / "ai_accounts.json.corrupt").is_file()
