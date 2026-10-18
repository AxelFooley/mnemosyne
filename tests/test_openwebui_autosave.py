import asyncio

from mnemosyne.integrations import auto_save_openwebui


def test_health_check_uses_openwebui_health_route(monkeypatch):
    requested = []

    async def fake_api_get_async(url, api_key=""):
        requested.append((url, api_key))
        return {"status": True}

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)

    assert asyncio.run(auto_save_openwebui._health_check("http://openwebui.test", "key"))
    assert requested == [("http://openwebui.test/health", "key")]


def test_sync_once_reads_current_openwebui_chat_payload(monkeypatch):
    chat_id = "chat/one"
    list_url = "http://openwebui.test/api/v1/chats/?page=1"
    detail_url = "http://openwebui.test/api/v1/chats/chat%2Fone"
    next_page_url = "http://openwebui.test/api/v1/chats/?page=2"
    responses = {
        list_url: [{"id": chat_id, "title": "Test chat"}],
        detail_url: {
            "user_id": "webui-user",
            "chat": {
                "history": {
                    "messages": {
                        "user-message": {
                            "role": "user",
                            "content": "Remember this",
                            "timestamp": 1_700_000_000,
                        }
                    },
                    "currentId": "user-message",
                },
                "messages": [
                    {
                        "id": "legacy-message",
                        "role": "assistant",
                        "content": "Saved too",
                        "timestamp": 1_700_000_001,
                    }
                ],
            },
        },
        next_page_url: [],
    }
    requested = []

    async def fake_api_get_async(url, api_key=""):
        requested.append((url, api_key))
        return responses[url]

    class FakeSaver:
        def __init__(self):
            self.saved = []

        def save_message(self, **kwargs):
            self.saved.append(kwargs)
            return True

        def persist(self):
            pass

        def get_stats(self):
            return {"saved": len(self.saved), "skipped": 0, "errors": 0}

    monkeypatch.setattr(auto_save_openwebui, "_api_get_async", fake_api_get_async)
    engine = object.__new__(auto_save_openwebui.SyncEngine)
    engine.url = "http://openwebui.test"
    engine.api_key = "key"
    engine.saver = FakeSaver()

    result = asyncio.run(engine.sync_once())

    assert requested == [
        (list_url, "key"),
        (detail_url, "key"),
        (next_page_url, "key"),
    ]
    assert result["saved"] == 2
    assert [item["message"]["id"] for item in engine.saver.saved] == [
        "user-message",
        "legacy-message",
    ]
    assert all(item["user_id"] == "webui-user" for item in engine.saver.saved)
