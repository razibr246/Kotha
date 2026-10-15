from types import SimpleNamespace

import httpx
import pytest

from kotha.agent import ClaudeAgent, Kotha, OfflineAgent, Turn
from kotha.courier import CourierError, HttpCourier, MockCourier
from kotha.tools import Toolbox


@pytest.fixture
def agent():
    return OfflineAgent(MockCourier())


def ask(agent, text, history=None):
    return agent.answer("demo", text, history or [])


def test_card_example_english(agent):
    r = ask(agent, "Where is order #4821?")
    assert r.text == "Shipped this morning. It reaches Mirpur by 5 pm today."
    assert r.sources["count"] == 2
    assert r.sources["labels"] == ["order DB", "courier API"]


def test_card_example_bangla(agent):
    r = ask(agent, "অর্ডার #৪৮২১ কোথায়?")
    assert r.lang == "bn"
    assert r.text == "আজ সকালে শিপ হয়েছে। আজ বিকেল ৫টার মধ্যে মিরপুর পৌঁছে যাবে।"


def test_out_for_delivery_cod_reminder(agent):
    r = ask(agent, "amar order 4810 kobe pabo")
    assert r.lang == "bn"
    assert "ডেলিভারির জন্য বের হয়েছে" in r.text and "৳১,৯২০" in r.text


def test_failed_attempt_masks_phone(agent):
    r = ask(agent, "where is 4807")
    assert "couldn't reach you" in r.text
    assert "•••• 807" in r.text and "01911224807" not in r.text


def test_delivered(agent):
    assert ask(agent, "4815 status?").text == "Order #4815 was delivered on 13 Oct."


def test_not_shipped_yet_skips_courier(agent):
    r = ask(agent, "#4822")
    assert "hasn't shipped" in r.text
    assert r.sources["labels"] == ["order DB"]


def test_cancelled_paid_order_plus_policy(agent):
    r = ask(agent, "order 4799 refund?")
    assert "refunded via bKash" in r.text and "3-5 working days" in r.text
    assert r.sources["labels"] == ["order DB", "help docs"]


def test_unknown_order(agent):
    r = ask(agent, "order 9999")
    assert "couldn't find order #9999" in r.text
    assert r.sources["count"] == 0


def test_other_shops_orders_are_invisible(agent):
    from kotha.db import connection
    with connection() as conn:
        conn.execute("INSERT INTO shops (id, name, support_phone) VALUES ('other', 'Other', '1') "
                     "ON CONFLICT DO NOTHING")
        conn.commit()
    r = agent.answer("other", "Where is order #4821?", [])
    assert "couldn't find" in r.text


def test_follow_up_uses_earlier_order_number(agent):
    hist = [Turn("user", "order #4821"), Turn("assistant", "Shipped this morning...")]
    r = ask(agent, "when will it arrive?", hist)
    assert "5 pm today" in r.text


def test_asks_for_order_number(agent):
    assert "order number" in ask(agent, "where is my order").text


def test_off_topic_offers_human(agent):
    r = ask(agent, "what is the meaning of life")
    assert "09611-482100" in r.text and r.sources["count"] == 0


def test_handoff(agent):
    r = ask(agent, "I want to talk to a human")
    assert r.handoff


def test_courier_outage_is_reported_not_invented():
    class Down:
        def track(self, tid):
            raise CourierError("timeout")
    r = OfflineAgent(Down()).answer("demo", "Where is order #4821?", [])
    assert "couldn't get live tracking" in r.text
    assert r.sources["labels"] == ["order DB"]


def test_http_courier_against_mock_api():
    from fastapi.testclient import TestClient
    from kotha.main import app
    with TestClient(app) as c:
        courier = HttpCourier("http://testserver/mock-courier", client=c)
        t = courier.track("PTH-88214821")
        assert t["status"] == "at_hub" and t["destination_area"] == "Mirpur"
        assert courier.track("NOPE") is None


def test_http_courier_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    courier = HttpCourier("http://x", client=httpx.Client(transport=transport))
    with pytest.raises(CourierError):
        courier.track("PTH-88214821")


# ------------------------------------------------------------- Claude tool loop

def _block(**kw):
    return SimpleNamespace(**kw)


class FakeMessages:
    """Scripted Claude: asks for get_order, then track_shipment, then answers."""

    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        n = len(self.calls)
        if n == 1:
            return _block(stop_reason="tool_use", content=[
                _block(type="tool_use", id="t1", name="get_order", input={"order_no": 4821})])
        if n == 2:
            return _block(stop_reason="tool_use", content=[
                _block(type="tool_use", id="t2", name="track_shipment",
                       input={"tracking_id": "PTH-88214821"})])
        return _block(stop_reason="end_turn", content=[
            _block(type="text", text="Shipped this morning. It reaches Mirpur by 5 pm today.")])


def make_claude(messages):
    a = ClaudeAgent.__new__(ClaudeAgent)
    a.client = SimpleNamespace(messages=messages)
    a.model = "test"
    a.courier = MockCourier()
    return a


def test_claude_tool_loop_grounds_and_cites():
    fake = FakeMessages()
    r = make_claude(fake).answer("demo", "Where is order #4821?", [])
    assert r.engine == "claude"
    assert [c["name"] for c in r.tool_calls] == ["get_order", "track_shipment"]
    assert r.sources["labels"] == ["order DB", "courier API"]
    # Tool results reached the model, with ready-made local time phrases.
    tool_result = fake.calls[2]["messages"][-1]["content"][0]["content"]
    assert '"eta_phrase"' in tool_result and "5 pm today" in tool_result
    assert "customer_phone" not in tool_result and "01712345821" not in tool_result
    assert "English" in fake.calls[0]["system"]


def test_claude_failure_falls_back_to_offline():
    class Broken:
        def create(self, **kw):
            raise RuntimeError("overloaded")
    k = Kotha(MockCourier())
    k.claude = make_claude(Broken())
    r = k.answer("demo", "Where is order #4821?")
    assert r.engine == "offline-fallback"
    assert r.text == "Shipped this morning. It reaches Mirpur by 5 pm today."


def test_toolbox_rejects_bad_tool_calls():
    tb = Toolbox("demo", MockCourier())
    assert "unknown tool" in tb.call("drop_tables", {})
    assert "bad arguments" in tb.call("get_order", {"nope": 1})
