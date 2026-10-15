from datetime import timedelta

from kotha.lang import area_name, detect_lang, extract_order_numbers, normalize_digits
from kotha.timefmt import part_of_day, when

from .conftest import NOW


def test_detect_lang():
    assert detect_lang("Where is order #4821?") == "en"
    assert detect_lang("অর্ডার #৪৮২১ কোথায়?") == "bn"
    assert detect_lang("আমার order টা কোথায়?") == "bn"          # code-mixed
    assert detect_lang("amar order kobe pabo") == "bn"           # Banglish
    assert detect_lang("when will my order arrive") == "en"


def test_order_numbers():
    assert extract_order_numbers("Where is order #4821?") == [4821]
    assert extract_order_numbers("অর্ডার #৪৮২১") == [4821]
    assert extract_order_numbers("অর্ডার নং ৪৮২১ আর 4810") == [4821]
    assert extract_order_numbers("4815 status?") == [4815]
    assert extract_order_numbers("call me on 01712345821") == []   # phone, not an order
    assert normalize_digits("৳১,২৫০") == "৳1,250"


def test_time_phrases():
    eta = (NOW.replace(hour=17)).isoformat()
    assert when(eta, "en") == "5 pm today"
    assert when(eta, "bn") == "আজ বিকেল ৫টা"
    assert when((NOW + timedelta(days=1, hours=1)).isoformat(), "en") == "11 am tomorrow"
    assert when(NOW.replace(hour=16, minute=20).isoformat(), "en", round_up=True) == "5 pm today"
    assert part_of_day(NOW.replace(hour=8).isoformat(), "en") == "this morning"
    assert part_of_day(NOW.replace(hour=8).isoformat(), "bn") == "আজ সকালে"
    assert part_of_day((NOW - timedelta(days=1)).replace(hour=15).isoformat(), "en") == "yesterday afternoon"


def test_area_names():
    assert area_name("Mirpur", "bn") == "মিরপুর"
    assert area_name("Mirpur", "en") == "Mirpur"
    assert area_name("Some New Area", "bn") == "Some New Area"
