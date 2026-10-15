import pytest

from kotha.helpdocs import chunk_markdown, search


def test_chunking_keeps_sections_and_positions():
    md = "# Title\n\nIntro text.\n\n## One\nFirst.\n\n## Two\nSecond."
    chunks = chunk_markdown(md, "doc", "en")
    assert [(c.section, c.content, c.pos) for c in chunks] == [
        ("Title", "Intro text.", 0), ("One", "First.", 1), ("Two", "Second.", 2)]


@pytest.mark.parametrize("query,lang,section", [
    ("How long do refunds take?", "en", "Refund timing"),
    ("রিফান্ড কতদিনে পাব?", "bn", "রিফান্ড কতদিনে পাব"),
    ("delivery charge?", "en", "Delivery charge"),
    ("ডেলিভারি চার্জ কত", "bn", "ডেলিভারি চার্জ"),
    ("how long does delivery take outside dhaka?", "en", "Delivery time"),
    ("ঢাকার বাইরে কত দিন লাগে?", "bn", "ডেলিভারি সময়"),
    ("can I pay with nagad?", "en", "Paying with bKash or Nagad"),
    ("money cut but payment failed", "en", "Payment failed but money was cut"),
    ("size exchange possible?", "en", "Return policy"),
])
def test_search_finds_the_right_section(query, lang, section):
    hits = search("demo", query, lang=lang)
    assert hits and hits[0].section == section


def test_search_answers_in_customer_language():
    # Banglish question matched against English text comes back as the Bangla twin.
    hits = search("demo", "order cancel korbo kivabe", lang="bn")
    assert hits[0].lang == "bn" and hits[0].section == "অর্ডার বাতিল"


@pytest.mark.parametrize("query", ["what is the meaning of life", "who won the football match",
                                   "is there a warranty?"])
def test_search_returns_nothing_for_off_topic(query):
    assert search("demo", query, lang="en") == []


def test_search_is_scoped_to_shop():
    assert search("some-other-shop", "refund") == []
