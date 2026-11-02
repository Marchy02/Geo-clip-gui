import pytest

from geoosint.fetch import FetchError, _check_host
from geoosint.geo.links import map_links, reverse_image_links, text_search_links


def test_map_links_well_formed():
    links = map_links(45.4408, 12.3155)
    assert len(links) >= 15
    assert all(link.url.startswith("https://") for link in links)
    assert any("45.440800" in link.url for link in links)


def test_reverse_links_with_and_without_url():
    assert all("url=" not in link.url for link in reverse_image_links())
    direct = reverse_image_links("https://example.com/a b.jpg")
    assert any("https%3A%2F%2Fexample.com%2Fa%20b.jpg" in link.url for link in direct)


def test_text_links_are_encoded():
    assert all(" " not in link.url for link in text_search_links("Via Roma 1 & co"))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/a.jpg",
        "http://127.0.0.1/a.jpg",
        "http://localhost/a.jpg",
        "http://10.0.0.1/x.png",
        "notaurl",
    ],
)
def test_fetch_blocks_unsafe_urls(url):
    with pytest.raises(FetchError):
        _check_host(url)
