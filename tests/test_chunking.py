from utils.chunking import sliding_window_chunking


def test_sliding_window_chunking():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = sliding_window_chunking(text, 20, 5)

    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunking_with_empty_text():
    chunks = sliding_window_chunking("", 100, 20)
    assert isinstance(chunks, list)
    assert len(chunks) == 0
