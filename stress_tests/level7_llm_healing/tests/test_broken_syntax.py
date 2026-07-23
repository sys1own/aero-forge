from broken_syntax import broken_syntax

def test_broken_syntax():
    assert broken_syntax(5) == 5
    assert broken_syntax(-3) == 3
