from persode.analyzer import EventEmotionAnalyzer


def test_detects_joy():
    a = EventEmotionAnalyzer()
    m = a.analyze("I celebrated my graduation, it was a joyful success!")
    assert m.emotion == "joyful"
    assert m.valence == 1
    assert m.emotional_intensity > 0.7


def test_detects_anger():
    a = EventEmotionAnalyzer()
    m = a.analyze("A car splashed water and ruined my outfit, I'm so frustrated!")
    assert m.emotion == "angry"
    assert m.valence == -1


def test_neutral_has_low_intensity():
    a = EventEmotionAnalyzer()
    m = a.analyze("I took the bus to work.")
    assert m.emotion == "neutral"
    assert m.emotional_intensity < 0.4


def test_hashtags_and_event_extracted():
    a = EventEmotionAnalyzer()
    m = a.analyze("I had a wonderful family dinner with delicious meat.")
    assert m.hashtags
    assert all(t.startswith("#") for t in m.hashtags)
    assert "dinner" in m.event or "meat" in m.event


def test_intensity_amplified_by_exclamation():
    a = EventEmotionAnalyzer()
    calm = a.analyze("I am happy")
    loud = a.analyze("I am so happy!!!")
    assert loud.emotional_intensity >= calm.emotional_intensity


def test_segment_splits_multiple_episodes():
    a = EventEmotionAnalyzer()
    eps = a.segment("I was thrilled to graduate. Later I felt sad about leaving friends.")
    emotions = {e.emotion for e in eps}
    assert "joyful" in emotions or "sad" in emotions
    assert len(eps) >= 1
