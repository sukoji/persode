from persode.analyzer import EventEmotionAnalyzer
from persode.onboarding import OnboardingPreferences
from persode.templates import DiaryTemplate, FewShotTemplateSystem


def test_style_prompt_reflects_toggles():
    prefs = OnboardingPreferences(conversation_style="emotional", personality="empathetic",
                                  response_length="detailed")
    p = prefs.build_style_prompt()
    assert "emotional" in p.lower()
    assert "empath" in p.lower()


def test_visual_style_includes_identity():
    prefs = OnboardingPreferences(glasses=True, fashion_style="trendy", hair="dyed yellow hair")
    s = prefs.build_visual_style()["descriptor_string"]
    assert "glasses" in s
    assert "trendy" in s
    assert "dyed yellow hair" in s


def test_few_shot_prompt_combines_meta_and_style():
    prefs = OnboardingPreferences(age=15, fashion_style="casual", hair="dyed yellow hair")
    meta = EventEmotionAnalyzer().analyze(
        "I was scolded by my mom for spending all my allowance, I feel regretful.")
    vp = FewShotTemplateSystem().build(meta, prefs)
    assert "dyed yellow hair" in vp.prompt
    assert "15-year-old" in vp.prompt
    assert meta.emotion in vp.prompt


def test_personalization_changes_prompt():
    meta = EventEmotionAnalyzer().analyze("I had a peaceful walk in the park at sunset.")
    a = FewShotTemplateSystem().build(meta, OnboardingPreferences(age=15, background_theme="city"))
    b = FewShotTemplateSystem().build(meta, OnboardingPreferences(age=40, background_theme="nature"))
    assert a.prompt != b.prompt


def test_offline_diary_is_nonempty_and_tagged():
    prefs = OnboardingPreferences()
    meta = EventEmotionAnalyzer().analyze("I celebrated my graduation, it was joyful!")
    diary = DiaryTemplate().compose_offline(meta, prefs)
    assert len(diary) > 0
    assert "#" in diary
