import unittest
from aphasia_speech_eval.models.types import (
    TokenType,
    WordType,
    PauseType,
    MarkerType,
    TerminatorType,
    PostCodeType,
    WordErrorType,
    PartOfSpeech,
    AffixType,
    SateliteMarkerType,
)


class TestTypes(unittest.TestCase):
    def test_token_types(self):
        self.assertEqual(TokenType.PHONOLOGICAL_FRAGMENT.value, "&+")
        self.assertEqual(TokenType.FILLER.value, "&-")
        self.assertEqual(TokenType.NON_WORD.value, "&~")
        self.assertEqual(TokenType.LAZY_OVERLAP.value, "+<")

    def test_word_types(self):
        self.assertEqual(WordType.NORMAL.value, 1)
        self.assertEqual(WordType.ADDITION.value, "@a")
        self.assertEqual(WordType.BABBLING.value, "@b")

    def test_pause_types(self):
        self.assertEqual(str(PauseType.SHORT), "(.)")
        self.assertEqual(str(PauseType.MEDIUM), "(..)")

    def test_marker_types(self):
        self.assertEqual(str(MarkerType.REPETITION), "[/]")
        self.assertEqual(str(MarkerType.RETRACING), "[//]")

    def test_terminator_types(self):
        self.assertEqual(str(TerminatorType.TRAILING_OFF), "+...")
        self.assertEqual(str(TerminatorType.TRAIL_OFF_TO_A_QUESTION), "+..?")

    def test_post_code_types(self):
        self.assertEqual(str(PostCodeType.GRAMMATICAL), "[+ gram]")
        self.assertEqual(str(PostCodeType.EXCLUDED), "[+ exc]")
        self.assertEqual(str(PostCodeType.EMPTY_SPEECH), "[+ es]")

    def test_word_error_types(self):
        self.assertEqual(str(WordErrorType.SEMANTIC), "[* s]")
        self.assertEqual(str(WordErrorType.PHONOLOGICAL), "[* p]")

    def test_part_of_speech(self):
        self.assertEqual(PartOfSpeech.ADJECTIVE.value, "adj")
        self.assertEqual(PartOfSpeech.ADVERB.value, "adv")

    def test_affix_types(self):
        self.assertEqual(AffixType.PRESENT_PARTICIPLE.value, "PRESP")
        self.assertEqual(AffixType.PAST.value, "PAST")

    def test_satelite_marker_types(self):
        self.assertEqual(str(SateliteMarkerType.PREFIX), "‡")
        self.assertEqual(str(SateliteMarkerType.SUFFIX), "„")


if __name__ == "__main__":
    unittest.main()
