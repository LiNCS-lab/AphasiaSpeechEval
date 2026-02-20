import unittest
from unittest.mock import patch, MagicMock
from noisy_speech_eval.models import *

from noisy_speech_eval.modules.chat_file_parser import (
    _get_participants,
    process_chat,
    _tokenize,
    _extract_special_form_markers,
    _extract_prefixes,
    _extract_gestures,
    _extract_retracing_overlap,
    _extract_special_utterance_terminators,
    _extract_post_codes,
    _extract_word_error,
)


class TestChatFileParser(unittest.TestCase):

    def setUp(self):
        self.participant_data = {
            "CHI": {
                "name": "Child",
                "language": "eng",
                "corpus": "TestCorpus",
                "age": "3;6.",
                "sex": "female",
                "group": "Control",
                "ses": "middle",
                "role": "Target_Child",
                "education": None,
                "custom": None,
            }
        }

    def test_get_participants(self):
        participants = _get_participants(self.participant_data)
        self.assertEqual(len(participants), 1)
        self.assertEqual(participants[0].name, "Child")
        self.assertEqual(participants[0].language, "eng")

    @patch("noisy_speech_eval.modules.chat_file_parser.pla.Reader.from_files")
    def test_process_chat(self, mock_from_files):
        mock_chat = MagicMock()
        mock_chat.headers.return_value = [{"Participants": self.participant_data}]
        mock_chat.languages.return_value = ["eng"]
        mock_chat.utterances.return_value = []
        mock_chat.words.return_value = []

        mock_from_files.return_value = mock_chat

        utterances, participants, languages, header_media, header_options = (
            process_chat("dummy_path", "dummy_session")
        )
        self.assertEqual(len(participants), 1)
        self.assertEqual(participants[0].name, "Child")
        self.assertEqual(languages, ["eng"])

    def test_tokenize(self):
        token = _tokenize("xxx")
        self.assertIsInstance(token, Token)
        self.assertEqual(token.token_type, TokenType.UNINTELLIGIBLE)

        token = _tokenize("„")
        self.assertIsInstance(token, SateliteMarker)
        self.assertEqual(token.satelite_marker, SateliteMarkerType.SUFFIX)

        token = _tokenize("+<")
        self.assertIsInstance(token, Token)
        self.assertEqual(token.token_type, TokenType.LAZY_OVERLAP)

        token = _tokenize("(..)")
        self.assertIsInstance(token, Pause)
        self.assertEqual(token.pause_type, PauseType.MEDIUM)

        token = _tokenize("(1:2.3)")
        self.assertIsInstance(token, Pause)
        self.assertEqual(token.pause_type, PauseType.DURATION)
        self.assertEqual(token.duration, 62.3)

        self.assertRaises(ValueError, _tokenize, "(.....)")

        token = _tokenize("hello")
        self.assertIsInstance(token, Word)
        self.assertEqual(token.word_type, WordType.NORMAL)

    def test_extract_special_form_markers(self):
        token = _extract_special_form_markers("word@u")
        self.assertIsInstance(token, Word)
        self.assertEqual(token.word_type, WordType.IPA_TRANSCRIPTION)
        self.assertRaises(ValueError, _extract_special_form_markers, "word@123")

    def test_extract_prefixes(self):
        token = _extract_prefixes("&+Tuz")
        self.assertIsInstance(token, Token)
        self.assertEqual(token.token_type, TokenType.PHONOLOGICAL_FRAGMENT)
        self.assertRaises(ValueError, _extract_prefixes, "&+Tuz&+two")

    def test_extract_gestures(self):
        token = _extract_gestures("&=head:yes")
        self.assertIsInstance(token, Token)
        self.assertEqual(token.token_type, TokenType.GESTURE)
        self.assertRaises(ValueError, _extract_gestures, "&=head:yes&=head:no")

    def test_extract_retracing_overlap(self):
        marker = _extract_retracing_overlap("[/]", 0, 1)
        self.assertIsInstance(marker, Marker)
        self.assertEqual(marker.marker_type, MarkerType.REPETITION)
        self.assertRaises(ValueError, _extract_retracing_overlap, "[x]", 0, 1)
        self.assertRaises(ValueError, _extract_retracing_overlap, "[/][/]", 0, 1)

    def test_extract_special_utterance_terminators(self):
        terminator = _extract_special_utterance_terminators("+...")
        self.assertIsInstance(terminator, Terminator)
        self.assertEqual(terminator.terminator, TerminatorType.TRAILING_OFF)
        self.assertRaises(ValueError, _extract_special_utterance_terminators, "+......")
        self.assertRaises(ValueError, _extract_special_utterance_terminators, "+/.+!?")

    def test_extract_post_codes(self):
        post_code = _extract_post_codes("[+gram]")
        self.assertIsInstance(post_code, PostCode)
        self.assertEqual(post_code.post_code, PostCodeType.GRAMMATICAL)
        self.assertRaises(ValueError, _extract_post_codes, "[+gram][+gram]")
        self.assertRaises(ValueError, _extract_post_codes, "[+wrongcode]")
        self.assertIsNone(_extract_post_codes("x"))

    def test_extract_word_error(self):
        word_error = _extract_word_error("p")
        self.assertIsInstance(word_error, WordErrorType)
        self.assertEqual(word_error, WordErrorType.PHONOLOGICAL)
        self.assertRaises(ValueError, _extract_word_error, "x")


if __name__ == "__main__":
    unittest.main()
