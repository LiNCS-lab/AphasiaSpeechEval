from noisy_speech_eval import Session, TokenType
import os
import unittest
from noisy_speech_eval.models.session import *
from tests.test_data import test_data_cwd


class TestSession(unittest.TestCase):

    def test_detect_pauses(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))

        cpt = 0
        for u in sess.utterances:
            for t in u:
                if t.token_type == TokenType.PAUSE:
                    cpt += 1

        self.assertEqual(cpt, 3)
        # counts pause tokens + syllable pauses
        self.assertEqual(sess.count_unfilled_pauses(), 4)

    def test_load_file(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "simple.cha"))
        # Check total words count dans per-speaker count
        self.assertEqual(sess.count_words(), 14)
        self.assertEqual(sess.count_words("PAR"), 1)
        self.assertEqual(sess.count_words("INV"), 13)

        # Valider les tours de paroles
        self.assertEqual(len(sess.utterances), 2)

    def test_get_participant_utts(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "supersimple.cha"))
        inv_utt_true = ["one ."]
        par_utt_true = ["two ."]
        all_utt_true = ["one .", "two ."]

        participants = sess.participants
        par = sess.participants[0]
        inv = sess.participants[1]

        self.assertEqual(
            [str(u) for u in sess.get_participant_utts("INV")], inv_utt_true
        )
        self.assertEqual(
            [str(u) for u in sess.get_participant_utts("PAR")], par_utt_true
        )
        self.assertEqual([str(u) for u in sess.get_participant_utts()], all_utt_true)
        self.assertEqual([str(u) for u in sess.get_participant_utts(par)], par_utt_true)
        self.assertEqual([str(u) for u in sess.get_participant_utts(inv)], inv_utt_true)
        self.assertEqual(
            [str(u) for u in sess.get_participant_utts([par, inv])], all_utt_true
        )
        self.assertEqual(
            [str(u) for u in sess.get_participant_utts(["PAR", "INV"])], all_utt_true
        )
        self.assertEqual(
            [str(u) for u in sess.get_participant_utts(participants)], all_utt_true
        )

    def test_count_retraced_sequences(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "simple.cha"))
        self.assertEqual(sess.count_retraced_sequences(), 1)
        self.assertEqual(sess.count_retraced_sequences("PAR"), 0)
        self.assertEqual(sess.count_retraced_sequences("INV"), 1)

    def test_detect_syllable_pauses(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))

        self.assertEqual(1, sess.count_syllable_pauses())

    def test_detect_trailing_off(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "trailing_off.cha"))

        self.assertEqual(sess.count_abandoned_utterances(), 4)

    def test_detect_empty_speech(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "postcodes.cha"))

        self.assertEqual(sess.count_empty_speech(), 3)

    def test_detect_jargon(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "jargon.cha"))

        self.assertEqual(sess.count_neologisms(), 1)
        self.assertEqual(sess.count_jargon(), 1)
        self.assertEqual(sess.count_jargon("INV"), 1)
        self.assertEqual(sess.count_jargon("PAR"), 0)
        self.assertEqual(sess.count_agrammatic_utterances(), 1)
        self.assertEqual(sess.utterances[0].post_codes, {PostCodeType.GRAMMATICAL})
        self.assertEqual(sess.utterances[1].post_codes, {PostCodeType.JARGON})

    def test_detect_overlaps(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "overlaps.cha"))

        self.assertEqual(sess.count_overlaps(), 27)

    def test_time(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "supersimple.cha"))

        self.assertEqual(len(sess), 2)

        self.assertEqual(sess.duration(), 60)
        self.assertEqual(sess.duration("PAR"), 15)
        self.assertEqual(sess.duration(sess.participants[0]), 15)
        self.assertEqual(sess.duration("INV"), 45)

        self.assertEqual(sess.words_per_minute(), 2)
        self.assertEqual(sess.words_per_minute("PAR"), 1 / 0.25)
        self.assertEqual(sess.words_per_minute("INV"), 1 / 0.75)

    def test_mean_length_of_utterance(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "meanLength.cha"))

        self.assertEqual(sess.mean_length_of_utterance(), 25 / 5)
        self.assertEqual(sess.mean_length_of_utterance("PAR"), 19 / 3)
        self.assertEqual(sess.mean_length_of_utterance("INV"), 6 / 2)

    def test_count_utterance_terminators(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "terminators.cha"))

        self.assertEqual(sess.count_utterance_terminators(), 3)
        self.assertEqual(sess.count_utterance_terminators("PAR"), 1)
        self.assertEqual(sess.count_utterance_terminators("INV"), 2)

    def test_count_filled_pauses(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))

        self.assertEqual(sess.count_filled_pauses(), 3)
        self.assertEqual(sess.count_filled_pauses("PAR"), 3)
        self.assertEqual(sess.count_filled_pauses("INV"), 0)

    def test_count_generic_errors(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "wordErrors.cha"))

        self.assertEqual(sess.count_generic_errors(), 1)
        self.assertEqual(sess.count_generic_errors("PAR"), 0)
        self.assertEqual(sess.count_generic_errors("INV"), 1)

    def test_count_semantic_errors(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "wordErrors.cha"))

        self.assertEqual(sess.count_semantic_errors(), 2)
        self.assertEqual(sess.count_semantic_errors("PAR"), 1)
        self.assertEqual(sess.count_semantic_errors("INV"), 1)

    def test_count_morphological_errors(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "wordErrors.cha"))

        self.assertEqual(sess.count_morphological_errors(), 1)
        self.assertEqual(sess.count_morphological_errors("PAR"), 1)
        self.assertEqual(sess.count_morphological_errors("INV"), 0)

    def test_count_dysfluency_errors(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "wordErrors.cha"))

        self.assertEqual(sess.count_dysfluency_errors(), 3)
        self.assertEqual(sess.count_dysfluency_errors("PAR"), 0)
        self.assertEqual(sess.count_dysfluency_errors("INV"), 3)

    def test_count_phonological_errors(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "wordErrors.cha"))

        self.assertEqual(sess.count_phonological_errors(), 1)
        self.assertEqual(sess.count_phonological_errors("PAR"), 0)
        self.assertEqual(sess.count_phonological_errors("INV"), 1)

    def test_count_neologisms(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "wordErrors.cha"))

        self.assertEqual(sess.count_neologisms(), 4)
        self.assertEqual(sess.count_neologisms("PAR"), 1)
        self.assertEqual(sess.count_neologisms("INV"), 3)

    def test_pos(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "simple.cha"))

        self.assertEqual(sess.count_free_morphemes(), 6)
        self.assertEqual(sess.count_closed_class_words(), 4)
        self.assertEqual(sess.count_open_class_words(), 2)
        self.assertEqual(sess.count_pronouns(), 3)
        self.assertEqual(sess.count_nouns(), 0)
        self.assertEqual(sess.count_bound_morphemes(), 4)

    def test_false_starts(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "falseStart.cha"))

        self.assertEqual(sess.count_false_starts(), 1)
        self.assertEqual(sess.count_false_starts("PAR"), 1)
        self.assertEqual(sess.count_false_starts("INV"), 0)

    def test_unintelligible(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "unintelligible.cha"))

        self.assertEqual(sess.count_unintelligible_sequences(), 2)
        self.assertEqual(sess.count_unintelligible_sequences("PAR"), 2)
        self.assertEqual(sess.count_unintelligible_sequences("INV"), 0)

    def test_to_chat(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "supersimple.cha"))
        tochat = sess.to_chat()
        self.assertTrue(tochat.pop(0).startswith("@UTF8"))
        self.assertTrue(tochat.pop(0).startswith("@Begin"))
        self.assertTrue(tochat.pop(0).startswith("@Languages:"))
        self.assertTrue(tochat.pop(0).startswith("@Participants:"))
        self.assertTrue(tochat.pop(0).startswith("@Options:"))
        self.assertTrue(tochat.pop(0).startswith("@ID:"))
        self.assertTrue(tochat.pop(0).startswith("@ID:"))
        self.assertTrue(tochat.pop(0).startswith("*INV:"))
        self.assertTrue(tochat.pop(0).startswith("%wor:"))
        self.assertTrue(tochat.pop(0).startswith("%mor:"))
        self.assertTrue(tochat.pop(0).startswith("*PAR:"))
        self.assertTrue(tochat.pop(0).startswith("%wor:"))
        self.assertTrue(tochat.pop(0).startswith("%mor:"))
        self.assertTrue(tochat.pop(0).startswith("@End"))

    def test_to_chat_verbatim(self):
        sess = Session.from_chat(os.path.join(test_data_cwd, "benchmark1.cha"))
        tochat = sess.to_chat()
        chat = """@UTF8
@Begin
@Languages:	eng
@Participants:	PAR Simple Participant, INV Investigator Investigator
@Options:	multi
@ID:	eng|Simple|PAR|22;00.|female|aphasia||Participant|||
@ID:	eng|Simple|INV|||||Investigator|||"""
        self.assertEqual(tochat[:7], chat.split("\n"))
        self.assertEqual(tochat[-1], "@End")


class TestUtterance(unittest.TestCase):
    def setUp(self):
        # hello (2.5) +...
        tokens = [
            Word("hello", WordType.NORMAL),
            Pause("(2.5)", PauseType.DURATION, duration=2.5),
            Terminator("+...", TerminatorType.TRAILING_OFF),
        ]
        speaker = Participant("PAR", "John", "Participant")
        self.utterance = Utterance(tokens, speaker, timemarks=(0, 1500))

    def test_utterance_initialization(self):
        self.assertEqual(len(self.utterance), 3)
        self.assertEqual(self.utterance.speaker.code, "PAR")
        self.assertEqual(self.utterance.speaker.name, "John")
        self.assertEqual(self.utterance.speaker.role, "Participant")
        self.assertEqual(self.utterance.timemarks, (0, 1500))
        self.assertEqual(len(self.utterance.markers), 0)

    def test_utterance_str(self):
        self.assertEqual(str(self.utterance), "hello (2.5) +...")
        self.assertEqual(repr(self.utterance), "hello (2.5) +...")

    def test_utterance_iter(self):
        tokens = list(iter(self.utterance))
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].text, "hello")
        self.assertEqual(tokens[1].text, "(2.5)")
        self.assertEqual(tokens[2].text, "+...")

    def test_utterance_getitem(self):
        token = self.utterance[0:5]
        self.assertEqual(token, "hello")

    def test_utterance_get_token_spans(self):
        spans = self.utterance.get_token_spans()
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0], (0, 5))
        self.assertEqual(spans[1], (6, 11))
        self.assertEqual(spans[2], (12, 16))

    def test_utterance_verbatim_string(self):
        verbatim = self.utterance.verbatim_string()
        self.assertEqual(verbatim, "hello (2.5)")

    def test_utterance_verbatim_tokens(self):
        tokens = self.utterance.verbatim_tokens()
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].text, "hello")
        self.assertEqual(tokens[1].text, "(2.5)")

    def test_utterance_markers(self):
        self.assertEqual(len(self.utterance.markers), 0)


class TestParticipant_dual_condition(unittest.TestCase):
    def setUp(self):
        self.participant = Participant(
            code="PAR",
            name="John",
            role="Participant",
            language="English",
            age="30;",
            sex="M",
            corpus="some_corpus",
            group="some_group",
            ethnicity="some_ethnicity",
            ses="some_ses",
            education="some_education",
        )

    def test_participant_initialization_eth_and_ses(self):
        self.assertEqual(self.participant.code, "PAR")
        self.assertEqual(self.participant.name, "John")
        self.assertEqual(self.participant.role, "Participant")
        self.assertEqual(self.participant.language, "English")
        self.assertEqual(self.participant.age, "30;")
        self.assertEqual(self.participant.sex, "M")
        self.assertEqual(self.participant.corpus, "some_corpus")
        self.assertEqual(self.participant.group, "some_group")
        self.assertEqual(self.participant.eth, "some_ethnicity")
        self.assertEqual(self.participant.ses, "some_ses")
        self.assertEqual(self.participant.education, "some_education")
        self.assertEqual(self.participant.custom, "")

    def test_participant_id_header_eth_and_ses(self):
        self.assertEqual(
            self.participant.id_header,
            "@ID:	English|some_corpus|PAR|30;|M|some_group|some_ethnicity,some_ses|Participant|some_education||",
        )

    def test_participant_age(self):
        self.assertEqual(self.participant.age, "30;")
        p2 = Participant("PAR", "John", "Participant", age="30;")
        self.assertEqual(p2.age, "30;")
        p3 = Participant("PAR", "John", "Participant", age="30;05")
        self.assertEqual(p3.age, "30;05.")
        p3 = Participant("PAR", "John", "Participant", age="30;5.")
        self.assertEqual(p3.age, "30;05.")
        p4 = Participant("PAR", "John", "Participant", age="30;5.2")
        self.assertEqual(p4.age, "30;05.02")
        p5 = Participant("PAR", "John", "Participant", age=(30, 11))
        self.assertEqual(p5.age, "30;11.")
        p6 = Participant("PAR", "John", "Participant", age=(30, 6, 22))
        self.assertEqual(p6.age, "30;06.22")
        p7 = Participant("PAR", "John", "Participant", age=30)
        self.assertEqual(p7.age, "30;")
        p8 = Participant("PAR", "John", "Participant", age=30.5)
        self.assertEqual(p8.age, "30;06.")
        p9 = Participant("PAR", "John", "Participant", age=(21,))
        self.assertEqual(p9.age, "21;")

        self.assertRaises(
            ValueError, Participant, "PAR", "John", "Participant", age=(-2, 1, 1)
        )
        self.assertRaises(
            ValueError, Participant, "PAR", "John", "Participant", age=(2, 13, 1)
        )
        self.assertRaises(
            ValueError, Participant, "PAR", "John", "Participant", age=(2, 1, 40)
        )


class TestParticipant(unittest.TestCase):
    def setUp(self):
        self.participant = Participant(
            code="PAR",
            name="John",
            role="Participant",
            language="English",
            age="30",
            sex="M",
            corpus="some_corpus",
            group="some_group",
            ethnicity="some_ethnicity",
            education="some_education",
        )

    def test_participant_initialization(self):
        self.assertEqual(self.participant.code, "PAR")
        self.assertEqual(self.participant.name, "John")
        self.assertEqual(self.participant.role, "Participant")
        self.assertEqual(self.participant.language, "English")
        self.assertEqual(self.participant.age, "30;")
        self.assertEqual(self.participant.sex, "M")
        self.assertEqual(self.participant.corpus, "some_corpus")
        self.assertEqual(self.participant.group, "some_group")
        self.assertEqual(self.participant.eth, "some_ethnicity")
        self.assertEqual(self.participant.ses, "")
        self.assertEqual(self.participant.education, "some_education")
        self.assertEqual(self.participant.custom, "")

    def test_participant_id_header(self):
        self.assertEqual(
            self.participant.id_header,
            "@ID:	English|some_corpus|PAR|30;|M|some_group|some_ethnicity|Participant|some_education||",
        )


class TestToken(unittest.TestCase):

    def test_token_initialization(self):
        token = Token("hello", TokenType.WORD)
        self.assertEqual(token.text, "hello")
        self.assertEqual(token.token_type, TokenType.WORD)
        self.assertIsNone(token.timemarks)
        self.assertEqual(token.markers, [])
        self.assertIsNone(token.pos)
        self.assertIsNone(token.affix)

    def test_word_initialization(self):
        word = Word("hello", WordType.NORMAL)
        self.assertEqual(word.text, "hello")
        self.assertEqual(word.token_type, TokenType.WORD)
        self.assertEqual(word.word_type, WordType.NORMAL)
        self.assertIsNone(word.target_word)
        self.assertEqual(word.word_error_code, [])
        self.assertEqual(word.word_error_type, [])

    def test_pause_initialization(self):
        pause = Pause("(2.5)", PauseType.DURATION, duration=2.5)
        self.assertEqual(pause.text, "(2.5)")
        self.assertEqual(pause.token_type, TokenType.PAUSE)
        self.assertEqual(pause.pause_type, PauseType.DURATION)
        self.assertEqual(pause.duration, 2.5)

    def test_pause_initialization_without_duration(self):
        with self.assertRaises(ValueError) as e:
            Pause("...", PauseType.DURATION)
        self.assertEqual(
            str(e.exception), "Duration must be provided for PauseType.DURATION"
        )

    def test_terminator_initialization(self):
        terminator = Terminator("+...", TerminatorType.TRAILING_OFF)
        self.assertEqual(terminator.text, "+...")
        self.assertEqual(terminator.token_type, TokenType.TERMINATOR)
        self.assertEqual(terminator.terminator, TerminatorType.TRAILING_OFF)

    def test_satelite_marker_initialization(self):
        marker = SateliteMarker("‡", SateliteMarkerType.PREFIX)
        self.assertEqual(marker.text, "‡")
        self.assertEqual(marker.token_type, TokenType.SAT_MARKER)
        self.assertEqual(marker.satelite_marker, SateliteMarkerType.PREFIX)

    def test_post_code_initialization(self):
        post_code = PostCode("es", PostCodeType.EMPTY_SPEECH)
        self.assertEqual(post_code.text, "es")
        self.assertEqual(post_code.token_type, TokenType.POST_CODE)
        self.assertEqual(post_code.post_code, PostCodeType.EMPTY_SPEECH)

    def test_token_str(self):
        token = Token("hello", TokenType.WORD)
        self.assertEqual(str(token), "hello")
        self.assertEqual(repr(token), "hello")
        self.assertEqual(token.get_chat_format(), "hello")

        token = Token("filler", TokenType.FILLER, timemarks=(0, 1000))
        self.assertEqual(str(token), "filler")
        self.assertEqual(repr(token), "filler")
        self.assertEqual(token.get_chat_format(), "&-filler")
        self.assertEqual(
            token.get_chat_format(timemarks=True), f"&-filler \x150_1000\x15"
        )

    def test_word_str(self):
        word = Word("hello", WordType.NORMAL)
        self.assertEqual(str(word), "hello")

        word = Word("hello", WordType.IPA_TRANSCRIPTION, timemarks=(0, 1000))
        self.assertEqual(word.get_chat_format(), "hello@u")
        self.assertEqual(
            word.get_chat_format(timemarks=True), f"hello@u \x150_1000\x15"
        )
        word.target_word = "hola"
        self.assertEqual(word.get_chat_format(), "hello@u [:hola]")
        self.assertEqual(
            word.get_chat_format(timemarks=True), f"hello@u [:hola] \x150_1000\x15"
        )
        word.word_error_type = [WordErrorType.SEMANTIC]
        self.assertEqual(word.get_chat_format(), "hello@u [:hola] [* s]")
        self.assertEqual(
            word.get_chat_format(timemarks=True),
            f"hello@u [:hola] [* s] \x150_1000\x15",
        )

    def test_pause_str(self):
        pause = Pause("...", PauseType.DURATION, duration=2.5)
        self.assertEqual(str(pause), "(2.5)")

        pause = Pause("...", PauseType.DURATION, duration=62.5)
        self.assertEqual(str(pause), "(1:2.5)")

        pause = Pause("...", PauseType.DURATION, duration=62)
        self.assertEqual(str(pause), "(1:2.)")

        pause = Pause("...", PauseType.DURATION, duration=0.5)
        self.assertEqual(str(pause), "(0.5)")

        pause = Pause("...", PauseType.DURATION, duration=2)
        self.assertEqual(str(pause), "(2.)")

    def test_pause_str_without_duration(self):
        pause = Pause(".", PauseType.MEDIUM)
        self.assertEqual(str(pause), "(..)")

    def test_terminator_str(self):
        terminator = Terminator("anything", TerminatorType.TRAILING_OFF)
        self.assertEqual(str(terminator), "+...")

    def test_satelite_marker_str(self):
        marker = SateliteMarker("anything", SateliteMarkerType.PREFIX)
        self.assertEqual(str(marker), "‡")

    def test_post_code_str(self):
        post_code = PostCode("anything", PostCodeType.EMPTY_SPEECH)
        self.assertEqual(str(post_code), "[+ es]")

    def test_token_get_chat_format(self):
        token = Token("hello", TokenType.WORD)
        self.assertEqual(token.get_chat_format(), "hello")

    def test_word_get_chat_format(self):
        word = Word("hello", WordType.NORMAL)
        self.assertEqual(word.get_chat_format(), "hello")

    def test_pause_get_chat_format(self):
        pause = Pause("anything", PauseType.DURATION, duration=2.5)
        self.assertEqual(pause.get_chat_format(), "(2.5)")
        pause = Pause("anything", PauseType.DURATION, duration=62.5)
        self.assertEqual(pause.get_chat_format(), "(1:2.5)")

    def test_pause_get_chat_format_without_duration(self):
        pause = Pause("anything", PauseType.MEDIUM)
        self.assertEqual(pause.get_chat_format(), "(..)")

    def test_terminator_get_chat_format(self):
        terminator = Terminator("anything", TerminatorType.TRAILING_OFF)
        self.assertEqual(terminator.get_chat_format(), "+...")

    def test_satelite_marker_get_chat_format(self):
        marker = SateliteMarker("anything", SateliteMarkerType.PREFIX)
        self.assertEqual(marker.get_chat_format(), "‡")

    def test_post_code_get_chat_format(self):
        post_code = PostCode("anything", PostCodeType.EMPTY_SPEECH)
        self.assertEqual(post_code.get_chat_format(), "[+ es]")


if __name__ == "__main__":
    unittest.main()
