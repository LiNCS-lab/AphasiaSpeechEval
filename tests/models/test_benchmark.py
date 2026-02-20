import os
import unittest
import pandas as pd

from aphasia_speech_eval import (
    Session,
    Benchmark,
    combined_asr_performance,
)
from tests.test_data import test_data_cwd


class TestBenchmark(unittest.TestCase):

    def test_init(self):
        truth = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))
        pred = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))

        bench1 = Benchmark(reference=truth, prediction=pred)
        self.assertIsInstance(bench1, Benchmark)
        self.assertIsInstance(bench1.comparison_table(), str)

        bench2 = Benchmark(
            reference=os.path.join(test_data_cwd, "pauses_fillers.cha"),
            prediction=os.path.join(test_data_cwd, "pauses_fillers.cha"),
        )
        self.assertIsInstance(bench2, Benchmark)
        self.assertIsInstance(bench2.comparison_table(), str)

    def test_asr_performance_no_diff(self):

        # Test with the same file, no differences expected

        truth = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))
        pred = Session.from_chat(os.path.join(test_data_cwd, "pauses_fillers.cha"))

        bench = Benchmark(reference=truth, prediction=pred)
        df = bench.calculate_asr_performance()

        self.assertTrue((df.loc[(slice(None), "CER"), "Error Rate"] == 0).all())
        self.assertTrue((df.loc[(slice(None), "WER"), "Error Rate"] == 0).all())

    def test_asr_performance_diff(self):

        # Test with a different file, differences expected

        truth = Session.from_chat(os.path.join(test_data_cwd, "supersimple.cha"))
        pred = Session.from_chat(os.path.join(test_data_cwd, "supersimple_diff.cha"))

        bench = Benchmark(reference=truth, prediction=pred)
        df = bench.calculate_asr_performance()

        self.assertEqual(df.loc["TOTAL", "CER"]["Error Rate"], 11 / 18)
        self.assertEqual(df.loc["TOTAL", "WER"]["Error Rate"], 4 / 5)
        self.assertEqual(df.loc["TOTAL", "CER"]["Total"], 18)
        self.assertEqual(df.loc["TOTAL", "WER"]["Total"], 5)
        self.assertEqual(df.loc["TOTAL", "CER"]["Substitution"], 2)
        self.assertEqual(df.loc["TOTAL", "WER"]["Substitution"], 2)
        self.assertEqual(df.loc["TOTAL", "CER"]["Deletion"], 2)
        self.assertEqual(df.loc["TOTAL", "WER"]["Deletion"], 1)
        self.assertEqual(df.loc["TOTAL", "CER"]["Insertion"], 7)
        self.assertEqual(df.loc["TOTAL", "WER"]["Insertion"], 1)
        self.assertEqual(df.loc["TOTAL", "CER"]["Equal"], 7)
        self.assertEqual(df.loc["TOTAL", "WER"]["Equal"], 1)

    def test_asr_performance_diff_speaker(self):

        # Test with a different file, differences expected.
        # Different results for different speakers

        truth = Session.from_chat(os.path.join(test_data_cwd, "supersimple.cha"))
        pred = Session.from_chat(os.path.join(test_data_cwd, "supersimple_diff.cha"))
        bench = Benchmark(reference=truth, prediction=pred)

        df = bench.calculate_asr_performance(speaker="PAR")
        self.assertEqual(df.loc["TOTAL", "CER"]["Error Rate"], 3 / 5)
        self.assertEqual(df.loc["TOTAL", "WER"]["Error Rate"], 2 / 2)
        self.assertEqual(df.loc["TOTAL", "CER"]["Total"], 5)
        self.assertEqual(df.loc["TOTAL", "WER"]["Total"], 2)
        self.assertEqual(df.loc["TOTAL", "CER"]["Substitution"], 1)
        self.assertEqual(df.loc["TOTAL", "WER"]["Substitution"], 1)
        self.assertEqual(df.loc["TOTAL", "CER"]["Deletion"], 2)
        self.assertEqual(df.loc["TOTAL", "WER"]["Deletion"], 1)
        self.assertEqual(df.loc["TOTAL", "CER"]["Insertion"], 0)
        self.assertEqual(df.loc["TOTAL", "WER"]["Insertion"], 0)
        self.assertEqual(df.loc["TOTAL", "CER"]["Equal"], 2)
        self.assertEqual(df.loc["TOTAL", "WER"]["Equal"], 0)

        df = bench.calculate_asr_performance(speaker="INV")
        self.assertEqual(df.loc["TOTAL", "CER"]["Error Rate"], 8 / 12)
        self.assertEqual(df.loc["TOTAL", "WER"]["Error Rate"], 2 / 3)
        self.assertEqual(df.loc["TOTAL", "CER"]["Total"], 12)
        self.assertEqual(df.loc["TOTAL", "WER"]["Total"], 3)
        self.assertEqual(df.loc["TOTAL", "CER"]["Substitution"], 1)
        self.assertEqual(df.loc["TOTAL", "WER"]["Substitution"], 1)
        self.assertEqual(df.loc["TOTAL", "CER"]["Deletion"], 0)
        self.assertEqual(df.loc["TOTAL", "WER"]["Deletion"], 0)
        self.assertEqual(df.loc["TOTAL", "CER"]["Insertion"], 7)
        self.assertEqual(df.loc["TOTAL", "WER"]["Insertion"], 1)
        self.assertEqual(df.loc["TOTAL", "CER"]["Equal"], 4)
        self.assertEqual(df.loc["TOTAL", "WER"]["Equal"], 1)

    def test_asr_performance_counts(self):

        truth = Session.from_chat(os.path.join(test_data_cwd, "benchmark.cha"))
        pred = Session.from_chat(os.path.join(test_data_cwd, "benchmark.cha"))

        bench = Benchmark(reference=truth, prediction=pred)

        df = bench.calculate_asr_performance()
        self.assertEqual(df.loc["PAUSE", "CER"]["Total"], 5)
        self.assertEqual(df.loc["PAUSE", "WER"]["Total"], 1)

        self.assertEqual(df.loc["PHONOLOGICAL_FRAGMENT", "CER"]["Total"], 8)
        self.assertEqual(df.loc["PHONOLOGICAL_FRAGMENT", "WER"]["Total"], 1)

        self.assertEqual(df.loc["FILLER", "CER"]["Total"], 6)
        self.assertEqual(df.loc["FILLER", "WER"]["Total"], 1)

        self.assertEqual(df.loc["NON_WORD", "CER"]["Total"], 7)
        self.assertEqual(df.loc["NON_WORD", "WER"]["Total"], 1)

        self.assertEqual(df.loc["WORD_ERROR-SEMANTIC", "CER"]["Total"], 9)
        self.assertEqual(df.loc["WORD_ERROR-SEMANTIC", "WER"]["Total"], 1)

        self.assertEqual(df.loc["JARGON", "CER"]["Total"], 22)
        self.assertEqual(df.loc["JARGON", "WER"]["Total"], 3)

        self.assertEqual(df.loc["GRAMMATICAL", "CER"]["Total"], 22)
        self.assertEqual(df.loc["GRAMMATICAL", "WER"]["Total"], 3)

        self.assertEqual(df.loc["PAUSE", "CER"]["Error Rate"], 0)
        self.assertEqual(df.loc["PHONOLOGICAL_FRAGMENT", "CER"]["Error Rate"], 0)
        self.assertEqual(df.loc["FILLER", "CER"]["Error Rate"], 0)
        self.assertEqual(df.loc["NON_WORD", "CER"]["Error Rate"], 0)
        self.assertEqual(df.loc["WORD_ERROR-SEMANTIC", "CER"]["Error Rate"], 0)
        self.assertEqual(df.loc["JARGON", "CER"]["Error Rate"], 0)
        self.assertEqual(df.loc["GRAMMATICAL", "CER"]["Error Rate"], 0)

        self.assertEqual(df.loc["PAUSE", "WER"]["Error Rate"], 0)
        self.assertEqual(df.loc["PHONOLOGICAL_FRAGMENT", "WER"]["Error Rate"], 0)
        self.assertEqual(df.loc["FILLER", "WER"]["Error Rate"], 0)
        self.assertEqual(df.loc["NON_WORD", "WER"]["Error Rate"], 0)
        self.assertEqual(df.loc["WORD_ERROR-SEMANTIC", "WER"]["Error Rate"], 0)
        self.assertEqual(df.loc["JARGON", "WER"]["Error Rate"], 0)
        self.assertEqual(df.loc["GRAMMATICAL", "WER"]["Error Rate"], 0)

    # def test_combined_asr(self):

    # TODO: Fix this test. Need to recount errors for each file.

    #     truth = Session.from_chat(os.path.join(test_data_cwd, "benchmark1.cha"))
    #     pred1 = Session.from_chat(os.path.join(test_data_cwd, "benchmark1_diff1.cha"))
    #     pred2 = Session.from_chat(os.path.join(test_data_cwd, "benchmark1_diff2.cha"))
    #     pred3 = Session.from_chat(os.path.join(test_data_cwd, "benchmark1_diff3.cha"))

    #     bench1 = Benchmark(reference=truth, prediction=pred1)  # Filler
    #     bench2 = Benchmark(reference=truth, prediction=pred2)  # Normal Word
    #     bench3 = Benchmark(reference=truth, prediction=pred3)  # Jargon, Grammatical

    #     df, plt, df_meta = combined_asr_performance([bench1, bench2, bench3])

    #     error_rate_avg = ((1 - (26 / 36)) + (1 - (22 / 37)) + (1 - (14 / 27))) / 3
    #     variance = (
    #         sum(
    #             [
    #                 abs(((1 - (26 / 36)) - error_rate_avg)) ** 2,
    #                 abs(((1 - (22 / 37)) - error_rate_avg)) ** 2,
    #                 abs(((1 - (14 / 27)) - error_rate_avg)) ** 2,
    #             ]
    #         )
    #         / 3
    #     )
    #     weighted_error_rate = 1 - ((26 + 22 + 14) / (36 + 37 + 27))
    #     weigthed_variance = sum(
    #         [
    #             36 * (abs(((1 - (26 / 36)) - error_rate_avg)) ** 2),
    #             37 * (abs(((1 - (22 / 37)) - error_rate_avg)) ** 2),
    #             27 * (abs(((1 - (14 / 27)) - error_rate_avg)) ** 2),
    #         ]
    #     ) / (36 + 37 + 27)

    #     self.assertAlmostEqual(
    #         df_meta.loc["*All", "TOTAL", "CER"]["Error Rate avg"], error_rate_avg
    #     )
    #     self.assertAlmostEqual(
    #         df_meta.loc["*All", "TOTAL", "CER"]["Variance"], variance
    #     )
    #     self.assertAlmostEqual(
    #         df_meta.loc["*All", "TOTAL", "CER"]["Weighted Error Rate"],
    #         weighted_error_rate,
    #     )
    #     self.assertAlmostEqual(
    #         df_meta.loc["*All", "TOTAL", "CER"]["Weighted Variance"], weigthed_variance
    #     )


if __name__ == "__main__":
    unittest.main()
