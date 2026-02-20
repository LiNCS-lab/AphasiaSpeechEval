from typing import List, Dict, Union, Tuple
import copy

import numpy as np
from tabulate import tabulate
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz.distance import Levenshtein, Opcodes
import pandas as pd

from .session import Participant, Session, Utterance, Token
from .types import (
    TokenType,
    WordType,
    WordErrorType,
    MarkerType,
    PostCodeType,
    PauseType,
    TerminatorType,
    VERBATIM_TOKEN_TYPES,
)


class Benchmark:
    """
    Class representing a benchmark for comparing reference and prediction sessions.

    Args:
        reference (Union[Session, str]): The reference session or the path to the reference chat file.
        prediction (Union[Session, str]): The predicted session or the path to the predicted chat file.
        corpus (str, optional): The corpus name. Defaults to None.
        verbose (bool, optional): Whether to show verbose output. Defaults to True.

    Attributes:
        reference_session (Session): The reference session.
        prediction_session (Session): The predicted session.
        metrics (dict): The benchmark metrics.

    Methods:
        __str__(): Returns a string representation of the benchmark metrics.
        comparison_table(): Returns an aligned comparison table of the reference and prediction sessions.
    """

    def __init__(
        self,
        reference: Union[Session, str],
        prediction: Union[Session, str],
        corpus: str = "",
        verbose=True,
    ) -> None:
        """
        Initializes a Benchmark object.

        Args:
            reference (Session or str): The reference session or the path to the reference chat file.
            prediction (Session or str): The predicted session or the path to the predicted chat file.
        """
        if isinstance(reference, str):
            reference = Session.from_chat(reference)
        if isinstance(prediction, str):
            prediction = Session.from_chat(prediction)

        self.reference_session = copy.deepcopy(reference)
        self.prediction_session = copy.deepcopy(prediction)
        self.corpus = corpus

        if verbose:
            # Show warning if the reference and prediction sessions have different participants
            ref_participants = [p.code for p in self.reference_session.participants]
            pred_participants = [p.code for p in self.prediction_session.participants]
            if set(ref_participants) != set(pred_participants):
                print(
                    f"Warning: The reference and prediction sessions have different participant codes. The benchmark metrics concerning speakers may not be accurate.",
                    f"  Reference participants: {ref_participants}",
                    f"  Prediction participants: {pred_participants}",
                    sep="\n",
                )

    def comparison_table(self, token_filter: callable = None, chat_format=True) -> str:
        """
        Generate a comparison table between the reference session and the prediction session.

        Returns:
            str: A formatted table displaying the speaker, utterance, and timemarks for both the reference and prediction sessions.
        """

        table = []

        ref_copy = copy.deepcopy(self.reference_session)
        pred_copy = copy.deepcopy(self.prediction_session)

        ref_copy, pred_copy = self._align_utterances(ref_copy, pred_copy)

        # pad the shorter session with empty utterances
        while len(ref_copy) < len(pred_copy):
            ref_copy.utterances.append(Utterance([], "", (0, 0)))
        while len(pred_copy) < len(ref_copy):
            pred_copy.utterances.append(Utterance([], "", (0, 0)))

        for ref_utt, pred_utt in zip(ref_copy, pred_copy):
            if token_filter:
                if not token_filter(ref_utt):
                    continue
            if chat_format:
                table.append(
                    [
                        ref_utt.speaker,
                        " ".join([t.get_chat_format() for t in ref_utt.tokens]),
                        ref_utt.timemarks,
                        pred_utt.speaker,
                        " ".join([t.get_chat_format() for t in pred_utt.tokens]),
                        pred_utt.timemarks,
                    ]
                )
            else:
                table.append(
                    [
                        ref_utt.speaker,
                        ref_utt,
                        ref_utt.timemarks,
                        pred_utt.speaker,
                        pred_utt,
                        pred_utt.timemarks,
                    ]
                )
        if table == []:
            return "No matching utterances found."
        return tabulate(
            table,
            headers=[
                "Ref Spkr",
                "Reference Utterance",
                "Ref Time",
                "Pred Spkr",
                "Predicted Utterance",
                "Pred Time",
            ],
            maxcolwidths=[None, 60, None, None, 60, None],
        )

    ############################
    # Alignment methods
    ############################

    def _find_nearest_timemarks(
        self, reference_word_boundaries: List[int], pred_timemarks: List[int]
    ) -> List[int]:
        """
        Finds the nearest timemarks in the reference session for the given prediction timemarks.

        Args:
            reference_word_boundaries (list): The reference word boundaries.
            pred_timemarks (list): The prediction timemarks.

        Returns:
            list: The nearest timemarks in the reference session for the given prediction timemarks.
        """
        nearest_timemarks = [None, None]
        for i, timemark in enumerate(pred_timemarks):
            nearest_timemark = min(
                reference_word_boundaries, key=lambda x: abs(x - timemark)
            )
            nearest_timemarks[i] = nearest_timemark
        return nearest_timemarks

    def _align_utterances(
        self, reference_session: Session, prediction_session: Session
    ) -> Tuple[Session, Session]:
        """
        Aligns the utterances in the reference and prediction sessions.
        """
        # Get the reference word boundaries
        reference_word_boundaries = []
        for ref_utt in reference_session:
            for ref_token in ref_utt:
                if ref_token.timemarks:
                    reference_word_boundaries.append(ref_token.timemarks[0])
                    reference_word_boundaries.append(ref_token.timemarks[1])
        reference_word_boundaries = sorted(reference_word_boundaries)

        # Modify the prediction timemarks to the nearest reference timemarks
        for pred_utt in prediction_session:
            for pred_token in pred_utt:
                if pred_token.timemarks:
                    pred_token.timemarks = self._find_nearest_timemarks(
                        reference_word_boundaries, pred_token.timemarks
                    )

        # Get the reference utterance boundaries
        reference_utterance_boundaries = []
        for ref_utt in reference_session:
            if ref_utt.timemarks:
                reference_utterance_boundaries.append(ref_utt.timemarks[1])
        reference_utterance_boundaries = sorted(reference_utterance_boundaries)

        # Create new utterances with the aligned timemarks of the reference utterances
        pred_tokens = []
        for pred_utt in prediction_session:
            for pred_token in pred_utt:
                speaker = pred_utt.speaker
                pred_tokens.append((pred_token, speaker))

        aligned_prediction_utterances = []
        tokens = []
        speaker_of_tokens = []
        while pred_tokens:
            pred_token = pred_tokens.pop(0)
            if pred_token[0].timemarks:
                if pred_token[0].timemarks[1] <= reference_utterance_boundaries[0]:
                    tokens.append(pred_token[0])
                    speaker_of_tokens.append(pred_token[1])
                else:
                    # get earliest timemark in token list
                    earliest_timemark = None
                    for token in tokens:
                        if token.timemarks:
                            if (
                                earliest_timemark is None
                                or token.timemarks[0] < earliest_timemark
                            ):
                                earliest_timemark = token.timemarks[0]
                    latest_timemark = None
                    for token in tokens:
                        if token.timemarks:
                            if (
                                latest_timemark is None
                                or token.timemarks[1] > latest_timemark
                            ):
                                latest_timemark = token.timemarks[1]
                    speaker = max(set(speaker_of_tokens), key=speaker_of_tokens.count)
                    aligned_prediction_utterances.append(
                        Utterance(
                            tokens,
                            speaker,
                            timemarks=(earliest_timemark, latest_timemark),
                        )
                    )
                    tokens = [pred_token[0]]
                    speaker_of_tokens = [pred_token[1]]
                    reference_utterance_boundaries.pop(0)
            else:
                tokens.append(pred_token[0])
                speaker_of_tokens.append(pred_token[1])

        prediction_session = Session(
            prediction_session.name,
            aligned_prediction_utterances,
            prediction_session.participants,
            prediction_session.languages,
            prediction_session.header_media,
            prediction_session.header_options,
        )

        return reference_session, prediction_session

    ############################
    # Metrics calculation #
    ############################

    def calculate_asr_performance(
        self,
        speaker: str | Participant | List[str] | List[Participant] = None,
        included_token_types: List[TokenType] = VERBATIM_TOKEN_TYPES,
        raw=False,
    ) -> pd.DataFrame:
        """
        For each token type in reference, see in prediction if the text is present regardless of token type.
        Ex. A phonological fragment in reference is present or not in prediction as any token type.
        This is to evaluate the ASR performance regardless of the tokenization applied to the transcription.

        Args:
            speaker (str or Participant or list of str or list of Participant, optional): The speaker code or Participant object for which to calculate the ASR performance. Defaults to using all speakers.
            included_token_types (list of TokenType, optional): The token types to include in the ASR performance calculation. Defaults to VERBATIM_TOKEN_TYPES from types.py.
            raw (bool, optional): Returns all rows, including for token types with no errors. Defaults to False.

        Returns:
            dataframe: A dataframe containing the CER and WER by reference token types.
        """
        mega_ref_utt = Utterance([], "", (0, 0))
        mega_pred_utt = Utterance([], "", (0, 0))

        # Combine all tokens in the reference and prediction sessions into single utterances

        ref_utts = self.reference_session.get_participant_utts(speaker)
        pred_utts = self.prediction_session.get_participant_utts(speaker)

        for ref_utt in ref_utts:
            mega_ref_utt.tokens.extend(
                ref_utt.verbatim_tokens(types=VERBATIM_TOKEN_TYPES)
            )
        for pred_utt in pred_utts:
            mega_pred_utt.tokens.extend(
                pred_utt.verbatim_tokens(types=VERBATIM_TOKEN_TYPES)
            )

        # get string representations of the mega utterances
        str_ref = str(mega_ref_utt)
        str_pred = str(mega_pred_utt)
        str_ref_list = [str(token).casefold() for token in mega_ref_utt]
        str_pred_list = [str(token).casefold() for token in mega_pred_utt]

        # get the edit ops between the reference and prediction strings
        ops_cer = Levenshtein.opcodes(
            str_ref, str_pred, processor=lambda x: x.casefold()
        )
        ops_wer = Levenshtein.opcodes(str_ref_list, str_pred_list)

        # get the token spans and their types for the reference utterance
        token_spans_cer = self._get_token_type_spans(mega_ref_utt)
        token_spans_wer = self._get_token_type_spans(mega_ref_utt, wer=True)

        # Calculate CER by reference token types
        cer_op_str, cer_ref_spans = self._custom_string_comparison(
            str_ref, str_pred, ops_cer, token_spans_cer, wer=False
        )
        cer = self._cer_by_reference_types(
            cer_op_str, cer_ref_spans, included_token_types
        )

        # Calculate WER by reference token types
        wer_op_str, wer_ref_spans = self._custom_string_comparison(
            str_ref_list, str_pred_list, ops_wer, token_spans_wer, wer=True
        )
        wer = self._wer_by_reference_types(
            wer_op_str, wer_ref_spans, included_token_types
        )

        # Create dataframe for CER and WER:
        # Header: multiindex(Token Type, Measure (CER or WER)), Error Rate, Total, Equal, Substitution, Deletion, Insertion
        rows = []
        for k, v in cer.items():
            rows.append(
                [
                    k,
                    "CER",
                    v["equal"],
                    v["substitution"],
                    v["deletion"],
                    v["insertion"],
                ]
            )
            rows.append(
                [
                    k,
                    "WER",
                    wer[k]["equal"],
                    wer[k]["substitution"],
                    wer[k]["deletion"],
                    wer[k]["insertion"],
                ]
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "Token Type",
                "Measure",
                "Equal",
                "Substitution",
                "Deletion",
                "Insertion",
            ],
        )
        df.set_index(["Token Type", "Measure"], inplace=True)

        # compute columns for error rate and total
        df["Total"] = (
            df["Equal"] + df["Substitution"] + df["Deletion"] + df["Insertion"]
        )
        df["Error Rate"] = (df["Substitution"] + df["Deletion"] + df["Insertion"]) / df[
            "Total"
        ]
        if not raw:
            df = df.dropna()
            return df
        return df

    def _cer_by_reference_types(self, ops, ref_spans, included_token_types):
        """
        Calculate the CER by reference token types.
        """

        def add_cer(op_string, key, cer_dict):
            e_sum = op_string.count("E")
            s_sum = op_string.count("S")
            d_sum = op_string.count("D")
            i_sum = op_string.count("I")
            cer_dict[key] = {
                "equal": e_sum,
                "substitution": s_sum,
                "deletion": d_sum,
                "insertion": i_sum,
            }

        cer_dict = {}

        # Total CER for all tokens
        add_cer(ops, "TOTAL", cer_dict)

        for tokentype in included_token_types:

            # CER by each token type
            op_string = ""
            for span in ref_spans:
                if span["token"].token_type == tokentype:
                    op_string += ops[span["start"] : span["end"]]

            add_cer(op_string, tokentype.name, cer_dict)

            if tokentype == TokenType.WORD:

                # CER by each word type
                for word_type in WordType:
                    op_string = ""
                    for span in ref_spans:
                        if (
                            span["token"].token_type == TokenType.WORD
                            and span["token"].word_type == word_type
                        ):
                            op_string += ops[span["start"] : span["end"]]
                    # if len(op_string) == 0:
                    #     continue
                    add_cer(op_string, "WORD-" + word_type.name, cer_dict)

                # CER by each word error type
                for error_type in WordErrorType:
                    op_string = ""
                    for span in ref_spans:
                        if (
                            span["token"].token_type == TokenType.WORD
                            and error_type in span["token"].word_error_type
                        ):
                            op_string += ops[span["start"] : span["end"]]
                    # if len(op_string) == 0:
                    #     continue
                    add_cer(op_string, "WORD_ERROR-" + error_type.name, cer_dict)

        for marker_type in MarkerType:

            # CER by each marker type
            op_string = ""
            for span in ref_spans:
                if marker_type in span["token"].markers:
                    op_string += ops[span["start"] : span["end"]]

            add_cer(op_string, marker_type.name, cer_dict)

        for postcode_type in PostCodeType:

            # CER by each post code type
            op_string = ""
            for span in ref_spans:
                if (
                    postcode_type in span["token"].utterance.post_codes
                    and span["token"].token_type in VERBATIM_TOKEN_TYPES
                ):
                    op_string += ops[span["start"] : span["end"]]

            add_cer(op_string, postcode_type.name, cer_dict)

        return cer_dict

    def _wer_by_reference_types(self, ops, ref_spans, included_token_types) -> dict:

        def add_wer(op_string, key, wer_dict):
            e_sum = op_string.count("E")
            s_sum = op_string.count("S")
            d_sum = op_string.count("D")
            i_sum = op_string.count("I")
            wer_dict[key] = {
                "equal": e_sum,
                "substitution": s_sum,
                "deletion": d_sum,
                "insertion": i_sum,
            }

        wer_dict = {}

        # Total WER for all tokens
        add_wer(ops, "TOTAL", wer_dict)

        for tokentype in included_token_types:

            # WER by each token type
            op_string = ""
            for span in ref_spans:
                if span[1].token_type == tokentype:
                    op_string += ops[span[0]]

            add_wer(op_string, tokentype.name, wer_dict)

            if tokentype == TokenType.WORD:

                # WER by each word type
                for word_type in WordType:
                    op_string = ""
                    for span in ref_spans:
                        if (
                            span[1].token_type == TokenType.WORD
                            and span[1].word_type == word_type
                        ):
                            op_string += ops[span[0]]
                    # if len(op_string) == 0:
                    #     continue
                    add_wer(op_string, "WORD-" + word_type.name, wer_dict)

                # WER by each word error type
                for error_type in WordErrorType:
                    op_string = ""
                    for span in ref_spans:
                        if (
                            span[1].token_type == TokenType.WORD
                            and error_type in span[1].word_error_type
                        ):
                            op_string += ops[span[0]]
                    # if len(op_string) == 0:
                    #     continue
                    add_wer(op_string, "WORD_ERROR-" + error_type.name, wer_dict)

        for marker_type in MarkerType:

            # WER by each marker type
            op_string = ""
            for span in ref_spans:
                if marker_type in span[1].markers:
                    op_string += ops[span[0]]

            add_wer(op_string, marker_type.name, wer_dict)

        for postcode_type in PostCodeType:

            # WER by each post code type
            op_string = ""
            for span in ref_spans:
                if (
                    postcode_type in span[1].utterance.post_codes
                    and span[1].token_type in VERBATIM_TOKEN_TYPES
                ):
                    op_string += ops[span[0]]

            add_wer(op_string, postcode_type.name, wer_dict)

        return wer_dict

    def _get_token_type_spans(
        self, tokens, wer=False
    ) -> List[Dict[str, Union[int, Token]]]:

        if wer:
            # return spans based on token index
            spans = []
            for idx, token in enumerate(tokens):
                spans.append([idx, token])
            return spans
        spans = []
        for token in tokens:
            if len(spans) == 0:
                spans.append({"start": 0, "end": len(str(token)), "token": token})
            else:
                spans.append(
                    {
                        "start": spans[-1]["end"] + 1,
                        "end": spans[-1]["end"] + 1 + len(str(token)),
                        "token": token,
                    }
                )
        return spans

    def _custom_string_comparison(
        self,
        reference: str | List[str],
        hypothesis: str | List[str],
        ops: Opcodes,
        ref_spans: List[Dict[str, Union[int, Token]]] = None,
        wer=False,
    ) -> str:
        """
        Method adapted from jiwer library (alignment.py)
        """
        ref_str = ""
        hyp_str = ""
        op_str = ""

        spans_copy = copy.deepcopy(ref_spans)
        for op in ops:
            if op.tag == "equal" or op.tag == "replace":
                ref = reference[op.src_start : op.src_end]
                hyp = hypothesis[op.dest_start : op.dest_end]
                op_char = "e" if op.tag == "equal" else "s"
            elif op.tag == "delete":
                ref = reference[op.src_start : op.src_end]
                hyp = ["*" for _ in range(len(ref))]
                op_char = "d"
            elif op.tag == "insert":
                hyp = hypothesis[op.dest_start : op.dest_end]
                ref = ["*" for _ in range(len(hyp))]
                op_char = "i"
                if wer:
                    for span, span_copy in zip(ref_spans, spans_copy):
                        if span[0] >= op.src_start:
                            span_copy[0] += 1
                else:
                    if ref_spans:
                        for span, span_copy in zip(ref_spans, spans_copy):
                            if span["start"] >= op.src_start:
                                span_copy["start"] += len(hyp)
                                span_copy["end"] += len(hyp)
                            elif span["end"] >= op.src_start:
                                # insertion in middle of token
                                span_copy["end"] += len(hyp)

            op_chars = [op_char for _ in range(len(ref))]
            for rf, hp, c in zip(ref, hyp, op_chars):
                str_len = max(len(rf), len(hp), len(c))

                if rf == "*":
                    rf = "".join(["*"] * str_len)
                elif hp == "*":
                    hp = "".join(["*"] * str_len)

                ref_str += f"{rf:>{str_len}}"
                hyp_str += f"{hp:>{str_len}}"
                op_str += f"{c.upper():>{str_len}}"

        return op_str.replace(" ", ""), spans_copy


############################
# Benchmark Combination Functions
############################


def combined_asr_performance(
    benchmarks: List[Benchmark],
    participant=None,
    include_token_types=VERBATIM_TOKEN_TYPES,
):
    """
    Combines the ASR performance of multiple benchmarks as a per session average across all benchmarks.
    This function calculates the average Word Error Rate (WER) and Character Error Rate (CER) across all benchmarks, every session is weighted equally.

    Returns a dataframe of all results.
    Dataframe structure:
    - Index: Corpus, Session, Token Type, Measure (CER or WER)
    - Columns: Equal, Substitution, Deletion, Insertion, Total, Error Rate

    Return dataframe of the computed fields (average and sum) for each corpus and all corpora combined.

    Also returns boxplots of the CER and WER by reference token types.

    Args:
        benchmarks (List[Benchmark]): A list of benchmarks.
        participant (str or Participant or list of str or list of Participant, optional): The speaker code or Participant object for which to calculate the ASR performance. Defaults to using all speakers.
        include_token_types (list of TokenType, optional): The token types to include in the ASR performance calculation. Defaults to VERBATIM_TOKEN_TYPES from types.py.

    Returns:
        df: A dataframe containing the CER and WER by reference token types for each session.
        plt: A scatter plot of the CER by reference token types.
        df_computed_fields: A dataframe containing the computed fields (average and sum) for each corpus and all corpora combined.
    """

    ######### DATAFRAME #########

    df_list = []
    for benchmark in benchmarks:
        df = benchmark.calculate_asr_performance(
            speaker=participant, included_token_types=include_token_types, raw=False
        )
        # add corpus and session name to the dataframe index
        df.index = pd.MultiIndex.from_tuples(
            [
                (benchmark.corpus, benchmark.prediction_session.name, type, measure)
                for type, measure in df.index
            ]
        )
        # add column with PAR participant condition
        participants = benchmark.reference_session.participants
        target = "PAR"
        if participants:
            participant = next((p for p in participants if p.code == target), None)
        df["Group"] = participant.group if (participant and participant.group) else None
        df["Age"] = participant.age if (participant and participant.age) else None
        df["Gender"] = participant.sex if (participant and participant.sex) else None

        #### Generally, in AphasiaBank the Aphasia Quotient is provided in the custom field
        df["Aphasia Quotient"] = (
            participant.custom if (participant and participant.custom) else None
        )

        df_list.append(df)
    df = pd.concat(df_list)

    # name the index levels
    df.index.names = ["Corpus", "Session", "Token Type", "Measure"]

    p_info_cols = ["Group", "Age", "Gender", "Aphasia Quotient"]

    df_p_info = df.copy()[p_info_cols]
    df = df.drop(columns=p_info_cols)

    ######### COMPUTED FIELDS #########

    ## Averages per corpus
    df_temp = df.groupby(["Corpus", "Token Type", "Measure"]).mean()
    df_temp = df_temp.add_suffix(" avg")

    # calculate variance across sessions
    for corpus in df_temp.index.get_level_values("Corpus").unique():
        for token_type in df_temp.index.get_level_values("Token Type").unique():
            for measure in df_temp.index.get_level_values("Measure").unique():
                df_temp.loc[(corpus, token_type, measure), "Variance"] = df.xs(
                    (corpus, token_type, measure),
                    level=["Corpus", "Token Type", "Measure"],
                )["Error Rate"].var(ddof=0)
    df_corpus_avgs = df_temp

    ## Sum per corpus
    df_temp = df.groupby(["Corpus", "Token Type", "Measure"]).sum()
    df_temp = df_temp.add_suffix(" sum")
    df_temp.drop(columns="Error Rate sum", inplace=True)

    # recalculate error rate
    df_temp["Weighted Error Rate"] = (
        df_temp["Substitution sum"] + df_temp["Deletion sum"] + df_temp["Insertion sum"]
    ) / df_temp["Total sum"]

    # calculate weighted variance across sessions
    for corpus in df_temp.index.get_level_values("Corpus").unique():
        for token_type in df_temp.index.get_level_values("Token Type").unique():
            for measure in df_temp.index.get_level_values("Measure").unique():
                error_rates = df.xs(
                    (corpus, token_type, measure),
                    level=["Corpus", "Token Type", "Measure"],
                )["Error Rate"]
                weights = df.xs(
                    (corpus, token_type, measure),
                    level=["Corpus", "Token Type", "Measure"],
                )["Total"]

                mean_error_rate = error_rates.mean()
                if np.isnan(mean_error_rate):
                    continue
                df_temp.loc[(corpus, token_type, measure), "Weighted Variance"] = (
                    np.sum(weights * ((error_rates - mean_error_rate) ** 2))
                    / np.sum(weights)
                )
    df_corpus_avgs = df_corpus_avgs.join(df_temp)

    ## Averages for all corpora
    df_temp = df.groupby(["Token Type", "Measure"]).mean()
    df_temp = df_temp.add_suffix(" avg")

    # calculate variance across all sessions
    for token_type in df_temp.index.get_level_values("Token Type").unique():
        for measure in df_temp.index.get_level_values("Measure").unique():
            df_temp.loc[(token_type, measure), "Variance"] = df.xs(
                (token_type, measure), level=["Token Type", "Measure"]
            )["Error Rate"].var(ddof=0)

    # Variance NaN to 0
    df_temp["Variance"] = df_temp["Variance"].fillna(0)
    df_total = df_temp

    ## Sum for all corpora
    df_temp = df.groupby(["Token Type", "Measure"]).sum()
    df_temp = df_temp.add_suffix(" sum")
    df_temp.drop(columns="Error Rate sum", inplace=True)

    # recalculate error rate
    df_temp["Weighted Error Rate"] = (
        df_temp["Substitution sum"] + df_temp["Deletion sum"] + df_temp["Insertion sum"]
    ) / df_temp["Total sum"]
    # calculate weighted variance across all sessions
    for token_type in df_temp.index.get_level_values("Token Type").unique():
        for measure in df_temp.index.get_level_values("Measure").unique():
            error_rates = df.xs((token_type, measure), level=["Token Type", "Measure"])[
                "Error Rate"
            ]
            weights = df.xs((token_type, measure), level=["Token Type", "Measure"])[
                "Total"
            ]
            mean_error_rate = error_rates.mean()
            if np.isnan(mean_error_rate):
                continue
            df_temp.loc[(token_type, measure), "Weighted Variance"] = np.sum(
                weights * ((error_rates - mean_error_rate) ** 2)
            ) / np.sum(weights)

    df_total = df_total.join(df_temp)
    df_total.index = pd.MultiIndex.from_tuples(
        [("*All", type, measure) for type, measure in df_total.index]
    )

    df_computed_fields = pd.concat([df_corpus_avgs, df_total])
    df_computed_fields.index.names = ["Corpus", "Token Type", "Measure"]

    # remove any rows with NaN values
    df_computed_fields = df_computed_fields.dropna()
    df = df.dropna()
    # add participant info back to df
    df = df.join(df_p_info)

    ######### PLOT #########

    plt.figure(figsize=(12, 12))

    cer_series = df.loc[(slice(None), slice(None), (slice(None)), "CER"), :]
    wer_series = df.loc[(slice(None), slice(None), (slice(None)), "WER"), :]

    # seperate multiindex into columns
    cer_series.reset_index(inplace=True)
    wer_series.reset_index(inplace=True)

    cer_order = (
        cer_series.groupby("Token Type")["Error Rate"].median().sort_values().index
    )
    wer_order = (
        wer_series.groupby("Token Type")["Error Rate"].median().sort_values().index
    )

    # Plot 1 - CER by reference token types
    plt.subplot(2, 2, 1)
    sns.boxplot(
        x=cer_series["Error Rate"],
        y=cer_series["Token Type"],
        orient="h",
        order=cer_order,
    )
    plt.title("CER by reference token types")
    plt.xlabel("CER")
    plt.ylabel("Reference token types")
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Plot 2 - Total characters by reference token types
    plt.subplot(2, 2, 2)

    sns.boxplot(
        x=cer_series["Total"], y=cer_series["Token Type"], orient="h", order=cer_order
    )
    plt.xscale("log")
    plt.title("Total characters by reference token types")
    plt.xlabel("Total characters")
    ticks, _ = plt.yticks()
    plt.yticks(ticks, ["" for _ in ticks])
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Plot 3 - WER by reference token types
    plt.subplot(2, 2, 3)
    sns.boxplot(
        x=wer_series["Error Rate"],
        y=wer_series["Token Type"],
        orient="h",
        order=wer_order,
    )
    plt.title("WER by reference token types")
    plt.xlabel("WER")
    plt.ylabel("Reference token types")
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Plot 4 - Total words by reference token types
    plt.subplot(2, 2, 4)
    sns.boxplot(
        x=wer_series["Total"], y=wer_series["Token Type"], orient="h", order=wer_order
    )
    plt.xscale("log")
    plt.title("Total words by reference token types")
    plt.xlabel("Total words")
    ticks, _ = plt.yticks()
    plt.yticks(ticks, ["" for _ in ticks])
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Title
    participant_str = (
        f"{participant}" if participant is not None else "All Participants"
    )
    # name corpus if only one corpus is present in df
    corpus_str = (
        df.index.get_level_values("Corpus").unique()[0]
        if len(df.index.get_level_values("Corpus").unique()) == 1
        else None
    )

    (
        plt.suptitle(f"ASR Performance - {participant_str} - {corpus_str}")
        if corpus_str
        else plt.suptitle(f"ASR Performance - {participant_str}")
    )
    plt.tight_layout()

    return df, plt, df_computed_fields


def combined_asr_performance_with_speaker_view(
    benchmarks: List[Benchmark],
    include_token_types=VERBATIM_TOKEN_TYPES,
):
    """
    Combines the ASR performance of multiple benchmarks as a per session average across all benchmarks.
    This function calculates the average Word Error Rate (WER) and Character Error Rate (CER) across all benchmarks, every session is weighted equally.

    Returns a dataframe of all results.
    Dataframe structure:
    - Index: Corpus, Session, Token Type, Measure (CER or WER)
    - Columns: Equal, Substitution, Deletion, Insertion, Total, Error Rate

    Return dataframe of the computed fields (average and sum) for each corpus and all corpora combined.

    Also returns boxplots of the CER and WER by reference token types.

    Args:
        benchmarks (List[Benchmark]): A list of benchmarks.
        participant (str or Participant or list of str or list of Participant, optional): The speaker code or Participant object for which to calculate the ASR performance. Defaults to using all speakers.
        include_token_types (list of TokenType, optional): The token types to include in the ASR performance calculation. Defaults to VERBATIM_TOKEN_TYPES from types.py.

    Returns:
        df: A dataframe containing the CER and WER by reference token types for each session.
        plt: A scatter plot of the CER by reference token types.
        df_computed_fields: A dataframe containing the computed fields (average and sum) for each corpus and all corpora combined.
    """

    ######### DATAFRAME #########
    participant = None
    df = None
    for par in benchmarks[0].reference_session.participants:
        df_list = []
        for benchmark in benchmarks:
            df2 = benchmark.calculate_asr_performance(
                speaker=par, included_token_types=include_token_types
            )

            # add corpus and session name to the dataframe index
            df2.index = pd.MultiIndex.from_tuples(
                [
                    (benchmark.corpus, benchmark.prediction_session.name, type, measure)
                    for type, measure in df2.index
                ]
            )

            df_list.append(df2)
        df2 = pd.concat(df_list)
        # add participant name to the dataframe index
        df2.index = pd.MultiIndex.from_tuples(
            [(par.code, *index) for index in df2.index]
        )
        if df is None:
            df = df2
        else:
            df = pd.concat([df, df2])

    # name the index levels
    df.index.names = ["Participant", "Corpus", "Session", "Token Type", "Measure"]

    ######### PLOT #########

    plt.figure(figsize=(12, 12))

    cer_series = df.loc[
        (slice(None), slice(None), slice(None), (slice(None)), "CER"), :
    ]
    wer_series = df.loc[
        (slice(None), slice(None), slice(None), (slice(None)), "WER"), :
    ]

    # seperate multiindex into columns
    cer_series.reset_index(inplace=True)
    wer_series.reset_index(inplace=True)

    cer_order = (
        cer_series.groupby("Token Type")["Error Rate"].median().sort_values().index
    )
    wer_order = (
        wer_series.groupby("Token Type")["Error Rate"].median().sort_values().index
    )

    # Plot 1 - CER by reference token types
    plt.subplot(2, 2, 1)
    sns.boxplot(
        data=cer_series,
        x="Error Rate",
        y="Token Type",
        hue="Participant",
        orient="h",
        order=cer_order,
    )
    plt.title("CER by reference token types")
    plt.xlabel("CER")
    plt.ylabel("Reference token types")
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Plot 2 - Total characters by reference token types
    plt.subplot(2, 2, 2)

    sns.boxplot(
        data=cer_series,
        x="Total",
        y="Token Type",
        hue="Participant",
        orient="h",
        order=cer_order,
    )
    plt.xscale("log")
    plt.title("Total characters by reference token types")
    plt.xlabel("Total characters")
    ticks, _ = plt.yticks()
    plt.yticks(ticks, ["" for _ in ticks])
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Plot 3 - WER by reference token types
    plt.subplot(2, 2, 3)
    sns.boxplot(
        data=wer_series,
        x="Error Rate",
        y="Token Type",
        hue="Participant",
        orient="h",
        order=wer_order,
    )
    plt.title("WER by reference token types")
    plt.xlabel("WER")
    plt.ylabel("Reference token types")
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Plot 4 - Total words by reference token types
    plt.subplot(2, 2, 4)
    sns.boxplot(
        data=wer_series,
        x="Total",
        y="Token Type",
        hue="Participant",
        orient="h",
        order=wer_order,
    )
    plt.xscale("log")
    plt.title("Total words by reference token types")
    plt.xlabel("Total words")
    ticks, _ = plt.yticks()
    plt.yticks(ticks, ["" for _ in ticks])
    plt.grid(True, which="both", linestyle="--", linewidth=0.25)
    plt.tight_layout()

    # Title
    participant_str = (
        f"{participant}" if participant is not None else "All Participants"
    )
    # name corpus if only one corpus is present in df
    corpus_str = (
        df.index.get_level_values("Corpus").unique()[0]
        if len(df.index.get_level_values("Corpus").unique()) == 1
        else None
    )

    (
        plt.suptitle(f"ASR Performance - {participant_str} - {corpus_str}")
        if corpus_str
        else plt.suptitle(f"ASR Performance - {participant_str}")
    )
    plt.tight_layout()

    return df, plt, None


class TokenFilter:
    """
    Class that filters utterances based on the given token types.
    If the utterance contains any token that matches the given token types, it will be returned, otherwise it will be filtered out.

    Args:
        token_types (list of TokenType, optional): The token types to filter. Defaults to an empty list.
        word_types (list of WordType, optional): The word types to filter. Defaults to an empty list.
        word_error_types (list of WordErrorType, optional): The word error types to filter. Defaults to an empty list.
        marker_types (list of MarkerType, optional): The marker types to filter. Defaults to an empty list.
        postcode_types (list of PostCodeType, optional): The postcode types to filter. Defaults to an empty list.
        pause_types (list of PauseType, optional): The pause types to filter. Defaults to an empty list.
        terminator_types (list of TerminatorType, optional): The terminator types to filter. Defaults to an empty list.
        inclusive (bool, optional): If True, the filter will include utterances that match the given types. If False, it will exclude them. Defaults to True.
    """

    def __init__(
        self,
        token_types: List[TokenType] = [],
        word_types: List[WordType] = [],
        word_error_types: List[WordErrorType] = [],
        marker_types: List[MarkerType] = [],
        postcode_types: List[PostCodeType] = [],
        pause_types: List[PauseType] = [],
        terminator_types: List[TerminatorType] = [],
        inclusive: bool = True,
    ):
        self.token_types = token_types
        self.word_types = word_types
        self.word_error_types = word_error_types
        self.marker_types = marker_types
        self.postcode_types = postcode_types
        self.pause_types = pause_types
        self.terminator_types = terminator_types
        self.inclusive = inclusive

    def __call__(self, utterance: Utterance) -> bool:
        match = False
        for token in utterance:
            if (
                token.token_type in self.token_types
                or (
                    token.token_type == TokenType.WORD
                    and token.word_type in self.word_types
                )
                or (
                    token.token_type == TokenType.WORD
                    and any(
                        error_type in token.word_error_type
                        for error_type in self.word_error_types
                    )
                )
                or (
                    token.token_type == TokenType.POST_CODE
                    and any(
                        postcode in utterance.post_codes
                        for postcode in self.postcode_types
                    )
                )
                or (
                    token.token_type == TokenType.PAUSE
                    and token.pause_type in self.pause_types
                )
                or (
                    token.token_type == TokenType.TERMINATOR
                    and token.terminator in self.terminator_types
                )
            ):
                match = True
                break
        for marker in utterance.markers:
            if marker.marker_type in self.marker_types:
                match = True
                break
        return match if self.inclusive else not match
