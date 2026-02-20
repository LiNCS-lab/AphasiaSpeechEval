from typing import List
import os
import copy
import re

import iso639

from .types import (
    TokenType,
    WordType,
    PauseType,
    TerminatorType,
    SateliteMarkerType,
    PostCodeType,
    MarkerType,
    PartOfSpeech,
    AffixType,
    WordErrorType,
    VERBATIM_TOKEN_TYPES,
)


class Token:
    """
    Represents a token in a session.
    Args:
        text (str): The text of the token.
        token_type (TokenType): The type of the token.
        timemarks (Optional[Tuple[float, float]]): The timemarks associated with the token.

    Attributes:
        syllable_pauses (int): The number of pause between syllables.
        markers (List[MarkerType]): The tags to track repetitions, retracings, reformulations, etc. by token.
        pos (PartOfSpeech | None): The POS tags from the POS tagger or MOR tier.
        affix (AffixType | None): The affix type.
    """

    def __init__(
        self, text: str, token_type: TokenType, timemarks: tuple[int, int] = None
    ):

        # Pause Between Syllables ex: rhi^noceros
        self.syllable_pauses: int = 0
        # Tags to track repetitions, retracings, reformulations, etc. by token
        self.markers: List[MarkerType] = []
        # POS tags from the POS tagger or MOR tier
        self.pos: PartOfSpeech | str | None = None
        # Copy of pos tag to allow modification of self.pos if user needs to store result of tagging tool in self.pos.
        self.original_pos: PartOfSpeech | None = self.pos
        self.affix: AffixType | None = None
        # Parent Utterance
        self.utterance: Utterance = None
        # For IPA_TRANSCRIPTION word_type
        self.target_word: str = None
        # If Word error code is present
        # p:n, p:w, p:m, p:n, s:r, s:ur, s:uk, etc.
        self.word_error_code: List[str] = []
        # Semantic, Phonological, Neologism, Morphological, Dysfluency
        self.word_error_type: List[WordErrorType] = []

        # Set attributes
        self.text: str = text
        self.token_type: TokenType = token_type
        self.timemarks: tuple[int, int] = timemarks

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        # Detect pause between syllables
        if "^" in text:
            self.syllable_pauses = text.count("^")
        self._text = text.replace("^", "")

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(str(self))

    def get_chat_format(self, timemarks=False):
        txt = str(self)
        if self.token_type in [
            TokenType.FILLER,
            TokenType.PHONOLOGICAL_FRAGMENT,
            TokenType.NON_WORD,
        ]:
            txt = self.token_type.value + txt
        if self.target_word:
            txt = txt + " [:" + self.target_word + "]"
        if self.word_error_type:
            txt = txt + " " + " ".join([str(w) for w in self.word_error_type])
        if timemarks:
            if self.timemarks:
                txt += f" \x15{int(self.timemarks[0])}_{int(self.timemarks[1])}\x15"
        return txt


class Word(Token):
    """
    Represents a word token.
    Args:
        text (str): The text of the word.
        word_type (WordType): The type of the word.
        timemarks (Optional[Tuple[float, float]]): The timemarks associated with the word.
    Attributes:
        target_word (str): The target word for IPA_TRANSCRIPTION word_type.
        word_error_code (List[str]): The error code for the word.
        word_error_type (List[WordErrorType]): The type of error for the word.
    """

    def __init__(
        self, text: str, word_type: WordType, timemarks: tuple[int, int] = None
    ):
        super().__init__(text, TokenType.WORD, timemarks)
        self.word_type = word_type

    def get_chat_format(self, timemarks=False):
        txt = self.text
        if not self.word_type == WordType.NORMAL:
            txt += self.word_type.value
        if self.target_word:
            txt = txt + " [:" + self.target_word + "]"
        if self.word_error_type:
            txt = txt + " " + " ".join([str(w) for w in self.word_error_type])
        if timemarks:
            if self.timemarks:
                txt += f" \x15{int(self.timemarks[0])}_{int(self.timemarks[1])}\x15"
        return txt


class Pause(Token):
    """
    Represents a pause token.
    Args:
        text (str): The text of the pause.
        pause_type (PauseType): The type of the pause.
        duration (Optional[float]): The duration of the pause.
        timemarks (Optional[Tuple[float, float]]): The timemarks associated with the pause.
    """

    def __init__(
        self,
        text: str,
        pause_type: PauseType,
        duration=None,
        timemarks: tuple[int, int] = None,
    ):
        super().__init__(text, TokenType.PAUSE, timemarks)
        self.pause_type = pause_type
        if pause_type == PauseType.DURATION:
            if duration is None:
                raise ValueError("Duration must be provided for PauseType.DURATION")
            self.duration = duration

    def __str__(self):
        if self.pause_type == PauseType.DURATION:
            # convert duration to minutes and seconds
            minutes = int(self.duration / 60)
            seconds = self.duration - minutes * 60
            # if seconds is round number, add . after it
            if seconds % 1 == 0:
                seconds = int(seconds)
                seconds = str(seconds) + "."
            if minutes == 0:
                return f"({seconds})"
            return f"({minutes}:{seconds})"
        return str(self.pause_type)


class Terminator(Token):
    """
    Represents a terminator token.
    Args:
        text (str): The text of the terminator.
        terminator (TerminatorType): The type of the terminator.
        timemarks (Optional[Tuple[float, float]]): The timemarks associated with the terminator.
    """

    def __init__(
        self, text: str, terminator: TerminatorType, timemarks: tuple[int, int] = None
    ):
        super().__init__(text, TokenType.TERMINATOR, timemarks)
        self.terminator = terminator

    def __str__(self):
        return str(self.terminator)


class SateliteMarker(Token):
    """
    Represents a satelite marker token.
    Args:
        text (str): The text of the satelite marker.
        satelite_marker (SateliteMarkerType): The type of the satelite marker.
        timemarks (Optional[Tuple[float, float]]): The timemarks associated with the satelite marker.
    """

    def __init__(
        self,
        text: str,
        satelite_marker: SateliteMarkerType,
        timemarks: tuple[int, int] = None,
    ):
        super().__init__(text, TokenType.SAT_MARKER, timemarks)
        self.satelite_marker = satelite_marker

    def __str__(self):
        return str(self.satelite_marker)


class PostCode(Token):
    """
    Represents a post code token.
    Args:
        text (str): The text of the post code.
        post_code (PostCodeType): The type of the post code.
        timemarks (Optional[Tuple[float, float]]): The timemarks associated with the post code.
    """

    def __init__(
        self, text: str, post_code: PostCodeType, timemarks: tuple[int, int] = None
    ):
        super().__init__(text, TokenType.POST_CODE, timemarks)
        self.post_code = post_code

    def __str__(self):
        return str(self.post_code)


class Marker:
    """
    A Marker object represents a marked span of tokens in an utterance. This can be a repetition, retracing, reformulation, etc.
    Args:
        marker_type (MarkerType): The type of the marker.
        span (tuple[int, int]): The span of the marker in the utterance, measured in tokens.
    """

    def __init__(self, marker_type: MarkerType, span: tuple[int, int]):
        self.marker_type = marker_type
        self.span = span

    def __str__(self):
        return f"{self.marker_type} {self.span}"

    def __repr__(self):
        return f"{self.marker_type} {self.span}"


class Participant:
    """
    Represents a participant in a session.
    Args:
        code (str): The code of the participant.
        name (str): The name of the participant.
        role (str): The role of the participant.
        language (str): The language of the participant.
        corpus (str): The corpus of the participant.
        age (str | int | float | tuple[int] | tuple[int, int] | tuple[int, int, int]): The age of the participant. Can be a string, integer, float, or a tuple of integers representing year and month and day.
        sex (str): The sex of the participant.
        group (str): The group of the participant.
        eth (str): The ethnicity of the participant.
        ses (str): The socioeconomic status of the participant.
        education (str): The education of the participant.
        custom (str): Custom information about the participant.
    Attributes:
        id_header (str): The ID header of the participant.
    """

    def __init__(
        self,
        code: str,
        name: str,
        role: str,
        language: str = "",
        corpus: str = "",
        age: (
            str | int | float | tuple[int] | tuple[int, int] | tuple[int, int, int]
        ) = "",
        sex: str = "",
        group: str = "",
        ethnicity: str = "",
        ses: str = "",
        education: str = "",
        custom: str = "",
    ):

        if age:
            year = None
            month = None
            day = None

            if isinstance(age, float):
                year = int(age)
                month = int((age - year) * 12)
            if isinstance(age, int):
                year = age
            if isinstance(age, str):
                # regex to match age in the format of 1;2.3 or 1;2. or 1; or 1
                regex = re.compile(r"(\d+);?(\d+)?\.?(\d+)?")
                match = regex.match(age)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2)) if match.group(2) else None
                    day = int(match.group(3)) if match.group(3) else None
            if isinstance(age, tuple):
                if len(age) == 1:
                    year = age[0]
                if len(age) == 2:
                    year = age[0]
                    month = age[1]
                if len(age) == 3:
                    year = age[0]
                    month = age[1]
                    day = age[2]

            # Validate age values
            if year < 0:
                raise ValueError("Age year cannot be negative")
            if month:
                if 0 < month > 11:
                    raise ValueError("Age month must be between 0 and 11")
            if day:
                if 0 < day > 31:
                    raise ValueError("Age day must be between 0 and 31")

            # Convert age to string
            if year:
                age = f"{year};"
            if month is not None:
                if month < 10:
                    age += f"0{month}."
                else:
                    age += f"{month}."
            if day is not None:
                if day < 10:
                    age += f"0{day}"
                else:
                    age += f"{day}"

        self.name = name
        self.language = language
        self.corpus = corpus
        self.code = code.upper()
        self.age = age
        self.sex = sex
        self.group = group
        self.eth = ethnicity
        self.ses = ses  # Socioeconomic status
        self.role = role
        self.education = education
        self.custom = custom

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return self.code

    @property
    def id_header(self):
        if self.eth and self.ses:
            return f"@ID:	{self.language}|{self.corpus}|{self.code}|{self.age}|{self.sex}|{self.group}|{self.eth},{self.ses}|{self.role}|{self.education}|{self.custom}|"
        else:
            return f"@ID:	{self.language}|{self.corpus}|{self.code}|{self.age}|{self.sex}|{self.group}|{self.eth}{self.ses}|{self.role}|{self.education}|{self.custom}|"


class Utterance:
    """
    Represents an utterance in a session.
    Args:
        tokens (List[Token]): The list of tokens in the utterance.
        speaker (Participant): The speaker of the utterance.
        timemarks (tuple[int, int], optional): The start and end timemarks of the utterance. Defaults to None.
        markers (List[Marker], optional): The list of markers in the utterance. Defaults to None.
        tiers (dict, optional): The tiers associated with the utterance. Defaults to None.
    Methods:
        get_token_spans(): Returns the token spans in the utterance.
        verbatim_string(): Returns a string representation of the verbatim tokens in the utterance.
        verbatim_tokens(): Returns the verbatim tokens in the utterance.
    """

    def __init__(
        self,
        tokens: List[Token],
        speaker: Participant,
        timemarks: tuple[int, int] = None,
        markers: List[Marker] = None,
        tiers: dict = None,
    ):
        self.tokens: List[Token] = tokens
        for token in self.tokens:
            token.utterance = self
            token.markers = []

        self.post_codes: set[PostCodeType] = set(
            [
                token.post_code
                for token in self.tokens
                if token.token_type == TokenType.POST_CODE
            ]
        )

        self.speaker: Participant = speaker
        self.timemarks: tuple[int, int] = timemarks
        self.markers: List[Marker] = markers if markers else []
        self.tiers = tiers

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    # make subscriptable
    def __getitem__(self, index):
        return str(self)[index]

    def get_token_spans(self):
        """
        Returns the token spans in the utterance.
        Example: "This is a test" -> [(0, 4), (5, 7), (8, 9), (10, 14)]
        """
        token_spans = []
        for token in self.tokens:
            if token_spans:
                token_spans.append(
                    (token_spans[-1][1] + 1, token_spans[-1][1] + 1 + len(token))
                )
            else:
                token_spans.append((0, len(token)))
        return token_spans

    def verbatim_string(self, types: List[Token] = VERBATIM_TOKEN_TYPES):
        return " ".join(
            [str(token) for token in self.tokens if token.token_type in types]
        )

    def verbatim_tokens(self, types: List[Token] = VERBATIM_TOKEN_TYPES):
        return [token for token in self.tokens if token.token_type in types]

    @property
    def markers(self):
        from .. import utils

        if not self._markers:
            markers = utils.repetitions(self.tokens)
            self._markers = markers

            for marker in markers:
                for i in range(marker.span[0], marker.span[1] + 1):
                    self.tokens[i].markers.append(marker.marker_type)

        return self._markers

    @markers.setter
    def markers(self, markers: List[Marker]):
        self._markers = markers

        # Apply marker tag to involved tokens
        if markers:
            for marker in markers:
                for i in range(marker.span[0], marker.span[1] + 1):
                    self.tokens[i].markers.append(marker.marker_type)


class Session:
    """
    Represents a session in a chat file.
    Args:
        name (str): The name of the session.
        utterances (List[Utterance]): The list of utterances in the session.
        languages (set[str], optional): The languages of the session.
        participants (set[Participant], optional): The participants in the session.
        header_options (str, optional): The header options of the session. Defaults to None.
        header_media (str, optional): The header media of the session. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        utterances: List[Utterance],
        participants: set[Participant],
        languages: set[str],
        header_comment: str = None,
        header_media: str = None,
        header_options: str = None,
    ):
        self.name = name
        self.utterances = utterances
        self._languages: set[iso639.Language] = set()
        self.languages = languages
        self.participants: set[Participant] = participants
        self.header_comment = header_comment
        self.header_media = header_media
        self.header_options = header_options

    def __iter__(self):
        return iter(self.utterances)

    def __getitem__(self, index):
        return self.utterances[index]

    def __str__(self):
        str_txt = ""
        for line in self.to_chat():
            str_txt += line + "\n"
        return str_txt

    def __len__(self):
        return len(self.utterances)

    @property
    def languages(self):
        return self._languages

    @languages.setter
    def languages(self, languages_set: set[str]):
        """
        Set the languages of the session respecting the ISO 639-3 standard.
        """
        iso_langs = set()
        for lang in languages_set:
            iso_langs.add(iso639.Language.match(lang))
        self._languages = {lang.part3 for lang in iso_langs} if iso_langs else set()

    def get_participant_utts(
        self,
        participant: Participant | str | List[str] | List[Participant] | None = None,
    ):
        """
        Returns the utterances of the participant.
        If participant is not provided, returns all utterances in the session.
        May also provide a list of participants.
        """

        # Convert str, Participant or List[Participant] to List[str]
        if participant:

            if isinstance(participant, str):
                str_list = [participant]
            elif isinstance(participant, Participant):
                str_list = [participant.code]
            elif isinstance(participant, List):
                if participant == []:
                    str_list = []
                elif isinstance(participant[0], Participant):
                    str_list = [p.code for p in participant]
                elif isinstance(participant[0], str):
                    str_list = participant

            return [utt for utt in self.utterances if utt.speaker.code in str_list]

        # If participant is not provided, return all utterances
        return self.utterances

    def duration(self, participant: Participant | str | None = None):
        """
        Returns the duration of the session in seconds.
        If participant is provided, returns the duration of the participant's speech.
        """
        if participant:
            times = []
            utterances = self.get_participant_utts(participant)
            for utt in utterances:
                if utt.timemarks:
                    times.append(utt.timemarks[1] - utt.timemarks[0])
            return sum(times) / 1000

        start = self.utterances[0].timemarks[0]
        end = self.utterances[-1].timemarks[1]
        return (end - start) / 1000

    def words_per_minute(self, participant: Participant | str | None = None):
        """
        Returns the words spoken per minute in the session.
        If participant is provided, returns the words spoken per minute by the participant.
        """
        if participant:
            words = self.count_words(participant)
            duration = self.duration(participant)
        else:
            words = self.count_words()
            duration = self.duration()
        return words / (duration / 60)

    def mean_length_of_utterance(self, participant: Participant | str | None = None):
        """
        Returns the mean length of utterances in the session in morphemes. (Tokens that have a part of speech tag)
        If participant is provided, returns the mean length of utterance by the participant.
        Revisions, fillers, and unintelligible utterances excluded.
        """
        utterances = self.get_participant_utts(participant)

        count = sum([len([token for token in utt if token.pos]) for utt in utterances])
        return count / len(utterances)

    def count_utterance_terminators(self, participant: Participant | str | None = None):
        """
        Returns the count of utterance terminators in the session. Ex : +..., +..?, +!? etc.
        If participant is provided, returns the count of utterance terminators by the participant.
        """
        utterances = self.get_participant_utts(participant)

        count = sum(
            [
                len(
                    [token for token in utt if token.token_type == TokenType.TERMINATOR]
                )
                for utt in utterances
            ]
        )
        return count

    def count_abandoned_utterances(self, participant: Participant | str | None = None):
        """
        Returns the count of abandoned utterances in the session.
        ie : +..., +..? (trailing off, trailing off to a question)
        If participant is provided, returns the count of abandoned utterances by the participant.
        """
        utterances = self.get_participant_utts(participant)
        # Utterance terminators +… and +..? for abandoned utterances
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.TERMINATOR
                    and token.terminator
                    in [
                        TerminatorType.TRAILING_OFF,
                        TerminatorType.TRAIL_OFF_TO_A_QUESTION,
                    ]
                ]
            )
        return count

    def count_empty_speech(self, participant: Participant | str | None = None):
        """
        Returns the count of empty speech word tokens in the session.
        Counts postcode [+es] for empty speech
        If participant is provided, returns the count of empty speech by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            for token in utt:
                if (
                    token.token_type == TokenType.POST_CODE
                    and token.post_code == PostCodeType.EMPTY_SPEECH
                ):
                    count += len(
                        [
                            token
                            for token in utt.tokens
                            if token.token_type == TokenType.WORD
                        ]
                    )
        return count

    def count_unfilled_pauses(self, participant: Participant | str | None = None):
        """
        Returns the count of unfilled pauses in the session.
        If participant is provided, returns the count of unfilled pauses by the participant.
        """
        utterances = self.get_participant_utts(participant)
        # Sum of pause tokens for unfilled pauses
        count = 0
        for utt in utterances:
            count += len(
                [token for token in utt if token.token_type == TokenType.PAUSE]
            )
            count += sum([token.syllable_pauses for token in utt])
        return count

    def count_syllable_pauses(self, participant: Participant | str | None = None):
        """
        Returns the count of syllable pauses in the session.
        ie: rhi^noceros
        If participant is provided, returns the count of syllable pauses by the participant.
        """
        utterances = self.get_participant_utts(participant)
        # Sum of syllable pause tokens
        count = 0
        for utt in utterances:
            count += sum([token.syllable_pauses for token in utt])
        return count

    def count_filled_pauses(self, participant: Participant | str | None = None):
        """
        Returns the count of filled pauses in the session.
        ie: filler words like "um", "uh", "er", etc.
        If participant is provided, returns the count of filled pauses by the participant.
        """
        utterances = self.get_participant_utts(participant)
        # Fillers count
        count = 0
        for utt in utterances:
            count += len(
                [token for token in utt if token.token_type == TokenType.FILLER]
            )
        return count

    def count_words(self, participant: Participant | str | None = None):
        """
        Returns the count of words in the session.
        ie: word tokens only
        If participant is provided, returns the count of words by the participant
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len([token for token in utt if token.token_type == TokenType.WORD])
        return count

    def count_semantic_errors(self, participant: Participant | str | None = None):
        """
        Returns the count of semantic errors in the session.
        Counts word error codes [*s:k] and [*s:uk].
        If participant is provided, returns the count of semantic errors by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and WordErrorType.SEMANTIC in token.word_error_type
                ]
            )
        return count

    def count_generic_errors(self, participant: Participant | str | None = None):
        """
        Returns the count of generic errors in the session.
        Counts word error codes [*].
        If participant is provided, returns the count by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and WordErrorType.GENERIC in token.word_error_type
                ]
            )
        return count

    def count_morphological_errors(self, participant: Participant | str | None = None):
        """
        Returns the count of morphological errors in the session.
        Counts word error codes [*m].
        If participant is provided, returns the count by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and WordErrorType.MORPHOLOGICAL in token.word_error_type
                ]
            )
        return count

    def count_dysfluency_errors(self, participant: Participant | str | None = None):
        """
        Returns the count of dysfluency errors in the session.
        Counts word error codes [*d].
        If participant is provided, returns the count by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and WordErrorType.DYSFLUENCY in token.word_error_type
                ]
            )
        return count

    def count_phonological_errors(self, participant: Participant | str | None = None):
        """
        Returns the count of phonological errors in the session.
        Counts word error codes [*p:k] and [*p:uk].
        If participant is provided, returns the count of phonological errors by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and WordErrorType.PHONOLOGICAL in token.word_error_type
                ]
            )
        return count

    def count_neologisms(self, participant: Participant | str | None = None):
        """
        Returns the count of neologisms in the session.
        Counts word error codes [*n:k] and [*n:uk].
        Counts words with word type NEOLOGISM.
        If participant is provided, returns the count of neologisms by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and (
                        WordErrorType.NEOLOGISM in token.word_error_type
                        or token.word_type == WordType.NEOLOGISM
                    )
                ]
            )
        return count

    def count_jargon(self, participant: Participant | str | None = None):
        """
        Returns the count of jargon word tokens in the session.
        Counts postcodes [+jar].
        If participant is provided, returns the count of jargon by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            for token in utt:
                if (
                    token.token_type == TokenType.POST_CODE
                    and token.post_code == PostCodeType.JARGON
                ):
                    count += len(
                        [
                            token
                            for token in utt.tokens
                            if token.token_type == TokenType.WORD
                        ]
                    )
        return count

    def count_bound_morphemes(self, participant: Participant | str | None = None):
        """
        Returns the count of bound morphemes in the session.
        Counts affixes in the %mor tier : PRESP, PAST, PASTP, 3S, PL, 1S.
        If participant is provided, returns the count of bound morphemes by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD and token.affix
                ]
            )
        return count

    def count_free_morphemes(self, participant: Participant | str | None = None):
        """
        Returns the count of free morphemes in the session.
        If participant is provided, returns the count of free morphemes by the participant.
        Free morphemes include nouns, verbs, auxiliaries, prepositions, adjectives, adverbs, conjunctions, determiners/articles, pronouns.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and token.pos
                    in [
                        PartOfSpeech.NOUN,
                        PartOfSpeech.VERB,
                        PartOfSpeech.VERB_AUXILIARY,
                        PartOfSpeech.PREPOSITION,
                        PartOfSpeech.ADJECTIVE,
                        PartOfSpeech.ADVERB,
                        PartOfSpeech.CONJUNCTION,
                        PartOfSpeech.DETEMINER,
                        PartOfSpeech.PRONOUN,
                    ]
                ]
            )
        return count

    def count_closed_class_words(self, participant: Participant | str | None = None):
        """
        Returns the count of closed class words in the session.
        If participant is provided, returns the count of closed class words by the participant.
        Closed class words include auxiliaries, prepositions, conjunctions, determiners/articles, pronouns.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and token.pos
                    in [
                        PartOfSpeech.VERB_AUXILIARY,
                        PartOfSpeech.PREPOSITION,
                        PartOfSpeech.CONJUNCTION,
                        PartOfSpeech.DETEMINER,
                        PartOfSpeech.PRONOUN,
                    ]
                ]
            )
        return count

    def count_open_class_words(self, participant: Participant | str | None = None):
        """
        Returns the count of open class words in the session.
        If participant is provided, returns the count of open class words by the participant.
        Open class words include nouns, verbs, adjectives, adverbs.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and token.pos
                    in [
                        PartOfSpeech.NOUN,
                        PartOfSpeech.VERB,
                        PartOfSpeech.ADJECTIVE,
                        PartOfSpeech.ADVERB,
                    ]
                ]
            )
        return count

    def count_pronouns(self, participant: Participant | str | None = None):
        """
        Counts the number of pronouns in the utterances of a participant.
        Args:
            participant (Participant | str | None, optional): The participant whose utterances will be considered.
                Defaults to None, which means all participants will be considered.
        Returns:
            int: The total count of pronouns in the utterances.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and token.pos == PartOfSpeech.PRONOUN
                ]
            )
        return count

    def count_nouns(self, participant: Participant | str | None = None):
        """
        Counts the number of nouns in the utterances of a participant.
        Args:
            participant (Participant | str | None): The participant whose utterances will be counted.
                If None, all participants' utterances will be counted.
        Returns:
            int: The total count of nouns in the utterances.
        """

        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and token.pos == PartOfSpeech.NOUN
                ]
            )
        return count

    def count_agrammatic_utterances(self, participant: Participant | str | None = None):
        """
        Returns the count of agrammatic utterances in the session.
        If participant is provided, returns the count of agrammatic utterances by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.POST_CODE
                    and token.post_code == PostCodeType.GRAMMATICAL
                ]
            )
        return count

    def count_retraced_sequences(self, participant: Participant | str | None = None):
        """
        Returns the count of retraced sequences in the session.
        If participant is provided, returns the count of retraced sequences by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [
                    marker
                    for marker in utt.markers
                    if marker.marker_type
                    in [MarkerType.REPETITION, MarkerType.RETRACING]
                ]
            )
        return count

    def count_false_starts(self, participant: Participant | str | None = None):
        """
        Returns the count of false starts in the session.
        If participant is provided, returns the count of false starts by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:

            count += len(
                [
                    marker
                    for marker in utt.markers
                    if marker.marker_type in [MarkerType.FALSE_START_WITHOUT_RETRACING]
                ]
            )
        return count

    def count_unintelligible_sequences(
        self, participant: Participant | str | None = None
    ):
        """
        Returns the count of unintelligible sequences in the session.
        If participant is provided, returns the count of unintelligible sequences by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:
            count += len(
                [token for token in utt if token.token_type == TokenType.UNINTELLIGIBLE]
            )
        return count

    def count_overlaps(self, participant: Participant | str | None = None):
        """
        Returns the count of overlapped words in the session.
        If participant is provided, returns the count of overlaps by the participant.
        """
        utterances = self.get_participant_utts(participant)
        count = 0
        for utt in utterances:

            tokens = set()
            tokens.update(
                [
                    token
                    for token in utt
                    if token.token_type == TokenType.WORD
                    and any(
                        [
                            markertype
                            in [
                                MarkerType.OVERLAP_FOLLOWING,
                                MarkerType.OVERLAP_PRECEDING,
                            ]
                            for markertype in token.markers
                        ]
                    )
                ]
            )
            for token in utt:
                if token.token_type == TokenType.LAZY_OVERLAP:
                    tokens.update([t for t in utt if t.token_type == TokenType.WORD])
                    break
            count += len(tokens)
        return count

    @classmethod
    def from_chat(cls, file_path: str, participants: list = None):
        """
        Create a Session object from a chat file.
        Args:
            file_path (str): The path to the chat file.
        Raises:
            ValueError: If the file format is not .cha.
        Returns:
            Session: The Session object created from the chat file.
        """
        if not file_path.endswith(".cha"):
            raise ValueError("Invalid file format. File must be a .cha file")
        parse = cls._parse_chat(file_path, participants=participants)
        return cls(*parse)

    @classmethod
    def from_chat_zip(cls, file_path: str, session_id: str, participants: list = None):
        """
        Create a session object from a chat zip file.
        Args:
            file_path (str): The path to the chat zip file.
            session_id (str): The session ID.
        Returns:
            Session: A session object created from the chat zip file.
        Raises:
            ValueError: If the file format is not a .zip file.
        """

        if not file_path.endswith(".zip"):
            raise ValueError("Invalid file format. File must be a .zip file")
        parse = cls._parse_chat(file_path, session_id, participants=participants)
        return cls(*parse)

    @staticmethod
    def _parse_chat(file_path: str, session_id: str = None, participants: list = None):

        from ..modules.chat_file_parser import process_chat
        from ..utils.chat_utils import get_sessions

        if not session_id:
            session_ids = get_sessions(file_path)
            if len(session_ids) > 1:
                raise ValueError(
                    "Multiple sessions found. Please provide a unique session ID"
                )
            session_id = session_ids[0]
        parse = process_chat(file_path, session_id, participants_list=participants)
        return (session_id, *parse)

    def to_chat(self) -> List[str]:
        """
        Converts the session object into a chat format.
        Returns:
            List[str]: A list of strings representing the chat format, line by line.
        """

        def get_header():
            # @participants and @ID lines
            participant_lines = []
            id_lines = []
            for p in self.participants:
                participant_lines.append(p.code + " " + p.name + " " + p.role)
                id_lines.append(p.id_header)

            header = []
            header.append("@UTF8")
            header.append("@Begin")
            header.append("@Languages:\t" + ", ".join(self.languages))
            header.append("@Participants:\t" + ", ".join(participant_lines))
            if self.header_options:
                header.append("@Options:\t" + self.header_options)
            for id_line in id_lines:
                header.append(id_line)
            if self.header_media:
                header.append(f"@Media:\t{self.header_media}")
            if self.header_comment:
                header.append(f"@Comment:\t{self.header_comment}")

            return header

        def insert_markers(utterance, text):

            utterance.markers.sort(key=lambda x: x.span[0], reverse=True)

            for marker in utterance.markers:
                if not marker.span[0] == marker.span[1]:
                    text[marker.span[0]] = "<" + text[marker.span[0]]
                    if " \x15" in text[marker.span[1]]:
                        text[marker.span[1]] = text[marker.span[1]].replace(
                            " \x15", "> \x15"
                        )
                    else:
                        text[marker.span[1]] = text[marker.span[1]] + ">"

                if " \x15" in text[marker.span[1]]:
                    text[marker.span[1]] = text[marker.span[1]].replace(
                        " \x15", f" {marker.marker_type} \x15"
                    )
                else:
                    text[marker.span[1]] = (
                        text[marker.span[1]] + " " + str(marker.marker_type)
                    )
            return text

        def get_main_tier(u):

            utterance_tokens_text = [t.get_chat_format() for t in u.tokens]

            utterance_tokens_text = insert_markers(u, utterance_tokens_text)

            if u.timemarks:
                return f"*{u.speaker.code}:\t{' '.join(utterance_tokens_text)} \x15{int(u.timemarks[0])}_{int(u.timemarks[1])}\x15"
            else:
                return f"*{u.speaker.code}:\t{' '.join(utterance_tokens_text)}"

        def get_wor_tier(u):

            utterance_tokens_text = [
                t.get_chat_format(timemarks=True) for t in u.tokens
            ]

            utterance_tokens_text = insert_markers(u, utterance_tokens_text)

            return f"%wor:\t{' '.join(utterance_tokens_text)}"

        def get_mor_tier(u):

            mor_tier = []
            for token in u.tokens:
                if token.pos:
                    if token.affix:
                        mor_tier.append(
                            f"{token.pos.value}|{token.text}-{token.affix.value}"
                        )
                    else:
                        mor_tier.append(f"{token.pos.value}|{token.text}")
                elif token.token_type == TokenType.PUNCTUATION and not token.markers:
                    mor_tier.append(token.text)

            return f"%mor:\t{' '.join(mor_tier)}"

        def get_utterance_tiers(utterance):

            main_tier = get_main_tier(utterance)
            wor_tier = get_wor_tier(utterance)
            mor_tier = get_mor_tier(utterance)

            tiers = [main_tier, wor_tier, mor_tier]
            return tiers

        chat = []
        chat.extend(get_header())
        for u in self.utterances:
            u_copy = copy.deepcopy(u)
            chat.extend(get_utterance_tiers(u_copy))
        chat.append("@End")
        return chat

    def save_chat(self, file_path: str):
        """
        Save the chat session to a .cha file.
        Args:
            file_path (str): The path to the file where the chat session will be saved.
        Raises:
            ValueError: If the file format is invalid. The file must be a .cha file.
        """

        if file_path.count(".") > 0 and file_path.split(".")[-1] != "cha":
            raise ValueError("Invalid file format. File must be a .cha file")
        if not file_path.endswith(".cha"):
            file_path += ".cha"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for line in self.to_chat():
                f.write(line)
                f.write("\n")
