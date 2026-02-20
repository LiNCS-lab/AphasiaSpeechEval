import re
from typing import List
import pylangacq as pla

from ..models import (
    Participant,
    Utterance,
    Token,
    Word,
    Pause,
    PostCode,
    Marker,
    SateliteMarker,
    Terminator,
    TokenType,
    WordType,
    PauseType,
    PostCodeType,
    MarkerType,
    SateliteMarkerType,
    TerminatorType,
    AffixType,
    PartOfSpeech,
    WordErrorType,
)


def _get_participants(p_data, participants_selected=None) -> List[Participant]:
    participants = []
    for p in p_data:

        if participants_selected is not None and p not in participants_selected:
            break

        code = p
        name = p_data[p]["name"] if "name" in p_data[p] else None
        language = p_data[p]["language"]
        corpus = p_data[p]["corpus"]
        age = p_data[p]["age"]
        sex = p_data[p]["sex"]
        group = p_data[p]["group"]
        ses = p_data[p]["ses"]
        role = p_data[p]["role"]
        education = p_data[p]["education"]
        custom = p_data[p]["custom"]

        participants.append(
            Participant(
                name=name,
                language=language,
                corpus=corpus,
                code=code,
                age=age,
                sex=sex,
                group=group,
                ses=ses,
                role=role,
                education=education,
                custom=custom,
            )
        )

    code_list = [p.code for p in participants]
    if len(code_list) != len(set(code_list)):
        raise ValueError("Error: Multiple participants with the same code found")

    return participants


def process_chat(
    file_path: str, session_id: str, participants_list: list = None
) -> tuple:
    chat = pla.Reader.from_files([file_path], match=session_id, parallel=False)

    participants = _get_participants(
        chat.headers()[0]["Participants"], participants_list
    )
    languages = chat.languages()
    header_media = chat.headers()[0]["Media"] if "Media" in chat.headers()[0] else None
    header_options = (
        chat.headers()[0]["Options"] if "Options" in chat.headers()[0] else None
    )

    if participants_list is not None:
        chat_utterances = chat.utterances(participants=participants_list)
        chat_words = chat.words(participants=participants_list, by_utterances=True)
    else:
        chat_utterances = chat.utterances()
        chat_words = chat.words(by_utterances=True)

    utterances = []
    last_timemark_empty = 0

    for u_idx, u in enumerate(chat_utterances):
        speaker_code = u.participant
        for p in participants:
            if p.code == speaker_code:
                speaker = p
                break

        if u.time_marks:
            timemarks = u.time_marks
        # This is a special case where the previous utterance has no timemark
        # This shouldnt happen but when it does we set the timemark to the end
        # of the span from the end of the previous utterance to the start of the current utterance
        if last_timemark_empty:
            if timemarks:
                last_timemark_empty = 0
                for i in range(last_timemark_empty):

                    utterances[-i].timemarks = (
                        utterances[-(i + 1)].timemarks[1],
                        timemarks[0],
                    )

        if not u.time_marks:
            last_timemark_empty += 1
            timemarks = None

        # We privilege %wor tier because it contains per word timemarks. Fallback to participant tier if %wor is not available
        if "%wor" not in u.tiers:
            text = u.tiers[u.participant]
        else:
            text = u.tiers["%wor"]

        # special case: remove hyphen if it appears between a word and a timemark. Ex day-§168042 =>  day §168042 (in the case of day-to-day)
        matches = re.finditer(r"([^\s\d])\x15", text)
        if matches:
            for match in matches:
                text = text[: match.start(1)] + " " + text[match.start(1) + 1 :]

        # Quick fix for pause between syllables
        # text = re.sub(r"(?<!\+)\^", " ", text)

        # Simplify capture of postcodes, errors, and phonetic targets by removing any whitespace between the brackets.
        text = re.sub(r"(\[[:*+])\s*(\S*)\s*(\])", r"\1\2\3", text)

        # split text into tokens
        proto_tokens = text.split()

        # create token objects
        utterance_tokens = []
        utterance_markers = []
        rep_marker_start = []
        while proto_tokens:

            current_token = proto_tokens.pop(0)

            # if token is a timemark, add it to the last token
            word_timemarks = _extract_word_timemarks(current_token)
            if word_timemarks:
                if len(utterance_tokens) == 0:
                    raise ValueError(
                        "Timemark should not be the first token in an utterance"
                    )
                utterance_tokens[-1].timemarks = word_timemarks
                continue

            # if token start with <, keep token number in memory. when a token ends with >, next token will be a retracing or repetition marker
            if current_token.startswith("<"):
                rep_marker_start.append(len(utterance_tokens))
                current_token = current_token[1:]

            if current_token.endswith(">") and rep_marker_start:
                rep_marker_end = len(utterance_tokens)
                current_token = current_token[:-1]
                token = _tokenize(current_token)
                if token:
                    if isinstance(token, list):
                        utterance_tokens.extend(token)
                    else:
                        utterance_tokens.append(token)
                rep_token = proto_tokens.pop(0)

                # if token is a timemark, add it to the last token
                word_timemarks = _extract_word_timemarks(rep_token)
                if word_timemarks:
                    utterance_tokens[-1].timemarks = word_timemarks
                    rep_token = proto_tokens.pop(0)

                # print(rep_token, rep_marker_start[-1], rep_marker_end)
                marker = _extract_retracing_overlap(
                    rep_token, rep_marker_start.pop(), rep_marker_end
                )
                if marker:
                    utterance_markers.append(marker)
                continue

            # if token is a repetition or retracing marker
            last_token_position = len(utterance_tokens) - 1
            marker = _extract_retracing_overlap(
                current_token, last_token_position, last_token_position
            )
            if marker:
                utterance_markers.append(marker)
                continue

            # if token is an attribute of the previous token
            if len(utterance_tokens) < 1:
                previous_token = None
            else:
                previous_token = utterance_tokens[-1]
            if _extract_token_attributes(current_token, previous_token):
                continue

            # tokenize the current string
            token = _tokenize(current_token)
            if token:
                # if list, extend the list
                if isinstance(token, list):
                    utterance_tokens.extend(token)
                else:
                    utterance_tokens.append(token)

        # MOR
        if "%mor" in u.tiers:
            mors = u.tiers["%mor"].split()
            words = chat_words[u_idx]
            tokens_list = list(utterance_tokens)

            for mor, wor in zip(mors, words):
                for token in tokens_list:
                    if token.text == wor:
                        current_affix = None
                        if "#" in mor:
                            code = mor.split("|")[0].split(":")[0].split("#")[1]
                        else:
                            code = mor.split("|")[0].split(":")[0]

                        for affix in AffixType:
                            if affix.value in mor:
                                current_affix = affix

                        for pos in PartOfSpeech:
                            if pos.value == code:
                                token.pos = pos
                                token.affix = current_affix
                                break

                        break

        # create utterance object
        utterances.append(
            Utterance(
                utterance_tokens,
                speaker,
                timemarks,
                markers=utterance_markers,
                tiers=u.tiers,
            )
        )
    return utterances, participants, languages, header_media, header_options


def _tokenize(text):
    # Unidentifiable
    if text == "www":
        return Token(text, TokenType.UNIDENTIFIABLE)

    # Untranscribable
    if text == "yyy":
        return Token(text, TokenType.UNTRANSCRIBABLE)

    # Lazy Overlap
    if text == "+<":
        return Token(text, TokenType.LAZY_OVERLAP)

    # Unintelligible coded as xxx.
    if text == "xxx":
        return Token(text, TokenType.UNINTELLIGIBLE)

    # Satellite markers
    if text == "‡":
        return SateliteMarker(text, SateliteMarkerType.PREFIX)
    if text == "„":
        return SateliteMarker(text, SateliteMarkerType.SUFFIX)

    # Post codes
    token = _extract_post_codes(text)
    if token:
        return token

    # Pauses
    pause = re.match(r"\((\.+)\)", text)
    if pause:
        # Pause type is determined by the number of dots
        if len(pause.group(1)) == 1:
            return Pause(text, PauseType.SHORT)
        elif len(pause.group(1)) == 2:
            return Pause(text, PauseType.MEDIUM)
        elif len(pause.group(1)) == 3:
            return Pause(text, PauseType.LONG)
        else:
            raise ValueError("Pause length is invalid: " + text)

    # Coded pauses (ex: (1:30.25) (5:30.) (2.) (0.75))
    pause = re.match(r"\((\d*:?\d*\.\d*)\)", text)
    if pause:
        if ":" in pause.group(1):
            minutes, seconds = pause.group(1).split(":")
            minutes = int(minutes)
            seconds = float(seconds)
        else:
            minutes = 0
            seconds = pause.group(1)
            seconds = float(seconds)
        return Pause(text, PauseType.DURATION, duration=minutes * 60 + seconds)

    # remove parentheses
    text = text.replace("(", "").replace(")", "")

    # Punctuation
    if re.match(r"[,.?!]", text):
        return Token(text, TokenType.PUNCTUATION)

    # Phonological fragments, fillers, non-words, and gestures ex: &+Tuz, &-uh, &~jiewqioje, &=head:yes
    token = _extract_prefixes(text)
    if token:
        return token

    # Special form markers ex: @u, @q, @n, @o, @k, @l
    token = _extract_special_form_markers(text)
    if token:
        return token

    # Gestures
    gestures = _extract_gestures(text)
    if gestures:
        return gestures

    # Special utterance terminators
    token = _extract_special_utterance_terminators(text)
    if token:
        return token

    # Normal words
    return Word(text, WordType.NORMAL)


def _extract_special_form_markers(text):
    """
    @u  IPA transcription
    @q  Meta-linguistic use
    @n  Neologism
    @o  Onomatopoeia
    @k  Multiple letters (spelling)
    @l  Letter
    """
    special_form = re.finditer(r"(\S+)(\@\w+)", text)
    if special_form:
        tolist = list(special_form)
        if len(tolist) > 1:
            raise ValueError("Multiple special form markers found in a single word")

        for match in tolist:
            marker = match.group(2)
            word = match.group(1)

            for wordtype in WordType:
                if marker == wordtype.value:
                    return Word(word, wordtype)
            raise ValueError("Invalid special form marker: " + marker)


def _extract_prefixes(text):
    """
    &+Tuz        Phonological fragment
    &-uh         Filler
    &~jiewqioje  Nonword
    &=head:yes   Gesture
    """

    def extract_markers(text, pattern, marker_type):
        markers = re.finditer(pattern, text)
        if markers:
            tolist = list(markers)
            if len(tolist) > 1:
                raise ValueError("Multiple markers found in a single word: " + text)
            for match in tolist:
                return Token(match.group(1), marker_type)

    # extract phonological fragments (ex: &+Tuz)
    match = extract_markers(text, r"&\+(\w+)", TokenType.PHONOLOGICAL_FRAGMENT)
    if match:
        return match

    # extract Fillers (ex: &-uh)
    match = extract_markers(text, r"&\-(\w+)", TokenType.FILLER)
    if match:
        return match

    # extract nonwords (ex: &~jiewqioje)
    match = extract_markers(text, r"&~(\w+)", TokenType.NON_WORD)
    if match:
        return match


def _extract_gestures(text):
    """
    &=head:yes   Gesture
    &=claps      Gesture
    """
    gesture_tokens = []
    # gesture with colon (ex: &=head:yes)
    match = re.finditer(r"&=(\w+\:\w+)", text)
    if match:
        tolist = list(match)
        for m in tolist:
            gesture_tokens.append(Token(m.group(1), TokenType.GESTURE))

    # gesture without colon (ex: &=claps)
    match = re.finditer(r"&=(\w+)", text)
    if match:
        tolist = list(match)
        for m in tolist:
            gesture_tokens.append(Token(m.group(1), TokenType.GESTURE))

    return gesture_tokens


def _extract_retracing_overlap(text, start, end):
    """
    Repetition : <I wanted> [/]
    Retracing  : <the fish is> [//]
    """
    extract = re.finditer(r"\[(?![*:+])(\D*?)(\d*?)]", text)
    if extract:
        tolist = list(extract)
        if len(tolist) > 1:
            raise ValueError(
                "Multiple retracing/overlap markers found in a single word: " + text
            )
        for match in tolist:
            pattern = match.group(1)

            for markertype in MarkerType:
                if pattern == markertype.value:
                    return Marker(markertype, [start, end])

            raise ValueError("Invalid retracing/overlap marker: " + pattern)
    return None


def _extract_special_utterance_terminators(text):
    """
    +...     Trailing off
    +..?     Trail off to a question
    +!?      Question with exclamation
    +/.      Interrupted
    +/?      Interupted question
    +//.     Self-interruption
    +//?     Self-interruption with question
    +.       Transcription Break
    +"/.     Quotation Follows
    +".      Quotation Precedes
    +"       Quoted Utterance
    +^       Quick Uptake
    +,       Self-Completion
    ++       Other Completion
    """
    # extract special utterance terminators
    extract = re.findall(r"(?<!&)\+(?!<)\B\S+", text)
    if extract:
        tolist = list(extract)
        if len(tolist) > 1:
            raise ValueError(
                "Multiple special utterance terminators found in a single word: " + text
            )

        term = tolist[0]

        for terminatortype in TerminatorType:
            if term == terminatortype.value:
                return Terminator(term, terminatortype)
        raise ValueError("Invalid special utterance terminator: " + term)


def _extract_post_codes(text):
    """
    [+ gram]   Grammatical error
    [+ exc]    Exc
    [+ es]     empty speech
    [+ jar]    jargon
    [+ cir]    Circumlocution
    [+ per]    Perseveration
    """
    bracketed_group_re_pattern = r"\[([+])(\S*?)\]"
    match = re.findall(bracketed_group_re_pattern, text)
    if match:
        tolist = list(match)
        if len(tolist) > 1:
            raise ValueError("Multiple post codes found in a single word: " + text)
        # if target is a post code
        code = match[0][1]
        for post_code_type in PostCodeType:
            if code == post_code_type.value:
                return PostCode(code, post_code_type)
        raise ValueError("Invalid post code: " + code)
    return None


def _extract_word_error(text):
    """
    [* ...]  Word error
    """
    codes = text.split(":")
    for word_error_type in WordErrorType:
        if codes[0][0] == word_error_type.value:
            return word_error_type
    raise ValueError("Invalid word error: " + text)


def _extract_token_attributes(text, previous_token):
    """
    [* ...]    Word error
    [: ...]    Phonetic target
    """
    ## Common error in some chat files:
    # [: x@n] is sometimes used to represent an unknown target but often written as [* x@n] instead but it is not a valid word error. We will replace it with the correct format.
    if text == "[*x@n]":
        text = "[:x@n]"

    # Captures the type of the marker and the code/target. Group 1 is the type of the marker (e.g. *, :, +), group 2 is the code/target (e.g. gram, exc, ...).
    bracketed_group_re_pattern = r"^\[([:*])(\S*)\]$"
    match = re.match(bracketed_group_re_pattern, text)
    if match:
        # group 1 is the type of the marker, group 2 is the code
        if match.group(1) == ":":
            # if token is a phonetic target
            if previous_token is None:
                raise ValueError(
                    "Phonetic target should not be the first token in an utterance"
                )
            target = match.group(2)
            previous_token.target_word = target
            return True
        elif match.group(1) == "*":
            # if target is WordError [* ...] token
            if previous_token is None:
                raise ValueError(
                    "Word error should not be the first token in an utterance"
                )
            if text == "[*]":
                previous_token.word_error_type.append(WordErrorType.GENERIC)
                previous_token.word_error_code.append(text)
                return True

            word_error_text = match.group(2)
            word_error_type = _extract_word_error(word_error_text)
            if word_error_type:
                # Add word error to the last word token
                previous_token.word_error_type.append(word_error_type)
                previous_token.word_error_code.append(word_error_text)
            return True


def _extract_word_timemarks(text):
    if re.match(r"\x15(\d+)_+(\d+)\x15", text):
        extracted_timemarks = re.match(r"\x15(\d+)_+(\d+)\x15", text)
        word_timemarks = (
            int(extracted_timemarks.group(1)),
            int(extracted_timemarks.group(2)),
        )
        return word_timemarks
