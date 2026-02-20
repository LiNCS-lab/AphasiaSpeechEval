from typing import List
import os

import pylangacq as pla

from ..models import Marker, MarkerType, Token, TokenType


def get_sessions(file_path: str) -> List[str]:
    """
    Get the list of sessions from a chat file.

    Args:
        file_path (str): The path to the chat file.

    Returns:
        List[str]: The list of session names.

    """
    chat = pla.Reader.from_files([file_path], parallel=False)
    return [os.path.basename(p).split(".")[0] for p in chat.file_paths()]


def repetitions(
    utterance_tokens: List[Token], max_repetition_length: int = 7
) -> List[Marker]:
    """
    Find n-gram repetitions in a list of utterance tokens.
    Looks only at WORD type tokens. Other token types are ignored.

    Args:
        utterance_tokens (List[Token]): The list of utterance tokens.

    Returns:
        List[Marker]: The list of repetition markers.
    """
    repetitions_list = []
    token_list = list(enumerate(utterance_tokens))
    word_token_list = []
    for index, token in token_list:
        if token.token_type == TokenType.WORD:
            word_token_list.append((index, token))

    for n in range(1, max_repetition_length):  # n-gram size
        offsets = range(len(word_token_list) - n * 2 + 1)  # sliding window
        j = 0
        while j < len(offsets) - 1:
            ngram_left = word_token_list[offsets[j] : offsets[j] + n]  # left n-gram
            ngram_right = word_token_list[
                offsets[j] + n : offsets[j] + 2 * n
            ]  # right n-gram
            for i in range(n):
                if ngram_left[i][1].text != ngram_right[i][1].text:
                    j += 1
                    break
                if i == n - 1:
                    repetitions_list.append(
                        Marker(
                            MarkerType.REPETITION, [ngram_left[0][0], ngram_left[-1][0]]
                        )
                    )
                    j += n

    # Group repetitions that are enclosed by other repetitions
    repetitions_list = sorted(repetitions_list, key=lambda x: x.span[0])
    i = 0
    while i < len(repetitions_list) - 1:
        if repetitions_list[i].span[1] >= repetitions_list[i + 1].span[1]:
            repetitions_list.pop(i)
        else:
            i += 1

    return repetitions_list
