from enum import Enum


class TokenType(Enum):
    PUNCTUATION = 0
    PAUSE = 1
    WORD = 2
    WORD_ERROR = 3
    POST_CODE = 4
    TERMINATOR = 5
    UNINTELLIGIBLE = 6
    UNTRANSCRIBABLE = 7
    UNIDENTIFIABLE = 8
    SAT_MARKER = 9
    PHONOLOGICAL_FRAGMENT = "&+"
    FILLER = "&-"
    NON_WORD = "&~"
    GESTURE = "&="
    LAZY_OVERLAP = "+<"


# Tokens that would be present in a non-coded, direct, transcription
VERBATIM_TOKEN_TYPES = [
    TokenType.WORD,
    TokenType.PUNCTUATION,
    TokenType.PAUSE,
    TokenType.PHONOLOGICAL_FRAGMENT,
    TokenType.FILLER,
    TokenType.NON_WORD,
]


class WordType(Enum):
    NORMAL = 1
    ADDITION = "@a"
    BABBLING = "@b"
    CHILD_INVENTED = "@c"
    DIALECT = "@d"
    ECHOLALIA = "@e"
    FAMILY_SPECIFIC = "@f"
    FILLED_PAUSE = "@fp"
    GENERAL_SPECIAL = "@g"
    INTERJECTION = "@i"
    SPELLING = "@k"
    LETTER = "@l"
    LETTER_PLURAL = "@lp"
    NEOLOGISM = "@n"
    ONOMATOPOEIA = "@o"
    PHONOLOGICAL = "@p"
    META_LINGUISTIC = "@q"
    SECOND_LANGUAGE = "@s"
    SINGING = "@si"
    SIGNED_LANGUAGE = "@sl"
    SIGN_AND_SPEECH = "@sas"
    TEST_WORD = "@t"
    IPA_TRANSCRIPTION = "@u"
    WORD_PLAY = "@wp"
    EXCLUDED_WORDS = "@x"
    USER_DEFINED = "@z"


class PauseType(Enum):
    SHORT = "(.)"
    MEDIUM = "(..)"
    LONG = "(...)"
    DURATION = "(:)"

    def __str__(self):
        return self.value


class MarkerType(Enum):
    REPETITION = "/"
    RETRACING = "//"
    REFORMULATION = "///"
    OVERLAP_FOLLOWING = ">"
    OVERLAP_PRECEDING = "<"
    FALSE_START_WITHOUT_RETRACING = "/-"
    UNCLEAR_RETRACING_TYPE = "/?"
    STRESSING = "!"
    CONTRASTIVE_STRESSING = "!!"
    BEST_GUESS = "?"
    EXCLUDED = "e"

    def __str__(self):
        return f"[{self.value}]"


class TerminatorType(Enum):
    TRAILING_OFF = "+..."
    TRAIL_OFF_TO_A_QUESTION = "+..?"
    QUESTION_WITH_EXCLAMATION = "+!?"
    INTERRUPTED = "+/."
    INTERRUPTED_QUESTION = "+/?"
    SELF_INTERRUPTION = "+//."
    SELF_INTERRUPTION_WITH_QUESTION = "+//?"
    TRANSCRIPTION_BREAK = "+."
    QUOTATION_FOLLOWS = """+"/."""
    QUOTATION_PRECEDES = """+"."""
    QUOTED_UTTERANCE = '''+"'''
    QUICK_UPTAKE = "+^"
    SELF_COMPLETION = "+,"
    OTHER_COMPLETION = "++"

    def __str__(self):
        return f"{self.value}"


class PostCodeType(Enum):
    GRAMMATICAL = "gram"
    EXCLUDED = "exc"
    EMPTY_SPEECH = "es"
    JARGON = "jar"
    CIRCUMLOCUTION = "cir"
    PERSEVERATION = "per"

    def __str__(self):
        return f"[+ {self.value}]"


class WordErrorType(Enum):
    GENERIC = ""
    SEMANTIC = "s"
    PHONOLOGICAL = "p"
    NEOLOGISM = "n"
    MORPHOLOGICAL = "m"
    DYSFLUENCY = "d"

    def __str__(self):
        return f"[* {self.value}]"


class PartOfSpeech(Enum):
    ADJECTIVE = "adj"
    ADVERB = "adv"
    COMMUNICATOR = "co"
    COMPLEMENTIZER = "comp"
    CONJUNCTION = "conj"
    COORDINATOR = "coord"
    DETEMINER = "det"
    FILLER = "fil"
    INFINITIVE = "inf"
    NEGATIVE = "neg"
    NOUN = "n"
    ONOMATOPOEIA = "on"
    PARTICLE = "part"
    POSTMODIFIER = "post"
    PREPOSITION = "prep"
    PRONOUN = "pro"
    QUANTIFIER = "qn"
    VERB = "v"
    VERB_AUXILIARY = "aux"
    VERB_COPULA = "cop"
    VERB_MODAL = "mod"


class AffixType(Enum):
    PRESENT_PARTICIPLE = "PRESP"
    PAST = "PAST"
    PAST_PARTICIPLE = "PASTP"
    SINGULAR_3RD = "3S"
    PLURAL = "PL"
    SINGULAR_1ST = "1S"


class SateliteMarkerType(Enum):
    PREFIX = "‡"
    SUFFIX = "„"

    def __str__(self):
        return self.value
