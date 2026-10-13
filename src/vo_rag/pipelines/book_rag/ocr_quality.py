"""Per-chunk OCR text-quality gate (VOR-1452) -- precision first.

The book-level guards (``validator.is_degenerate_chunk_output`` /
``is_runaway_chunk_output``) judge averages over a whole book, and the
boilerplate filter is an exact-match allow-list, so a chunk of OCR noise from
a scanned page sails through both and is embedded like prose. ``assess``
judges ONE chunk's text and returns:

- ``"drop"`` -- unmistakable noise: nothing readable survives. Dropping real
  content is far worse than keeping some garbage (the v2 tagger already marks
  non-content passages), so a chunk with real words, well-formed numbers,
  training notation, table structure or readable letter-spaced text is never
  dropped;
- ``"keep"`` -- everything else, including real content read through bad
  OCR. (An earlier "flag" verdict for low-confidence OCR was removed: it
  could not be made reliable enough -- hand-read precision ~70%.)

It is a pure, deterministic function -- no model, no network, no embedding
call:

1. **Normalise** markup that is not OCR noise (``normalise``): markdown links
   keep their text, URLs, domains, the ``Comparison across columns:``
   scaffolding of the table linearizer, markdown table rules, bullets, dot
   leaders and soft hyphens go. A chunk of nothing but links/URLs is ebook
   navigation residue and is dropped.
2. **Tokenise** on whitespace and the ``=`` of linearized table cells (math
   text also on operators) and classify each token:

   readable
       ``word`` -- a dictionary word: 4+ letters found in the bundled English
       dictionary (``data/english_words.txt.gz``, 75k words from SCOWL; see
       ``data/SCOWL-COPYRIGHT.txt``) or the track-and-field vocabulary, via
       plain lookup, a common suffix/prefix, or a two-word compound; 1-3
       letters found in a curated list of common short words (the
       dictionary's own short entries -- "oe", "ty", "pa" -- are too
       permissive for OCR debris). Word-shaped strings outside the
       dictionary (names, German/Dutch/Italian words) count once the chunk
       shows it is language; any non-Latin-script word counts.
       ``num`` (numbers, times, ranges, percentages, units after a number),
       ``notation`` (``x5``, ``3x60``, ``w/``, ``4'^``, ``@95%``) and
       ``spaced`` (letter-spaced OCR: ``o f th e N a tio n a l``).
   neutral
       acronyms, letter/digit codes, list markers (``A.``, ``(2)``, an
       ordered ``a b c``, a weekday column ``M T W T F S S``), reference
       initials, unit labels, connectors, math variables.
   garbage
       punctuation/digit soup, stray letters, symbol-only tokens, unknown
       2-3 letter fragments, glued columns (``220-last``), and odd-cased
       strings (``tT``) at half weight.
3. **Decide** (``assess_detail``): protection first (never drop), then the
   drop test. Each threshold states its rationale.

Labelled evidence: ``tests/fixtures/ocr_quality_samples*.json`` (three rounds
of hand-labelled chunks) plus a hand-read full-corpus dry run (task report).
"""

from __future__ import annotations

import gzip
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from importlib import resources
from typing import Final, Literal

Verdict = Literal["keep", "drop"]

#: Chunk text shorter than this (stripped) is not judged: headings, list
#: labels and one-line notes carry too few tokens for any ratio to mean
#: anything.
MIN_ASSESS_CHARS: Final[int] = 20
#: Chunk text longer than this (stripped) is not judged either. This is a defensive
#: bound: the chunker's token ceiling keeps a chunk near 2.4k characters (the
#: zero-chunk fallback splits oversized elements too), so only an abnormal
#: chunk that the chunker's splitting should have prevented gets here;
#: judging it would cost time on untrusted input, and keeping it drops
#: nothing (precision first).
MAX_ASSESS_TEXT_CHARS: Final[int] = 16_000
#: A token longer than this is classified ``"overlong"``, uncounted, without
#: running the per-token patterns: no real word, number or notation is
#: this long (the labelled fixtures' longest token is 93 characters), and
#: counting it neither way keeps the gate from dropping more text.
MAX_TOKEN_CHARS: Final[int] = 256

# --- protection: a chunk showing any of these is never dropped ---------------
# Each is a sign of real content that OCR noise does not produce by accident.

#: A readable phrase: at least this many consecutive real words, or one fewer
#: when two of them have 4+ letters ("The Technique of Hurdling"). Labelled
#: noise strings together at most 4 accidental, mostly short, words.
PROTECT_MIN_PHRASE: Final[int] = 5
#: Real words: at least this many dictionary words of 4+ letters ...
PROTECT_MIN_WORDS: Final[int] = 3
#: ... making up at least this share of the chunk's 4+ letter tokens. Noise
#: throws up the odd accidental word ("Bear", "west"); readable text, even a
#: garbled form, is built of them. Labelled noise with 3+ accidental words
#: stays at or below 0.44.
PROTECT_MIN_WORD_SHARE: Final[float] = 0.5
#: Adjacent real words of 4+ letters -- a label or heading ("Left pole",
#: "Comprehensive Weight Training Program"): this many when most of the
#: chunk's long tokens are words, one more otherwise (long noise throws up
#: the odd adjacent pair of accidental words).
PROTECT_MIN_LONG_PHRASE: Final[int] = 2
#: Training notation: at least this share of content tokens is ``x5``,
#: ``w/``, ``4'^``... Labelled coaching shorthand is >= 0.27; labelled noise
#: and garbled chunks stay <= 0.08.
NOTATION_MIN_SHARE: Final[float] = 0.15
#: A linearized table of at least NUMERIC_TABLE_MIN_CELLS non-empty cells,
#: at least this share of them numbers, is a real table however garbled its
#: labels and the rest of its cells.
NUMERIC_TABLE_MIN_CELLS: Final[int] = 3
NUMERIC_TABLE_MIN_SHARE: Final[float] = 0.15
#: Measured content: at least this many numbers with units ("40cm", "200
#: meters") -- one integer measurement alone is too easily an accident of
#: noise ("68m") -- or one decimal measurement ("15.00kg", "0.50W").
PROTECT_MIN_MEASUREMENTS: Final[int] = 2
#: A list of times: at least this many clock or race times ("1:48.6",
#: "2:40").
CLOCK_TIMES_MIN: Final[int] = 3
#: Counted content: at least this many "number + real word" pairs ("90 chest
#: passes", "30 around", "3 sets").
QUANTIFIED_MIN: Final[int] = 2
#: A numbered list: at least this many line-opening list markers ("A.",
#: "(1)") and this many numbers ("B.165 A.110 D.330 (1) 52").
NUMBERED_LIST_MIN: Final[int] = 3
#: A weekly training form: at least this many day markers in weekly order,
#: or WEEKDAY_LINE_RUN_MIN when each opens its own row and carries an entry.
WEEKDAY_RUN_MIN: Final[int] = 3
WEEKDAY_LINE_RUN_MIN: Final[int] = 2
#: A coded workout schedule (printed training forms: "1A-10A1", "/6ABCD", "6A(3)",
#: "3x", and "/06", "/0B" where OCR read the leading 1 as a slash): at least
#: this many session codes -- a number joined to session letters A-F, a set
#: count, or a slash-number code -- however garbled the rest.
SESSION_CODES_MIN: Final[int] = 2
#: A link or URL with a short label of real words ("Hex Bar http://...") is
#: a reference, not noise: this many content tokens at most.
LINK_LABEL_MAX_TOKENS: Final[int] = 6
#: A column of codes: at least this many lines each holding one capital code
#: of the same length (athlete initials, country codes).
CODE_COLUMN_MIN: Final[int] = 8

# --- drop ----------------------------------------------------------------------

#: An unprotected chunk is dropped when fewer than DROP_MAX_READABLE_ANY of its
#: scored (readable + garbage) tokens are readable; or fewer than
#: DROP_MAX_READABLE when it also has either at least DROP_MIN_GARBAGE_TOKENS
#: garbage tokens while most of its 4+ letter tokens are not words (noise that
#: happens to contain real words: "review be désadhed Hi € Badiegaty ...
#: paws"), or no real word of 4+ letters at all. A small figure-label chunk
#: with a real word and little garbage ("F l =483N Flight path") is kept.
DROP_MAX_READABLE_ANY: Final[float] = 0.5
DROP_MAX_READABLE: Final[float] = 0.75
DROP_MIN_GARBAGE_TOKENS: Final[int] = 4

# --- scoring -------------------------------------------------------------------

#: Weight of an odd-cased token (``tT``, ``spcEND``) in the garbage count:
#: noise makes them, so do coaching abbreviations.
ODD_TOKEN_WEIGHT: Final[float] = 0.5

# --- structure detectors -------------------------------------------------------

#: Letter-spaced OCR: a run of at least LETTER_SPACED_MIN_TOKENS 1-3 letter
#: fragments, at least half and at least LETTER_SPACED_MIN_SINGLES of them
#: single letters, at most LETTER_SPACED_MAX_WORD_SHARE real short words,
#: collapsing to >= LETTER_SPACED_MIN_CHARS mostly-lowercase letters with a
#: word-like vowel share.
LETTER_SPACED_MIN_TOKENS: Final[int] = 5
LETTER_SPACED_MIN_SINGLES: Final[int] = 4
LETTER_SPACED_MAX_WORD_SHARE: Final[float] = 0.25
LETTER_SPACED_MIN_CHARS: Final[int] = 6
#: Single letters counting a list (a, b, c) or a week (M T W T F S S) are
#: list markers when there are at least this many in order.
SEQUENCE_MIN: Final[int] = 3
#: Word-shaped strings outside the dictionary count as words when the chunk
#: shows it is language: this share of its 4+ letter tokens are dictionary
#: words, or at least SHAPE_EVIDENCE_MIN_TOKENS letter tokens of which
#: SHAPE_EVIDENCE_MIN are word-shaped (Dutch, Italian, Polish, Latin prose).
LEXICON_EVIDENCE_MIN: Final[float] = 0.35
SHAPE_EVIDENCE_MIN: Final[float] = 0.7
SHAPE_EVIDENCE_MIN_TOKENS: Final[int] = 10

# --- vocabulary -----------------------------------------------------------------

#: Common 1-3 letter words of the corpus languages and reference lists. The
#: bundled dictionary is only consulted for 4+ letters.
_SHORT_WORDS: Final[str] = """
a i ah am an as at be by do go he if in is it me my no of oh on or ox so to
up us we ok vs etc aka eg ie cf al et ed eds pp vol no nd st rd th fig figs
eq ref sci res int am br mon tue wed thu fri sat sun jan feb mar apr jun jul
aug sep oct nov dec min max sec hr hrs yd yds ft kg km cm mm lb lbs ms mph
bpm rpm kcal deg avg ns
str tel fax nr dr mr mrs jr sr co ltd bzw usw ggf vgl hrsg ca abb tab kap
age sex gov med non ups rep inc yrs mod lat cal rev tic apt alt tow tsp yaw
gen hey ind wk wt yr mg oz adv mil sol aft sq app neg ail err hum nil ado
pep lib sim sup ex ad id ii iii iv vi vii ix xi xii xiv xv xx tee chi pi
ace act add ago aid aim air ale all and ant any ape arc are ark arm art ash
ask ate awe axe bad bag ban bar bat bay bed bee beg bet bid big bin bit bow
box boy bud bug bus but buy bye cab can cap car cat cod cow cry cub cue cup
cut dad day den dew did die dig dim dip doe dog dot dry due dug dye ear eat
egg ego elf elk elm end era eve eye fan far fat fed fee few fig fin fir fit
fix flu fly foe fog for fox fry fun fur gap gas gel gem get gin god got gum
gun gut guy gym had ham has hat hay hen her hid him hip his hit hog hop hot
how hub hue hug hut ice icy ill ink inn ion its ivy jam jar jaw jet job jog
joy jug keg key kid kin kit lab lad lag lap law lay led leg let lid lie lip
lit log lot low mad man map mat may men met mid mix mob mom mop mud mug nap
net new nod nor not now nut oak oar oat odd off oil old one opt orb ore our
out owe owl own pad pal pan par pat paw pay pea peg pen per pet pie pig pin
pit ply pod pop pot pro pub pun pup put rag ram ran rap rat raw ray red rib
rid rig rim rip rob rod rot row rub rug rum run rut sad sag sat saw say sea
see set sew she shy sin sip sir sit six ski sky sly sob son sow soy spa spy
sub sue sum sun tab tag tan tap tar tax tea ten the tie tin tip toe ton too
top toy try tub tug two use van vat vet via vow wag war was wax way web wed
wet who why wig win wit woe won wow yes yet you zip zoo
der die das und ist mit von zu den dem des ein nicht sich auf bei aus nach
im um es er sie wir ihr wie als bis ab da ob nun nur vor hat war wer ins vom
ihn uns zur zum le la les et est un une du de en dans pour sur par pas qui
que au aux ce il se sa son ses ne ou ni mes nos vos ont el los las y con por
del lo al una su sus mas muy ya ha han sin
"""

#: Track-and-field and training vocabulary missing from the general
#: dictionary.
_DOMAIN_WORDS: Final[str] = """
fartlek plyometric plyometrics plyo deadlift deadlifts biomechanics
biomechanical periodization periodisation periodized microcycle microcycles
macrocycle macrocycles mesocycle mesocycles skinfold skinfolds bioenergetic
biomotor biodynamic warmup warmups cooldown cooldowns superseries
supersets superset halfback kettlebell kettlebells prehab isometrics
steeplechase triathlon heptathlon decathlon pentathlon hurdler hurdlers
heptathlete heptathletes decathlete decathletes pentathlete pentathletes
sprinter sprinters overspeed hypertrophy proprioception proprioceptive
lactate glycolytic anaerobic aerobic vertimax plyos
"""

_UNITS: Final[str] = """
m cm mm km mi s sec secs ms min mins h hr hrs kg g lb lbs ft in yd yds mph hz n
nm kn w j bpm mmol l ml kcal cal kj rpm deg mb hpa kpa mmhg
"""

SHORT_WORDS: Final[frozenset[str]] = frozenset(_SHORT_WORDS.split())
DOMAIN_WORDS: Final[frozenset[str]] = frozenset(_DOMAIN_WORDS.split())
_UNIT_SET: Final[frozenset[str]] = frozenset(_UNITS.split())
_UNIT_ONLY: Final[frozenset[str]] = _UNIT_SET - SHORT_WORDS
#: Single-letter units count only right after a number ("7 m", "10 s").
_SINGLE_LETTER_UNITS: Final[frozenset[str]] = frozenset("msgh")
_ORDINAL_SUFFIXES: Final[frozenset[str]] = frozenset({"st", "nd", "rd", "th"})
#: Words that legitimately follow a number with a hyphen ("5-minute rest").
_UNIT_WORD_LIST: Final[str] = """
meter meters metre metres mile miles yard yards minute minutes second seconds
hour hours day days week weeks point points kilo kilos pound pounds rep reps
set sets step steps stride strides man lap laps year years foot feet inch
inches degree degrees percent km kg lb lbs rm
"""
_UNIT_WORDS: Final[frozenset[str]] = frozenset(_UNIT_WORD_LIST.split())
_SUFFIXES: Final[tuple[str, ...]] = (
    "s", "es", "ed", "d", "ing", "ly", "er", "ers", "ness", "ment", "ments", "al", "ally",
)  # fmt: skip
_PREFIXES: Final[tuple[str, ...]] = (
    "re", "un", "non", "pre", "post", "over", "under", "super", "sub", "inter", "intra",
    "counter", "anti", "bio", "micro", "macro", "meso", "multi", "semi", "self", "hyper",
)  # fmt: skip
_MIN_STEM: Final[int] = 4


@cache
def english_dictionary() -> frozenset[str]:
    """The bundled dictionary plus the domain vocabulary (lazy: read once,
    on first use)."""
    data = resources.files("vo_rag.pipelines.book_rag").joinpath("data/english_words.txt.gz")
    with data.open("rb") as raw:
        words = gzip.decompress(raw.read()).decode("utf-8").split()
    return frozenset(words) | DOMAIN_WORDS


#: Entries kept by the ``is_word`` / ``_segments`` caches:
#: the dispatcher processes are long-lived and see every distinct token of
#: every book; a bound keeps memory flat while a book's vocabulary still fits.
WORD_CACHE_SIZE: Final[int] = 200_000

#: Letter runs at least this long that split wholly into dictionary words
#: (each at least _MIN_SEGMENT letters) are words whose spaces OCR lost.
_MIN_SEGMENTED: Final[int] = 10
_MIN_SEGMENT: Final[int] = 2


@lru_cache(maxsize=WORD_CACHE_SIZE)
def _segments(word: str) -> bool:
    """True when *word* splits wholly into dictionary words."""
    n = len(word)
    ok = [True] + [False] * n
    for end in range(1, n + 1):
        for start in range(max(0, end - 20), end - _MIN_SEGMENT + 1):
            if ok[start] and is_word(word[start:end]):
                ok[end] = True
                break
    return ok[n]


@lru_cache(maxsize=WORD_CACHE_SIZE)
def is_word(word: str) -> bool:
    """True when lowercase *word* is a real word (see module docstring)."""
    if len(word) <= 3:
        return word in SHORT_WORDS
    words = english_dictionary()
    if word in words:
        return True
    for suffix in _SUFFIXES:
        stem = word[: -len(suffix)]
        if word.endswith(suffix) and len(stem) >= _MIN_STEM and stem in words:
            return True
    for prefix in _PREFIXES:
        rest = word[len(prefix) :]
        if word.startswith(prefix) and len(rest) >= _MIN_STEM and rest in words:
            return True
    return any(
        word[:i] in words and word[i:] in words for i in range(_MIN_STEM, len(word) - _MIN_STEM + 1)
    )


# --- normalisation patterns -----------------------------------------------------
_MD_LINK: Final = re.compile(r"!?\\?\[([^\[\]]*)\\?\]\([^()\s]*\)")
#: A link cut off by the end of the chunk: "... [Pr".
_TRUNCATED_LINK: Final = re.compile(r"(?<=\))\s*\\?\[[^\[\]\n]{0,40}$")
_DOMAIN: Final = re.compile(r"\b[\w-]+\.(?:com|org|net|edu|gov|io|de|uk|fr)\b", re.IGNORECASE)
#: The runs a bare domain can lie in (see ``_in_runs``).
_DOMAIN_RUN: Final = re.compile(r"[\w.-]+")
#: The tail of a link whose start the chunk cut off: "…“](../Text/c3.html#s1)".
_LINK_TAIL: Final = re.compile(r"\]\([^()\s]*\)")
#: The "label=" of a linearized table cell (see _table_cells).
_CELL_LABEL: Final = re.compile(r"(?:(?<=[,:]\s)|^)[^=,:\n]{1,80}=", re.MULTILINE)
_ESCAPED_BRACKET: Final = re.compile(r"\\[\[\]]")
#: A chunk made only of folio numbers ("ix", "xiii  xiv", "112 113"): layout noise.
#: Roman folios need two letters, so a lone variable ("x") is not one. What a
#: folio chunk can be: Roman folios, any count; one number;
#: or two consecutive numbers, a facing-page spread. Any other integer list
#: is content: years ("1968 1972 1976"), a rep scheme ("10 8 6 4 2").
_ROMAN_FOLIO: Final = r"(?=[ivxl]{2})(?:xl|x{0,3})(?:ix|iv|v?i{0,3})"
_ROMAN_FOLIOS_ONLY: Final = re.compile(rf"{_ROMAN_FOLIO}(?:\s+{_ROMAN_FOLIO})*", re.IGNORECASE)
_NUMBER_FOLIOS: Final = re.compile(r"(\d{1,4})(?:\s+(\d{1,4}))?")


def _is_folio_only(text: str) -> bool:
    if _ROMAN_FOLIOS_ONLY.fullmatch(text):
        return True
    numbers = _NUMBER_FOLIOS.fullmatch(text)
    if numbers is None:
        return False
    first, second = numbers.groups()
    return second is None or int(second) == int(first) + 1


_URL: Final = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
#: Contact details: e-mail addresses and labelled phone/fax numbers.
_EMAIL: Final = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
#: The runs an address can lie in (see ``_in_runs``).
_EMAIL_RUN: Final = re.compile(r"[\w.+@-]+")
_PHONE: Final = re.compile(
    r"\b(?:tel|fax|phone|telefon|telephone|mobile|mob|handy)\b\.?:?\s*\+?[\d(][\d ()/.-]{5,}\d",
    re.IGNORECASE,
)
_TABLE_HEADER: Final = "Comparison across columns:"
#: Bullets, including "¢", which OCR makes of a round bullet.
_BULLETS: Final = re.compile(r"[•▪◦‣∙●○■□►➢✓¢]")  # noqa: RUF001
#: Dot leaders of contents pages, including leaders OCR read as letters
#: (".....cccccscccssessess"), rules and underscores.
_LEADERS: Final = re.compile(r"(?:\.{3,}[ceos]*|…+)(?:\s*[ivxlc]{1,6}\b)?|_{3,}|-{4,}")  # noqa: RUF001
#: Markdown table rules and cell bars.
#: Linear time: a dash run is taken whole and only from its first dash.
_MD_TABLE_RULE: Final = re.compile(r"\|?(?:\s*+:?(?<!-)-{3,}+:?\s*+\|)+\s*:?-*:?")
_SOFT_HYPHEN: Final = "\u00ad"
#: Control characters PDF extraction leaves at hyphenation points ("Concen\x02").
_CONTROL_CHARS: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# --- token patterns ---------------------------------------------------------------
_VOWELS: Final[str] = "aeiouyäöüàâáéèêëîïíôóûùúœæåø"
_EDGE_PUNCT: Final[str] = "()[]{}.,;:!?\"'`“”‘’*-–—"  # noqa: RUF001 - smart quotes/dashes are the point
_CONNECTORS: Final[frozenset[str]] = frozenset(
    "&+|/-–—:±°·%−×÷<>≈≤≥∆Δ→←↑↓⇑⇓⇔⇒=′″©®™$€£()[]{}_"  # noqa: RUF001
)  # noqa: RUF001
_MARKER: Final = re.compile(
    r"^(?:\(?(?:[A-Z]|\d{1,3})[.)]|\(?[a-z][)\]}]|#\d{1,3}|[a-z]\.)$"
)  # "A.", "(2)", "g)", "#5", "e." (as in "e. V.", "z. B.")
_MARKER_PREFIX: Final = re.compile(r"^[A-Z]\.(?=\S)")
_NUMBER: Final = re.compile(
    r"^[<>~+±$€£]?\.?[\d½¼¾⅓⅔][\d½¼¾⅓⅔.,:/'\"’′″±–-]*\+?[%°]?"  # noqa: RUF001
    r"(?:st|nd|rd|th|s|m|k|y|km|kg|min|sec|mph|mi|ft|in|lb|lbs|reps?|sets?|h|hr|cm|mm)?$",
    re.IGNORECASE,
)
_NOTATION: Final = re.compile(
    r"(?:^|[^a-z])x\d|\dx\d|^w/|\^|@\d|^@$|\d['’]s?$",  # noqa: RUF001
    re.IGNORECASE,
)
#: Measurements with units between number groups: "46.74m/153'4\"", "1m85".
#: Linear time: each run of number characters is taken whole (units are
#: letters, never part of a run, so a run has nothing to give back) and "a
#: digit in the first run" is a lookahead instead of a backtracking split.
_MEASUREMENT: Final = re.compile(
    r"^(?=[.,:/'\"’′″+–-]*+\d)[\d.,:/'\"’′″+–-]++"  # noqa: RUF001
    r"(?:(?:m|cm|mm|km|s|kg|g|lb|lbs|ft|in|yd|h|min|sec|ms)[\d.,:/'\"’′″+–-]++)+"  # noqa: RUF001
    r"(?:m|cm|mm|km|s|kg|g|lb|lbs|ft|in|yd|h|min|sec|ms)?$"
)
#: Grouped small numbers: "weeks 1+2", "4+5" (a lone "+" between long digit
#: runs, "191+8", is a misread digit and stays soup).
_SMALL_SUM: Final = re.compile(r"^\d{1,2}(?:\+\d{1,2})+$")
#: Roman-numeral section references in contents pages: "XIV.1", "XV.2".
#: Coded training sessions ("1A-4-3-8A", "14F"): numbers with session letters.
_SESSION_CODE: Final = re.compile(
    r"(?:\d[A-F]{1,4}(?![a-z])|/[A-F]{1,4}\b|(?<![\w.])\d{1,2}x(?![\w])"
    r"|(?<![\w/])/\d{1,2}[A-F]?(?![\w]))"
)
_NUMBER_THEN_WORD: Final = re.compile(r"^\d+(?:[.,]\d+)?[%'’]?([a-z]{2,})\d*$")  # noqa: RUF001
#: A word glued to its number: "TABLE9.4", "Figure8.16".
_WORD_THEN_NUMBER: Final = re.compile(r"^([A-Za-z]{3,})\d+(?:\.\d+)*$")
_SECTION_REF: Final = re.compile(r"^[IVXLC]+\.\d+(?:\.\d+)*$")
_UNIT_RATIO: Final = re.compile(r"^[a-zA-Z]{1,4}/[a-zA-Z]{1,4}[²2]?$")
_FIGURE_REF: Final = re.compile(r"^\d+(?:\.\d+)+[a-z](?:-[a-z])?$")
_DOTTED_ABBREVIATION: Final = re.compile(r"^[A-Za-z]{1,3}(?:\.[A-Za-z]{1,3}){1,3}$")
_DIGIT_LOOKALIKE: Final = re.compile(r"^(?=.*O)[OSIZB\d]{2,5}[a-z]{0,2}$")
_CODE: Final = re.compile(r"^(?=.*\d)(?=.*[^\W\d_])[^\W_]{2,10}$")
#: A 3+ digit number hyphen-joined to a word ("220-last"); shorter numbers
#: make ordinary compounds ("2-legged", "5-minute").
_GLUED: Final = re.compile(r"^\d{3,}+[\d/.%]*+-([^\W\d_]{3,})$")  # possessive: linear
_LETTERS_WITH_JOINS: Final = re.compile(r"^[^\W\d_]+(?:[-'’/][^\W\d_]+)*$")  # noqa: RUF001
_JOIN: Final = re.compile(r"[-'’/]")  # noqa: RUF001
_LONG_CONSONANT_RUN: Final = re.compile(rf"[^{_VOWELS}]{{5,}}")
_TRIPLE_LETTER: Final = re.compile(r"(.)\1\1")
_HAS_WORD_CHAR: Final = re.compile(r"\w")
_SPLIT: Final = re.compile(r"[\s=—]+")
_MATH_CONTEXT: Final = re.compile(
    r"[−×÷·√∑∏∫∂∆≈≠≤≥∞²³¹⁰⁴⁵⁶⁷⁸⁹\u0391-\u03a9\u03b1-\u03c9]"  # noqa: RUF001
    r"|\b(?:sin|cos|tan|arctan|atan|log|ln|exp|sqrt|lim)\s*[(\d]"
)
#: Formula text without special symbols: two or more "v =" assignments.
_ASSIGNMENT: Final = re.compile(r"(?:^|\s)[A-Za-z]{1,2}[,;]?\s*=\s*[A-Za-z?\d]")
_MIN_ASSIGNMENTS: Final[int] = 2
#: ... or one assignment to an expression of variables: "v = d / t".
_FORMULA: Final = re.compile(
    r"(?:^|\s)[A-Za-z]{1,2}\s*=\s*[A-Za-z\d.]{1,4}\s*[/*+×·÷−-]\s*[A-Za-z\d.]{1,4}(?:\s|$)"  # noqa: RUF001
)
_MATH_SPLIT: Final = re.compile(r"[\s=()\[\]{}+−×·÷*/^,;²³¹⁰⁴⁵⁶⁷⁸⁹—]+")  # noqa: RUF001
_MAX_VARIABLE_LEN: Final[int] = 3
_MAX_FRAGMENT_LEN: Final[int] = 3
_LATIN_SCRIPT_END: Final = "\u024f"
_WEEK: Final = "MTWTFSS"

_READABLE: Final[frozenset[str]] = frozenset({"word", "num", "notation", "spaced", "foreign"})
_GARBAGE: Final[frozenset[str]] = frozenset({"junk", "stray", "symbol", "glued", "fragment"})
_UNCOUNTED: Final[frozenset[str]] = frozenset(
    {"marker", "connector", "variable", "acronym", "code", "shape", "unit", "initial", "overlong"}
)
_VALID_PART: Final[frozenset[str]] = frozenset(
    {"word", "shape", "num", "code", "acronym", "notation"}
)


@dataclass(frozen=True)
class QualityAssessment:
    """``assess`` verdict plus the signals and rule that produced it."""

    verdict: Verdict
    reason: str
    content_tokens: int = 0
    readable_share: float = 1.0
    garbage_share: float = 0.0
    notation_share: float = 0.0
    long_words: int = 0
    word_share: float = 0.0
    letter_spaced_tokens: int = 0
    numeric_cell_share: float = 0.0
    protected_by: str = ""


def _sub_in_runs(pattern: re.Pattern[str], run: re.Pattern[str], text: str) -> str:
    """``pattern.sub(" ", text)`` in linear time.

    ``_EMAIL`` and ``_DOMAIN`` restart a scan to the end of a run of address
    characters at every position of it, so one long run ("1"*16000 + "a")
    costs quadratic time. Every match lies inside one maximal ``run`` (its
    characters all belong to it, and ``\\b`` sees a non-word character past
    either end of a run), so substituting run by run gives the same text.
    A run longer than MAX_TOKEN_CHARS is left alone: it is one over-long token,
    which ``classify_token`` leaves uncounted anyway.
    """

    def within(match: re.Match[str]) -> str:
        found = match.group(0)
        return pattern.sub(" ", found) if len(found) <= MAX_TOKEN_CHARS else found

    return run.sub(within, text)


def _count_in_runs(pattern: re.Pattern[str], run: re.Pattern[str], text: str) -> int:
    """``len(pattern.findall(text))`` in linear time (see ``_sub_in_runs``)."""
    return sum(
        len(pattern.findall(match.group(0)))
        for match in run.finditer(text)
        if len(match.group(0)) <= MAX_TOKEN_CHARS
    )


def normalise(text: str) -> tuple[str, bool]:
    """Strip markup that is not OCR noise. Returns the cleaned text and
    whether the chunk was nothing but links/URLs (navigation residue)."""
    text = _CONTROL_CHARS.sub("", text.replace(_SOFT_HYPHEN, ""))
    text = _TRUNCATED_LINK.sub(" ", text)
    had_markup = bool(_MD_LINK.search(text) or _URL.search(text))
    outside_links = _ESCAPED_BRACKET.sub(" ", _URL.sub(" ", _MD_LINK.sub(" ", text)))
    markup_only = had_markup and not any(ch.isalnum() for ch in outside_links)

    def link_text(match: re.Match[str]) -> str:
        label = match.group(1)
        return f" {label} " if any(ch.isalpha() for ch in label) else " "

    cleaned = _MD_LINK.sub(link_text, text)
    cleaned = _LINK_TAIL.sub(" ", cleaned)
    cleaned = _ESCAPED_BRACKET.sub(" ", cleaned)
    cleaned = _sub_in_runs(_EMAIL, _EMAIL_RUN, cleaned)
    cleaned = _PHONE.sub(" ", cleaned)
    cleaned = _URL.sub(" ", cleaned)
    cleaned = _sub_in_runs(_DOMAIN, _DOMAIN_RUN, cleaned)
    if _TABLE_HEADER in cleaned:
        # Each cell repeats its column label ("label=value"): count each
        # label once, or one garbled label is counted once per row.
        seen: set[str] = set()

        def first_label(match: re.Match[str]) -> str:
            label = match.group(0)[:-1].strip()
            if label in seen:
                return " "
            seen.add(label)
            return f" {label} "

        cleaned = _CELL_LABEL.sub(first_label, cleaned.replace(_TABLE_HEADER, " "))
    cleaned = _MD_TABLE_RULE.sub(" ", cleaned).replace("|", " ")
    cleaned = _BULLETS.sub(" ", cleaned).replace("**", " ").replace("__", " ")
    cleaned = _LEADERS.sub(" ", cleaned)
    return cleaned, markup_only


def _properly_cased(part: str) -> bool:
    return part.islower() or part.isupper() or (part[0].isupper() and part[1:].islower())


def _word_shaped(part: str) -> bool:
    """A letter string with the vowel structure of a real word in a Latin-
    script language: >= 4 letters, sane casing, 15-75% vowels, no run of 5+
    consonants or 3 identical letters."""
    low = part.lower()
    if len(low) < 4 or not _properly_cased(part):
        return False
    vowel_share = sum(ch in _VOWELS for ch in low) / len(low)
    if not 0.15 <= vowel_share <= 0.75:
        return False
    return not (_LONG_CONSONANT_RUN.search(low) or _TRIPLE_LETTER.search(low))


def _classify_letters(token: str, *, math: bool) -> str:
    if any(ch.isalpha() and ch > _LATIN_SCRIPT_END for ch in token):
        return "variable" if len(token) == 1 else "foreign"
    low = token.lower()
    if math and len(token) <= _MAX_VARIABLE_LEN and low not in SHORT_WORDS:
        return "variable"
    if len(token) <= _MAX_FRAGMENT_LEN and token.isupper() and len(token) > 1:
        return "acronym"  # "SIM", "TIN": capitals are not the short word
    if len(token) == 1:
        if token in ("x", "X"):
            return "times"  # resolved by its neighbours
        return "word" if token in ("a", "A", "I") else "stray"
    parts = [p for p in _JOIN.split(token) if p]
    if len(parts) > 1 and any(len(p) > 1 for p in parts):
        # A one-letter prefix or suffix of a compound is part of the word:
        # "E-Mail", "x-ray", "V-Handle", "T-shirt".
        parts = [p for p in parts if len(p) > 1]
    if all(_properly_cased(p) for p in parts):
        if all(is_word(p.lower()) for p in parts):
            return "word"
        if all(is_word(p.lower()) or _word_shaped(p) for p in parts):
            return "shape"
    if len(token) >= _MIN_STEM and all(is_word(p.lower()) for p in parts):
        return "word"  # small-caps OCR: "ENERGy", "SySTEM"
    if token.isupper() and len(token) <= 6:
        return "acronym"
    if len(token) >= _MIN_SEGMENTED and _segments(token.lower()):
        return "word"  # spaces lost: "PEAKINGPHASEWORKOUTS", "IntensityandVolume"
    if _DIGIT_LOOKALIKE.match(token):
        return "code"
    if len(token) <= _MAX_FRAGMENT_LEN and _properly_cased(token):
        return "fragment"
    return "odd"


def classify_token(token: str, *, math: bool = False) -> str:
    """Class of one edge-stripped token (see module docstring)."""
    if len(token) > MAX_TOKEN_CHARS:
        return "overlong"
    if not _HAS_WORD_CHAR.search(token.replace("_", "")):
        return "connector" if set(token) <= _CONNECTORS else "symbol"  # "_" rules count
    if token[-1] in "%°" and _LETTERS_WITH_JOINS.match(token[:-1]):
        token = token[:-1]
    glued = _GLUED.match(token)
    if glued and glued.group(1).lower() not in _UNIT_WORDS:
        return "glued"
    if _NOTATION.search(token):
        return "notation"
    if (
        _NUMBER.match(token)
        or _SMALL_SUM.match(token)
        or _MEASUREMENT.match(token)
        or _UNIT_RATIO.match(token)
        or _FIGURE_REF.match(token)
    ):
        return "num"
    if _SECTION_REF.match(token):
        return "marker"
    if _LETTERS_WITH_JOINS.match(token):
        return _classify_letters(token, math=math)
    if _DOTTED_ABBREVIATION.match(token):
        return "word"
    if _CODE.match(token):
        return "code"
    glued_word = _NUMBER_THEN_WORD.match(token)
    if glued_word and is_word(glued_word.group(1).lower()):
        return "num"  # a lost space: "70%of", "4to6"
    labelled = _WORD_THEN_NUMBER.match(token)
    if labelled and is_word(labelled.group(1).lower()):
        return "word"  # a lost space: "TABLE9.4"
    runs = re.findall(r"[^\W\d_]+", token)
    if (
        runs
        and sum(map(len, runs)) >= _MIN_SEGMENTED
        and all(len(r) < _MIN_SEGMENT or _segments(r.lower()) for r in runs)
    ):
        return "word"  # "TABLE12.5PEAKINGPHASEWORKOUTS"
    core = token.strip("-")
    if "-" in core:
        parts = core.split("-")
        if all(parts):
            kinds = [classify_token(p, math=math) for p in parts]
            if all(k in _VALID_PART for k in kinds):
                if "notation" in kinds:
                    return "notation"
                return "num" if all(k == "num" for k in kinds) else "code"
    return "junk"


def _is_math(text: str) -> bool:
    if _MATH_CONTEXT.search(text):
        return True
    # "v = ..." assignments mark formulas -- except in a linearized table,
    # whose every cell is "label=value".
    return _TABLE_HEADER not in text and (
        len(_ASSIGNMENT.findall(text)) >= _MIN_ASSIGNMENTS or bool(_FORMULA.search(text))
    )


@dataclass
class _Token:
    raw: str
    kind: str
    line: int = 0


def _tokens(text: str, *, math: bool | None = None) -> list[_Token]:
    """Classified tokens of normalised *text*, markers and connectors included."""
    tokens: list[_Token] = []
    if math is None:
        math = _is_math(text)
    split = _MATH_SPLIT if math else _SPLIT
    for line_no, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if not _HAS_WORD_CHAR.search(stripped):
            kind = "connector" if set(stripped) <= _CONNECTORS | {" "} else "symbol"
            tokens.append(_Token(stripped, kind, line_no))
            continue
        for raw in _split_paths(split.split(stripped)):
            if _MARKER_PREFIX.match(raw):
                raw = raw[2:]
            if not raw:
                continue
            if _MARKER.match(raw):
                tokens.append(_Token(raw, "marker", line_no))
                continue
            token = raw.strip(_EDGE_PUNCT)
            if not token:
                kind = "connector" if set(raw) <= _CONNECTORS | set(_EDGE_PUNCT) else "symbol"
                tokens.append(_Token(raw, kind, line_no))
                continue
            tokens.append(_Token(token, classify_token(token, math=math), line_no))
    _resolve_context(tokens)
    return tokens


#: A number carrying a unit: "40cm", "7.44m", "90%", "25°".
_MEASURE_UNITS: Final = (
    r"(?:m|cm|mm|km|mi|s|sec|secs|ms|kg|g|lbs?|ft|in|yds?|y|h|min|mph|kmh|km/h|m/s|W|kW|N|J|Hz"
    r"|meters?|metres?|yards?|miles?|seconds?|minutes?|mins|laps?|reps?|pounds)"
)
_MEASURED: Final = re.compile(rf"(?<![\w.])\d+(?:[.,]\d+)?\s?{_MEASURE_UNITS}(?![\w])")
#: A decimal measurement ("15.00kg", "0.50W", "6.16 secs") is precise enough
#: that noise does not make it: one protects a chunk.
_DECIMAL_MEASURED: Final = re.compile(rf"(?<![\w.])\d+[.,]\d+\s{{0,2}}{_MEASURE_UNITS}(?![\w])")
#: A count of something: a number followed by a real word ("90 chest passes",
#: "30 around", "3 sets"); see QUANTIFIED_MIN.
_QUANTIFIED: Final = re.compile(r"(?<![\w.])\d{1,4}\s+([a-z]{4,})\b")
#: Clock times and race times: "1:48.6", "2:40", "10:30".
_CLOCK_TIME: Final = re.compile(r"(?<![\w.:])\d{1,2}:\d{2}(?:[.,]\d{1,2})?(?![\w:])")
#: Numbered or lettered list markers opening a line ("A.", "(1)", "B.165").
_LIST_MARKER: Final = re.compile(r"^\s*(?:[A-Z]\.|\(\d{1,2}\))", re.MULTILINE)
#: Day markers of a weekly form: "M", "T/A", "W/B".
_DAY_MARKER: Final = re.compile(r"^[MTWFS](?:/|$)")


def _a_link_label_reads_as_language(text: str) -> bool:
    """Whether one link label of *text* holds a protected phrase.

    A chunk of nothing but links is navigation residue ("[Program Guide](#..)
    [The Program](#..)") -- unless a label is itself language: a hyperlinked
    "Further reading" list ("[Avery, T. Periodization: Theory and Methodology
    of Training](https://..)"). Each label is judged on its own, so short
    navigation labels never add up to a phrase across links.
    """
    for match in _MD_LINK.finditer(text):
        phrase, phrase_long = _longest_phrase(_tokens(match.group(1)))
        if phrase >= PROTECT_MIN_PHRASE or (
            phrase >= PROTECT_MIN_PHRASE - 1 and phrase_long >= PROTECT_MIN_LONG_PHRASE
        ):
            return True
    return False


def _longest_phrase(tokens: list[_Token]) -> tuple[int, int]:
    """(words, words of 4+ letters) of the longest run of consecutive real
    words (2+ letters, or a / I)."""
    best = (0, 0)
    current = (0, 0)
    for tok in tokens:
        if tok.kind in ("marker", "connector"):
            continue
        if tok.kind in ("word", "foreign") and (len(tok.raw) > 1 or tok.raw in ("a", "A", "I")):
            current = (current[0] + 1, current[1] + (len(tok.raw) >= _MIN_STEM))
            best = max(best, current)
        else:
            current = (0, 0)
    return best


def _longest_long_phrase(tokens: list[_Token]) -> int:
    """Longest run of consecutive real words of 4+ letters ("Left pole",
    "Comprehensive Weight Training Program"): noise throws up scattered
    accidental words, not adjacent ones."""
    best = current = 0
    for tok in tokens:
        if tok.kind in ("marker", "connector"):
            continue
        if tok.kind in ("word", "foreign") and len(tok.raw) >= _MIN_STEM:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _code_column(tokens: list[_Token]) -> int:
    """Most lines holding nothing but one capital code of the same length
    ("CON", "ACU", "STO"...): a table column of athlete or country codes."""
    per_line: dict[int, list[_Token]] = defaultdict(list)
    for tok in tokens:
        per_line[tok.line].append(tok)
    lengths = Counter(
        len(line[0].raw)
        for line in per_line.values()
        if len(line) == 1 and line[0].raw.isupper() and line[0].raw.isalpha()
    )
    return max(lengths.values(), default=0)


def _weekday_run(tokens: list[_Token], *, line_initial: bool = False) -> int:
    """Longest run of day markers in weekly order (M T W T F S S, up to two
    days may be missing between two markers): the rows of a hand-written weekly
    training form, however garbled the entries."""
    best = 0
    run = 0
    positions: set[int] = set()
    per_line = Counter(tok.line for tok in tokens)
    last_line = -1
    for tok in tokens:
        first_in_line = tok.line != last_line
        last_line = tok.line
        raw = tok.raw
        if not _DAY_MARKER.match(raw):
            continue
        if line_initial and not (first_in_line and per_line[tok.line] > 1):
            continue  # a row opens with its day and carries an entry
        day = raw[0].upper()
        nxt = {
            (p + step) % 7 for p in positions for step in (1, 2, 3) if _WEEK[(p + step) % 7] == day
        }
        if run and nxt:
            run += 1
            positions = nxt
        else:
            run = 1
            positions = {i for i, d in enumerate(_WEEK) if d == day}
        best = max(best, run)
    return best


def _split_paths(raws: list[str]) -> list[str]:
    """Split path-like tokens ("E4492/Avery/Periodization,5E/333215/Fig",
    two or more slashes) into their parts -- a production slug, not OCR
    soup -- and words from a bracket glued to them ("CSA(cm", "a(max)")."""
    out: list[str] = []
    for raw in raws:
        parts = raw.split("/") if raw.count("/") >= _PATH_MIN_SLASHES else [raw]
        for part in parts:
            for piece in _LOST_SPACE.split(part):
                out.extend(_GLUED_BRACKET.split(piece) if _GLUED_BRACKET.search(piece) else [piece])
    return out


#: A space lost between a word and a number: "Excellent.26", '16"andbelow'.
_LOST_SPACE: Final = re.compile(r"(?<=[^\W\d_]{3})\.(?=\d)|(?<=\d[\"'”’])(?=[^\W\d_]{2})")  # noqa: RUF001


#: A bracket between two letters: "CSA(cm", "a(max)".
#: A bracket glued between letters ("CSA(cm", "a(max)") or opening after a
#: number ("2.68(+0.03)").
_GLUED_BRACKET: Final = re.compile(r"(?<=[^\W\d_])[()](?=[^\W\d_])|(?<=\d)\((?=[+\-±\dt])")


_PATH_MIN_SLASHES: Final[int] = 2
_BARE_NUMBER: Final = re.compile(r"^[<>~+±-]?\d[\d.,:/-]*$")


def _resolve_context(tokens: list[_Token]) -> None:
    """Classes that depend on neighbours."""
    content = [t for t in tokens if t.kind not in ("marker", "connector")]
    for i, tok in enumerate(content):
        prev = content[i - 1].kind if i else ""
        nxt = content[i + 1].kind if i + 1 < len(content) else ""
        low = tok.raw.lower()
        after_bare_number = i > 0 and bool(_BARE_NUMBER.match(content[i - 1].raw))
        unit_after_number = (
            tok.raw in _SINGLE_LETTER_UNITS
            if len(tok.raw) == 1
            else low in _UNIT_SET or low in _ORDINAL_SUFFIXES
        )
        if after_bare_number and unit_after_number:
            tok.kind = "num"
            continue
        if len(tok.raw) > 1 and tok.raw in _UNIT_ONLY and tok.kind != "num":
            tok.kind = "unit"
            continue
        if tok.kind == "times":
            near_number = {"num", "notation"} & {prev, nxt}
            tok.kind = "notation" if near_number else "stray"
    _mark_sequences(content)
    _mark_letter_spacing(content)
    for i, tok in enumerate(content):
        if (
            i
            and tok.kind == "stray"
            and tok.raw.isupper()
            and content[i - 1].kind in ("word", "shape")
            and content[i - 1].line == tok.line
            and content[i - 1].raw[:1].isupper()
        ):
            tok.kind = "initial"  # "Hettinger T": a reference author


def _mark_sequences(content: list[_Token]) -> None:
    """Single letters that count a list (a b c) or a week (M T W T F S S)
    are list markers: noise does not spell either."""
    singles = [t for t in content if len(t.raw) == 1 and t.raw.isalpha()]

    def mark(run: list[_Token]) -> None:
        if len(run) >= SEQUENCE_MIN:
            for t in run:
                t.kind = "marker"

    run: list[_Token] = []
    for tok in singles:
        if run and ord(tok.raw.lower()) == ord(run[-1].raw.lower()) + 1:
            run.append(tok)
            continue
        mark(run)
        run = [tok] if tok.raw.lower() == "a" else []
    mark(run)

    run = []
    positions: set[int] = set()
    for tok in singles:
        nxt = {(p + 1) % 7 for p in positions if _WEEK[(p + 1) % 7] == tok.raw}
        if run and nxt:
            run.append(tok)
            positions = nxt
            continue
        mark(run)
        positions = {i for i, day in enumerate(_WEEK) if day == tok.raw}
        run = [tok] if positions else []
    mark(run)


def _mark_letter_spacing(content: list[_Token]) -> None:
    run: list[_Token] = []

    def close() -> None:
        if len(run) >= LETTER_SPACED_MIN_TOKENS:
            singles = sum(len(t.raw) == 1 for t in run)
            real = sum(len(t.raw) > 1 and t.raw.lower() in SHORT_WORDS for t in run)
            collapsed = "".join(t.raw for t in run)
            lower_share = sum(ch.islower() for ch in collapsed) / len(collapsed)
            vowel_share = sum(ch.lower() in _VOWELS for ch in collapsed) / len(collapsed)
            if (
                singles * 2 >= len(run)
                and singles >= LETTER_SPACED_MIN_SINGLES
                and real <= LETTER_SPACED_MAX_WORD_SHARE * len(run)
                and len(collapsed) >= LETTER_SPACED_MIN_CHARS
                and lower_share >= 0.5
                and 0.25 <= vowel_share <= 0.6
            ):
                for t in run:
                    t.kind = "spaced"
        run.clear()

    for tok in content:
        if tok.raw.isalpha() and len(tok.raw) <= 3 and tok.kind not in ("variable", "marker"):
            run.append(tok)
        else:
            close()
    close()


_HEADER_END: Final = re.compile(r"\.(?:\s|$)")
_CELL: Final = re.compile(r"([^=,:]{1,80}?)=([^=]*?)(?=,\s|\.\s|\.?$|;\s)")


def _cell_kinds(value: str, *, math: bool) -> list[str]:
    return [t.kind for t in _tokens(value, math=math) if t.kind not in _UNCOUNTED]


def _is_number_cell(kinds: list[str]) -> bool:
    """A cell holding a number and nothing that is garbage: "30.7",
    "2.60 (t0.06) - 2.68(+0.03)", "7.44 m"."""
    return "num" in kinds and not any(k in _GARBAGE or k == "odd" for k in kinds)


def _numeric_cell_share(text: str) -> float:
    """Share of a linearized table's non-empty body cells that hold a number
    (see ``_is_number_cell``); 0 for text that is not one, or has fewer than
    NUMERIC_TABLE_MIN_CELLS cells."""
    if _TABLE_HEADER not in text:
        return 0.0
    math = _is_math(text)
    text = _CONTROL_CHARS.sub("", text.replace(_SOFT_HYPHEN, ""))
    body = _URL.sub(" ", _MD_LINK.sub(" ", text)).partition(_TABLE_HEADER)[2]
    parts = _HEADER_END.split(body, maxsplit=1)
    body = parts[1] if len(parts) > 1 else ""
    cells = [
        _cell_kinds(value, math=math)
        for line in body.splitlines()
        for match in _CELL.finditer(line)
        if (value := _LEADERS.sub(" ", match.group(2)).strip(" ."))
    ]
    if len(cells) < NUMERIC_TABLE_MIN_CELLS:
        return 0.0
    return sum(_is_number_cell(kinds) for kinds in cells) / len(cells)


def assess_detail(text: str) -> QualityAssessment:
    """Verdict plus signals for one chunk's text (see module docstring)."""
    if len(text.strip()) > MAX_ASSESS_TEXT_CHARS:
        return QualityAssessment("keep", "too_long_to_judge")
    if _is_folio_only(text.strip()):
        return QualityAssessment("drop", "page_numbers_only")
    if len(text.strip()) < MIN_ASSESS_CHARS:
        return QualityAssessment("keep", "too_short_to_judge")
    cleaned, markup_only = normalise(text)
    linked = bool(_URL.search(text) or _MD_LINK.search(text))
    contacts = _count_in_runs(_EMAIL, _EMAIL_RUN, text) + len(_PHONE.findall(text))
    if markup_only and not _a_link_label_reads_as_language(text):
        return QualityAssessment("drop", "links_or_urls_only")
    tokens = _tokens(cleaned)
    kinds = Counter(t.kind for t in tokens)
    measurements = len(_MEASURED.findall(cleaned))
    decimal_measurements = len(_DECIMAL_MEASURED.findall(cleaned))
    quantified = sum(is_word(word) for word in _QUANTIFIED.findall(cleaned))
    list_markers = len(_LIST_MARKER.findall(cleaned))
    clock_times = len(_CLOCK_TIME.findall(cleaned))
    phrase, phrase_long = _longest_phrase(tokens)
    long_phrase = _longest_long_phrase(tokens)

    long_letters = [
        t for t in tokens if t.kind in ("word", "shape", "odd", "acronym") and len(t.raw) >= 4
    ]
    long_words = sum(t.kind == "word" for t in long_letters) + kinds["foreign"]
    word_share = long_words / (len(long_letters) + kinds["foreign"]) if long_words else 0.0
    letter_tokens = sum(
        kinds[k] for k in ("word", "foreign", "shape", "odd", "fragment", "acronym", "stray")
    )
    shaped = kinds["word"] + kinds["foreign"] + kinds["shape"]
    shape_language = (
        letter_tokens >= SHAPE_EVIDENCE_MIN_TOKENS and shaped / letter_tokens >= SHAPE_EVIDENCE_MIN
    )
    if word_share >= LEXICON_EVIDENCE_MIN or shape_language:
        kinds["word"] += kinds.pop("shape", 0)
    if shape_language:
        kinds.pop("fragment", None)

    content = sum(n for k, n in kinds.items() if k not in _UNCOUNTED)
    if not content:
        return QualityAssessment("keep", "no_content_tokens")
    readable = sum(kinds[k] for k in _READABLE)
    garbage = sum(kinds[k] for k in _GARBAGE) + ODD_TOKEN_WEIGHT * kinds["odd"]
    # Garbage other than lone capitals ("R L R", "F F L"): diagram labels.
    debris = garbage - sum(t.kind == "stray" and t.raw.isupper() for t in tokens)
    scored = readable + garbage
    s = QualityAssessment(
        "keep",
        "",
        content_tokens=content,
        readable_share=readable / scored if scored else 1.0,
        garbage_share=garbage / scored if scored else 0.0,
        notation_share=kinds["notation"] / content,
        long_words=long_words,
        word_share=word_share,
        letter_spaced_tokens=kinds["spaced"],
        numeric_cell_share=_numeric_cell_share(text),
    )

    protected_by = ""
    if phrase >= PROTECT_MIN_PHRASE or (
        phrase >= PROTECT_MIN_PHRASE - 1 and phrase_long >= PROTECT_MIN_LONG_PHRASE
    ):
        protected_by = "phrase"
    elif long_words >= PROTECT_MIN_WORDS and word_share >= PROTECT_MIN_WORD_SHARE:
        protected_by = "real_words"
    elif shape_language:
        protected_by = "language"
    elif contacts:
        protected_by = "contact_details"  # e-mail or phone: noise makes neither
    elif s.notation_share >= NOTATION_MIN_SHARE:
        protected_by = "training_notation"
    elif s.numeric_cell_share >= NUMERIC_TABLE_MIN_SHARE:
        protected_by = "numeric_table"
    elif s.letter_spaced_tokens:
        protected_by = "letter_spaced"
    elif measurements >= PROTECT_MIN_MEASUREMENTS or decimal_measurements:
        protected_by = "measurements"
    elif clock_times >= CLOCK_TIMES_MIN:
        protected_by = "times"
    elif long_words and debris <= 0:
        protected_by = "labels"  # "R L R Diagram 10-11": words and letter labels only
    elif quantified >= QUANTIFIED_MIN:
        protected_by = "quantities"
    elif list_markers >= NUMBERED_LIST_MIN and kinds["num"] >= NUMBERED_LIST_MIN:
        protected_by = "numbered_list"
    elif linked and kinds["word"] and content <= LINK_LABEL_MAX_TOKENS:
        protected_by = "link_with_label"  # "Hex Bar http://...": a reference
    elif _code_column(tokens) >= CODE_COLUMN_MIN:
        protected_by = "code_column"
    elif long_phrase >= PROTECT_MIN_LONG_PHRASE + 1 or (
        long_phrase >= PROTECT_MIN_LONG_PHRASE and word_share >= PROTECT_MIN_WORD_SHARE
    ):
        protected_by = "word_pair"
    elif len(_SESSION_CODE.findall(cleaned)) >= SESSION_CODES_MIN:
        protected_by = "coded_schedule"
    elif (
        _weekday_run(tokens) >= WEEKDAY_RUN_MIN
        or _weekday_run(tokens, line_initial=True) >= WEEKDAY_LINE_RUN_MIN
    ):
        protected_by = "weekly_schedule"
    s = replace(s, protected_by=protected_by)

    if not protected_by and (
        s.readable_share < DROP_MAX_READABLE_ANY
        or (
            s.readable_share < DROP_MAX_READABLE
            and (
                (garbage >= DROP_MIN_GARBAGE_TOKENS and word_share < PROTECT_MIN_WORD_SHARE)
                or not long_words
            )
        )
    ):
        return replace(s, verdict="drop", reason="unreadable")
    return replace(s, reason=protected_by or "readable")


def assess(text: str) -> Verdict:
    """``"keep"`` or ``"drop"`` for one chunk's text."""
    return assess_detail(text).verdict


__all__ = [
    "DOMAIN_WORDS",
    "SHORT_WORDS",
    "QualityAssessment",
    "Verdict",
    "assess",
    "assess_detail",
    "classify_token",
    "english_dictionary",
    "is_word",
    "normalise",
]
