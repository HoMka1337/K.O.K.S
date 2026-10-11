
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


STOP_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
    "как", "а", "то", "все", "она", "так", "его", "но",
    "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня",
    "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя",
    "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем",
    "была", "сам", "чтоб", "без", "будто", "чего",
    "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один",
    "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно",
    "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти",
    "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть",
    "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между",
}

STOP_EN = {
    "the", "a", "an", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "over", "after", "is",
    "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may",
    "might", "must", "can", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "when", "where",
    "why", "how",
}

STOPWORDS = STOP_RU | STOP_EN


SYNONYMS: Dict[str, Set[str]] = {
    "ии": {
        "ai", "искусственный", "интеллект",
        "artificial", "intelligence"
    },
    "ai": {
        "ии", "искусственный", "интеллект",
        "artificial", "intelligence"
    },
    "интеллект": {
        "ии", "ai", "разум", "ум"
    },
    "разум": {
        "интеллект", "ум", "сознание"
    },
    "нейросеть": {
        "нейросети", "нейронная", "сеть",
        "neural", "network"
    },
    "нейросети": {
        "нейросеть", "нейронная", "сеть",
        "neural", "network"
    },
    "neural": {
        "нейросеть", "нейросети", "нейронная"
    },
    "network": {
        "нейросеть", "нейросети", "сеть"
    },
    "мл": {
        "ml", "машинное", "обучение",
        "machine", "learning"
    },
    "ml": {
        "мл", "машинное", "обучение",
        "machine", "learning"
    },
    "machine": {
        "машинное", "обучение"
    },
    "learning": {
        "обучение", "изучение"
    },
    "python": {
        "питон", "пайтон"
    },
    "питон": {
        "python", "пайтон"
    },
    "пайтон": {
        "python", "питон"
    },
    "программирование": {
        "код", "кодинг", "разработка",
        "coding", "programming"
    },
    "код": {
        "программирование", "кодинг", "code"
    },
    "code": {
        "код", "программирование"
    },
    "transformer": {
        "трансформер", "трансформеры", "attention"
    },
    "трансформер": {
        "transformer", "трансформеры"
    },
    "attention": {
        "внимание", "transformer"
    },
    "внимание": {
        "attention", "внимательность"
    },
    "данные": {
        "data", "информация", "сведения"
    },
    "data": {
        "данные", "информация"
    },
    "информация": {
        "данные", "сведения", "info"
    },
    "компьютер": {
        "пк", "pc", "машина", "computer"
    },
    "пк": {
        "компьютер", "pc"
    },
    "pc": {
        "компьютер", "пк"
    },
    "модель": {
        "model", "модели", "моделька"
    },
    "model": {
        "модель"
    },
    "обучение": {
        "тренировка", "training", "учиться"
    },
    "training": {
        "обучение", "тренировка"
    },
    "сеть": {
        "network", "сети", "интернет"
    },
    "интернет": {
        "сеть", "internet", "web", "веб"
    },
    "internet": {
        "интернет", "сеть"
    },
    "файл": {
        "file", "документ", "док"
    },
    "file": {
        "файл", "документ"
    },
    "поиск": {
        "search", "искать", "найти"
    },
    "search": {
        "поиск"
    },
    "ответ": {
        "answer", "реакция", "результат"
    },
    "answer": {
        "ответ"
    },
    "вопрос": {
        "question", "запрос"
    },
    "question": {
        "вопрос"
    },
    "ошибка": {
        "error", "баг", "bug", "сбой", "проблема"
    },
    "error": {
        "ошибка", "баг", "bug"
    },
    "bug": {
        "ошибка", "баг", "error"
    },
    "баг": {
        "ошибка", "bug", "error"
    },
    "время": {
        "time", "час"
    },
    "time": {
        "время", "час"
    },
    "час": {
        "время", "time"
    },
}


QUESTION_TYPES = {
    "definition": (
        r"^что\s+такое\b",
        r"^что\s+значит\b",
        r"^дай\s+определение\b",
        r"^определи\b",
        r"^объясни\s+что\s+такое\b",
        r"\bчто\s+такое\b",
    ),
    "howto": (
        r"^как\s+",
        r"^каким\s+образом\b",
        r"^как\s+сделать\b",
        r"^как\s+научиться\b",
        r"^как\s+работает\b",
        r"^как\s+устроен\b",
    ),
    "why": (
        r"^почему\b",
        r"^зачем\b",
        r"^отчего\b",
        r"^из-за\s+чего\b",
    ),
    "who": (
        r"^кто\s+",
        r"^кто\s+такой\b",
        r"^кто\s+такая\b",
    ),
    "where": (
        r"^где\b",
        r"^куда\b",
    ),
    "when": (
        r"^когда\b",
        r"^во\s+сколько\b",
    ),
    "list": (
        r"^перечисли\b",
        r"^список\b",
        r"^какие\b",
        r"^назови\b",
    ),
    "yesno": (
        r"^(можно|нельзя|правда|верно|существует|есть)\b",
    ),
}


_RU_SUFFIXES = sorted({
    "иями", "ями", "ами", "ией", "иях", "ях",
    "ах", "ов", "ев", "ий", "ый", "ой",
    "ая", "яя", "ое", "ее", "ые", "ие",
    "ым", "им", "ем", "ом", "ей", "ую",
    "юю", "ать", "ять", "еть", "ить",
    "уть", "ешь", "ишь", "ете", "ите",
    "ут", "ют", "ат", "ят", "ла", "ло",
    "ли", "ть", "ся", "сь", "ия", "ии",
    "ью", "ья", "ье", "ам", "у", "ю",
    "а", "я", "о", "е", "ы", "и",
    "ь", "й",
}, key=len, reverse=True)

_EN_SUFFIXES = sorted({
    "ations", "ation", "ings", "ing",
    "ies", "ied", "ion", "ions",
    "ed", "es", "s", "ly",
    "ness", "ment", "able", "ible",
}, key=len, reverse=True)

_TOKEN_RE = re.compile(
    r"[А-Яа-яЁёA-Za-z0-9]{2,}",
    re.UNICODE
)

_WHITESPACE_RE = re.compile(r"\s+")

BAD_ANSWER_MARKERS = (
    "перейти к содержанию",
    "главное меню",
    "служебные страницы",
    "пожертвовать создать учётную запись",
    "полный индекс категории",
    "узнать инн",
    "требуется согласие на обработку",
    "welcome to python.org notice",
    "яндекс яндекс",
    "перейти к содержимому",
    "создать учётную запись войти",
    "содержание переместить в боковую панель",
)


def detect_question_type(text: str) -> str:
    if not isinstance(text, str):
        return "general"

    normalized = _WHITESPACE_RE.sub(
        " ",
        text.casefold().replace("ё", "е")
    ).strip()

    for question_type, patterns in QUESTION_TYPES.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                return question_type

    return "general"


def stem(word: str) -> str:
    if not isinstance(word, str):
        return ""

    normalized = word.casefold().replace("ё", "е")

    if len(normalized) <= 3:
        return normalized

    if re.fullmatch(r"[a-z]+", normalized):
        for suffix in _EN_SUFFIXES:
            if (
                normalized.endswith(suffix)
                and len(normalized) - len(suffix) >= 3
            ):
                return normalized[:-len(suffix)]

        return normalized

    if re.fullmatch(r"[а-я]+", normalized):
        for suffix in _RU_SUFFIXES:
            if (
                normalized.endswith(suffix)
                and len(normalized) - len(suffix) >= 3
            ):
                return normalized[:-len(suffix)]

    return normalized


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    normalized = text.casefold().replace("ё", "е")

    result = []

    for raw in _TOKEN_RE.findall(normalized):
        if raw in STOPWORDS:
            continue

        token = stem(raw)

        if token:
            result.append(token)

    return result


def expand_synonyms(tokens: List[str]) -> Set[str]:
    expanded = set()

    for token in tokens:
        if not isinstance(token, str):
            continue

        normalized = token.casefold().replace("ё", "е")
        base = stem(normalized)

        if not base:
            continue

        expanded.add(base)

        variants = SYNONYMS.get(normalized, set())

        if not variants:
            variants = SYNONYMS.get(base, set())

        for variant in variants:
            normalized_variant = variant.casefold().replace(
                "ё", "е"
            )

            if normalized_variant in STOPWORDS:
                continue

            variant_stem = stem(normalized_variant)

            if variant_stem:
                expanded.add(variant_stem)

    return expanded


def keywords(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()

    original = tokenize(text)

    if not original:
        return set()

    return expand_synonyms(original)


def _term_frequencies(tokens: List[str]) -> Counter:
    return Counter(tokens)


class SemanticIndex:
    def __init__(self, docs: List[Tuple[str, str]]):
        self.docs = list(docs) if docs is not None else []
        self.doc_tokens: List[Set[str]] = []
        self.df: Counter = Counter()

        for _, text in self.docs:
            tokens = set(tokenize(text))
            self.doc_tokens.append(tokens)
            self.df.update(tokens)

        self.n_docs = max(1, len(self.docs))

    def _idf(self, term: str) -> float:
        frequency = self.df.get(term, 0)

        return (
            math.log(
                (self.n_docs + 1.0) /
                (frequency + 1.0)
            )
            + 1.0
        )

    def score(self, query: str, doc_index: int) -> float:
        if not isinstance(query, str):
            return 0.0

        if not 0 <= doc_index < len(self.doc_tokens):
            return 0.0

        query_tokens = set(tokenize(query))

        if not query_tokens:
            return 0.0

        document_tokens = self.doc_tokens[doc_index]

        if not document_tokens:
            return 0.0

        expanded = expand_synonyms(
            list(query_tokens)
        )

        direct_matches = (
            query_tokens & document_tokens
        )

        synonym_matches = (
            (expanded - query_tokens) & document_tokens
        )

        if not direct_matches and not synonym_matches:
            return 0.0

        direct_score = sum(
            self._idf(token)
            for token in direct_matches
        )

        synonym_score = sum(
            self._idf(token) * 0.35
            for token in synonym_matches
        )

        query_coverage = (
            len(direct_matches)
            + min(
                len(synonym_matches) * 0.35,
                len(query_tokens)
            )
        ) / len(query_tokens)

        normalization = math.sqrt(
            len(query_tokens)
        )

        score = (
            direct_score + synonym_score
        ) / normalization

        score *= (
            0.5 + 0.5 * min(1.0, query_coverage)
        )

        if len(direct_matches) == len(query_tokens):
            score *= 1.15

        return max(0.0, score)

    def best(
        self,
        query: str,
        min_score: float = 0.5
    ) -> Optional[Tuple[float, int]]:
        if not isinstance(query, str):
            return None

        if not query.strip():
            return None

        best_score = 0.0
        best_index = -1

        for index in range(len(self.docs)):
            score = self.score(
                query,
                index
            )

            if score > best_score:
                best_score = score
                best_index = index

        if best_index < 0:
            return None

        if best_score < min_score:
            return None

        return best_score, best_index


def answer_looks_meaningful(
    answer: str,
    min_letters: int = 40
) -> bool:
    if not isinstance(answer, str):
        return False

    answer = answer.strip()

    if not answer:
        return False

    lowered = answer.casefold()

    if any(
        marker in lowered
        for marker in BAD_ANSWER_MARKERS
    ):
        return False

    if re.fullmatch(
        r"[\W_]+",
        answer,
        flags=re.UNICODE
    ):
        return False

    letters = sum(
        character.isalpha()
        for character in answer
    )

    if letters < max(0, int(min_letters)):
        return False

    if len(answer) < 20:
        return False

    words = _TOKEN_RE.findall(answer)

    if len(words) < 3:
        return False

    return True


def clean_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return ""

    text = answer.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if not text:
        return ""

    if len(text) <= 500:
        return text

    normalized = _WHITESPACE_RE.sub(
        " ",
        text
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized
    )

    selected = []
    length = 0

    for sentence in sentences:
        if not sentence:
            continue

        if length + len(sentence) > 600:
            break

        selected.append(sentence)
        length += len(sentence) + 1

        if len(selected) >= 3:
            break

    if selected:
        return " ".join(selected).strip()

    return normalized[:600].rstrip()
