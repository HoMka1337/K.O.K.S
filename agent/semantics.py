import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


STOP_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни",
    "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж",
    "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может",
    "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего",
    "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто",
    "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее",
    "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно",
    "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них",
    "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть",
    "том", "нельзя", "такой", "им", "более", "всегда", "конечно",
    "всю", "между",
}

STOP_EN = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "over",
    "after", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "when", "where", "why", "how",
}

STOPWORDS = STOP_RU | STOP_EN


SYNONYMS: Dict[str, Set[str]] = {
    "ии": {"искусственный", "интеллект", "ai", "artificial", "intelligence"},
    "ai": {"ии", "искусственный", "интеллект", "artificial", "intelligence"},
    "интеллект": {"ии", "ai", "разум", "ум"},
    "разум": {"интеллект", "ум", "сознание"},
    "нейросеть": {"нейронная", "сеть", "neural", "network", "нейросети"},
    "нейросети": {"нейросеть", "нейронная", "сеть", "neural", "network"},
    "neural": {"нейросеть", "нейронная", "сеть"},
    "network": {"сеть", "нейросеть"},
    "мл": {"машинное", "обучение", "ml", "machine", "learning"},
    "ml": {"мл", "машинное", "обучение", "machine", "learning"},
    "machine": {"машинное", "обучение"},
    "learning": {"обучение", "изучение"},
    "python": {"питон", "пайтон"},
    "питон": {"python"},
    "пайтон": {"python"},
    "программирование": {"код", "кодинг", "разработка", "coding", "programming"},
    "код": {"программирование", "кодинг", "code"},
    "code": {"код", "программирование"},
    "transformer": {"трансформер", "трансформеры", "attention"},
    "трансформер": {"transformer", "трансформеры"},
    "attention": {"внимание"},
    "внимание": {"attention", "внимательность"},
    "данные": {"data", "информация", "сведения"},
    "data": {"данные", "информация"},
    "информация": {"данные", "сведения", "info"},
    "компьютер": {"пк", "pc", "машина", "computer"},
    "пк": {"компьютер", "pc"},
    "pc": {"компьютер", "пк"},
    "модель": {"model", "модели", "моделька"},
    "model": {"модель"},
    "обучение": {"тренировка", "training", "учиться"},
    "training": {"обучение", "тренировка"},
    "сеть": {"network", "сети", "интернет"},
    "интернет": {"сеть", "internet", "web", "веб"},
    "internet": {"интернет", "сеть"},
    "файл": {"file", "документ", "док"},
    "file": {"файл", "документ"},
    "поиск": {"search", "искать", "найти"},
    "search": {"поиск"},
    "ответ": {"answer", "реакция", "результат"},
    "answer": {"ответ"},
    "вопрос": {"question", "запрос"},
    "question": {"вопрос"},
    "ошибка": {"error", "баг", "bug", "сбой", "проблема"},
    "error": {"ошибка", "баг"},
    "bug": {"ошибка", "баг"},
    "баг": {"ошибка", "bug"},
    "время": {"time", "час"},
    "time": {"время"},
    "час": {"время", "time"},
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


def detect_question_type(text: str) -> str:
    t = text.strip().lower()
    for qtype, patterns in QUESTION_TYPES.items():
        for p in patterns:
            if re.search(p, t):
                return qtype
    return "general"


_RU_SUFFIXES = sorted([
    "иями", "ями", "ами", "ией", "иях", "ях", "ах", "ов", "ев", "ий",
    "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ым", "им", "ем",
    "ом", "ой", "ей", "ую", "юю", "ать", "ять", "еть", "ить", "уть",
    "ешь", "ишь", "ете", "ите", "ут", "ют", "ат", "ят", "ла", "ло",
    "ли", "ть", "ся", "сь", "ий", "ия", "ие", "ии", "ью", "ья", "ье",
    "ов", "ам", "ах", "у", "ю", "а", "я", "о", "е", "ы", "и",
    "ь", "й",
], key=len, reverse=True)

_EN_SUFFIXES = sorted([
    "ations", "ation", "ings", "ing", "ies", "ied", "ion", "ions",
    "ed", "es", "s", "ly", "ness", "ment", "able", "ible",
], key=len, reverse=True)


def stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 3:
        return w
    if re.fullmatch(r"[a-z]+", w):
        for suf in _EN_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w
    for suf in _RU_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]{2,}", re.UNICODE)


def tokenize(text: str) -> List[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        tokens.append(stem(raw))
    return tokens


def expand_synonyms(tokens: List[str]) -> Set[str]:
    expanded: Set[str] = set(tokens)
    for t in tokens:
        syns = SYNONYMS.get(t)
        if syns:
            for s in syns:
                if s not in STOPWORDS:
                    expanded.add(stem(s))
    return expanded


def keywords(text: str) -> Set[str]:
    return expand_synonyms(tokenize(text))


def _term_frequencies(tokens: List[str]) -> Counter:
    return Counter(tokens)


class SemanticIndex:
    def __init__(self, docs: List[Tuple[str, str]]):
        self.docs = docs
        self.doc_tokens: List[Set[str]] = []
        self.df: Counter = Counter()
        for _, text in docs:
            toks = set(tokenize(text))
            self.doc_tokens.append(toks)
            for t in toks:
                self.df[t] += 1
        self.n_docs = max(1, len(docs))

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1.0

    def score(self, query: str, doc_index: int) -> float:
        q_tokens = keywords(query)
        if not q_tokens:
            return 0.0
        doc_tokens = self.doc_tokens[doc_index]
        if not doc_tokens:
            return 0.0
        common = q_tokens & doc_tokens
        if not common:
            return 0.0
        score = sum(self._idf(t) for t in common)
        score = score / (len(q_tokens) ** 0.5)
        return score

    def best(self, query: str, min_score: float = 0.5) -> Optional[Tuple[float, int]]:
        best_score = 0.0
        best_idx = -1
        for i in range(len(self.docs)):
            s = self.score(query, i)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0 or best_score < min_score:
            return None
        return best_score, best_idx


BAD_ANSWER_MARKERS = (
    "перейти к содержанию", "главное меню", "служебные страницы",
    "пожертвовать создать учётную запись", "полный индекс категории",
    "узнать инн", "требуется согласие на обработку",
    "welcome to python.org notice", "яндекс яндекс",
    "перейти к содержимому", "создать учётную запись войти",
    "содержание переместить в боковую панель",
)


def answer_looks_meaningful(answer: str, min_letters: int = 40) -> bool:
    if not answer or len(answer) < 20:
        return False
    low = answer.lower()
    if any(m in low for m in BAD_ANSWER_MARKERS):
        return False
    letters = sum(1 for c in answer if c.isalpha())
    if letters < min_letters:
        return False
    return True


def clean_answer(answer: str) -> str:
    text = re.sub(r"\s+", " ", answer).strip()
    if len(text) <= 500:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3]).strip()[:600]