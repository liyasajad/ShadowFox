import re
import math
import os
import sys
import time
import pickle
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autokey")



@dataclass
class Config:
    corpus_file: str = "corpus.txt"
    model_file:  str = "autokey_model.pkl"

    lambda1: float = 0.05   # unigram fallback
    lambda2: float = 0.20   # bigram
    lambda3: float = 0.75   # trigram (must sum to 1.0)

    unk_threshold: int = 1  # words seen ≤ this become <UNK>


    default_top_n: int = 8
    max_top_n:     int = 20

CFG = Config()



class BKTree:
    """
    BK-tree for O(log V) approximate nearest-word lookup.

    Exploits the triangle inequality of edit distance to prune entire
    subtrees, giving sub-linear average query time.
    Reference: Burkhard & Keller, 1973.
    """

    def __init__(self, words: list[str]):
        self._root: Optional[tuple] = None
        for w in words:
            self._add(w)

    def _add(self, word: str):
        if self._root is None:
            self._root = (word, {})
            return
        node = self._root
        while True:
            d = _edit(word, node[0])
            if d == 0:
                return
            children = node[1]
            if d not in children:
                children[d] = (word, {})
                return
            node = children[d]

    def search(self, query: str, max_dist: int) -> list[tuple[int, str]]:
        """Return all (distance, word) pairs within max_dist, sorted ascending."""
        if self._root is None:
            return []
        results: list[tuple[int, str]] = []
        stack = [self._root]
        while stack:
            word, children = stack.pop()
            d = _edit(query, word)
            if d <= max_dist:
                results.append((d, word))
            for k, child in children.items():
                if abs(d - k) <= max_dist:
                    stack.append(child)
        results.sort()
        return results


def _edit(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    if abs(m - n) > 2:
        return abs(m - n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


_SENT_END = re.compile(r'[.!?]+\s+')

_CONTRACTIONS = {
    "can't":"cannot", "won't":"will not", "don't":"do not",
    "doesn't":"does not", "didn't":"did not", "isn't":"is not",
    "aren't":"are not", "wasn't":"was not", "weren't":"were not",
    "i'm":"i am", "i've":"i have", "i'll":"i will", "i'd":"i would",
    "it's":"it is", "that's":"that is", "there's":"there is",
    "they're":"they are", "they've":"they have", "they'll":"they will",
    "we're":"we are", "we've":"we have", "we'll":"we will",
    "you're":"you are", "you've":"you have", "you'll":"you will",
    "he's":"he is", "she's":"she is", "let's":"let us",
    "what's":"what is", "who's":"who is", "how's":"how is",
}

_CONT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
    re.IGNORECASE,
)

def _expand_contractions(text: str) -> str:
    return _CONT_RE.sub(lambda m: _CONTRACTIONS[m.group().lower()], text)

def tokenize(text: str) -> list[str]:
    """
    Tokenise text:
    - Expands contractions  ("don't" → "do not")
    - Preserves numbers     ("2026", "3.14")
    - Normalises to lowercase
    - Inserts <S> sentence boundary markers
    - Strips lone punctuation tokens
    """
    text = text.lower()
    text = _expand_contractions(text)
    text = _SENT_END.sub(" <S> ", text)
    text = re.sub(r"[^a-z0-9\s<>_]", " ", text)
    tokens = [t for t in text.split() if t and t not in ("s",)]
    return tokens



UNK = "<UNK>"

class NGramModel:
    """
    Trigram language model with:
      • <UNK> handling for rare/unseen words
      • Cached frequency sums for O(1) probability denominators
      • Context-aware candidate pruning
      • BK-tree autocorrect (sub-linear edit-distance search)
      • Sentence-boundary-aware training
      • Pickle serialisation
    """

    def __init__(
        self,
        corpus: str,
        lambda1: float = CFG.lambda1,
        lambda2: float = CFG.lambda2,
        lambda3: float = CFG.lambda3,
        unk_threshold: int = CFG.unk_threshold,
    ):
        self.lambda1       = lambda1
        self.lambda2       = lambda2
        self.lambda3       = lambda3
        self.unk_threshold = unk_threshold

        self.unigram:  Counter     = Counter()
        self.bigram:   defaultdict = defaultdict(Counter)
        self.trigram:  defaultdict = defaultdict(Counter)

        self.unigram_total:  int  = 0
        self.bigram_totals:  dict = {}
        self.trigram_totals: dict = {}

        self.vocab:   set = set()
        self._V:      int = 0
        self._bktree: Optional[BKTree] = None

        self._build(corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenise inference text, mapping unseen words to <UNK>."""
        raw = tokenize(text)
        return [t if t in self.vocab else UNK for t in raw if t != "<S>"]

    def _build(self, corpus: str):
        """
        Two-pass build:
          Pass 1 — count raw frequencies to identify rare words.
          Pass 2 — replace rare words with <UNK>, build n-gram tables.
        """
        t0 = time.perf_counter()

        raw_tokens = tokenize(corpus)
        raw_counts: Counter = Counter(t for t in raw_tokens if t != "<S>")

        def unk_map(t: str) -> str:
            if t == "<S>":
                return "<S>"
            return t if raw_counts[t] > self.unk_threshold else UNK

        tokens = [unk_map(t) for t in raw_tokens]

        self.vocab = set(t for t in tokens if t != "<S>")
        self.vocab.add(UNK)
        self._V = len(self.vocab)

        for i, w in enumerate(tokens):
            if w == "<S>":
                continue
            self.unigram[w] += 1

            prev1 = tokens[i - 1] if i >= 1 else None
            prev2 = tokens[i - 2] if i >= 2 else None

            if prev1 and prev1 != "<S>":
                self.bigram[prev1][w] += 1
            if prev1 and prev2 and prev1 != "<S>" and prev2 != "<S>":
                self.trigram[(prev2, prev1)][w] += 1

        self.unigram_total = sum(self.unigram.values())
        self.bigram_totals = {
            ctx: sum(cnts.values()) for ctx, cnts in self.bigram.items()
        }
        self.trigram_totals = {
            ctx: sum(cnts.values()) for ctx, cnts in self.trigram.items()
        }

        real_words = [w for w in self.vocab if not w.startswith("<")]
        self._bktree = BKTree(real_words)

        elapsed = time.perf_counter() - t0
        log.info(
            "Model built in %.2fs | vocab=%d | tokens=%d | bigrams=%d | trigrams=%d",
            elapsed, self._V, self.unigram_total,
            len(self.bigram), len(self.trigram),
        )

    def _p1(self, w: str) -> float:
        """Unigram probability with Laplace smoothing."""
        return (self.unigram[w] + 1) / (self.unigram_total + self._V)

    def _p2(self, prev: str, w: str) -> float:
        """Bigram probability with Laplace smoothing."""
        denom = self.bigram_totals.get(prev, 0) + self._V
        return (self.bigram[prev][w] + 1) / denom

    def _p3(self, p2: str, p1: str, w: str) -> float:
        """Trigram probability with Laplace smoothing."""
        ctx   = (p2, p1)
        denom = self.trigram_totals.get(ctx, 0) + self._V
        return (self.trigram[ctx][w] + 1) / denom

    def interpolated(self, context: list[str], word: str) -> float:
        """
        Linear interpolation of trigram, bigram, and unigram probabilities.
        Maps unseen context words to <UNK> before lookup.
        """
        ctx = [c if c in self.vocab else UNK for c in context]
        u   = self._p1(word)
        b   = self._p2(ctx[-1], word)          if len(ctx) >= 1 else u
        t   = self._p3(ctx[-2], ctx[-1], word) if len(ctx) >= 2 else b
        return self.lambda3 * t + self.lambda2 * b + self.lambda1 * u

    def predict(self, text: str, top_n: int = CFG.default_top_n) -> list[dict]:
        """
        Predict next words or complete current prefix.

        Modes (auto-detected from trailing whitespace):
          • Prefix mode  — text ends mid-word → complete the partial word
          • Next-word    — text ends with space → predict next full word

        Next-word uses context-aware candidate pruning:
          trigram context available → candidates = words seen after that trigram
          bigram  context available → candidates = words seen after that bigram
          else                      → top-N unigram words (frequency fallback)
        """
        t0 = time.perf_counter()

        if text and not text[-1].isspace():
            raw = tokenize(text)
            if not raw:
                return []
            prefix  = raw[-1]
            context = [t for t in raw[:-1] if t != "<S>"]

            candidates = [
                w for w in self.vocab
                if w.startswith(prefix) and w != prefix and not w.startswith("<")
            ]
            if not candidates:
                return []

            def _score(w: str) -> float:
                base  = self.unigram.get(w, 0)
                boost = self.interpolated(context, w) * 200 if context else 0
                return base + boost

            ranked  = sorted(candidates, key=_score, reverse=True)[:top_n]
            max_s   = _score(ranked[0]) or 1.0
            elapsed = time.perf_counter() - t0
            log.debug("prefix predict '%.10s' → %d results in %.1fms",
                      prefix, len(ranked), elapsed * 1000)
            return [
                {"word": w, "prob": round(_score(w) / max_s, 6),
                 "pct": round(_score(w) / max_s * 100, 1)}
                for w in ranked
            ]

        raw     = tokenize(text)
        context = [t if t in self.vocab else UNK for t in raw]

        candidates: set | None = None

        if len(context) >= 2:
            ctx3 = (context[-2], context[-1])
            if ctx3 in self.trigram:
                candidates = set(self.trigram[ctx3].keys())

        if candidates is None and len(context) >= 1:
            ctx2 = context[-1]
            if ctx2 in self.bigram:
                candidates = set(self.bigram[ctx2].keys())

        if not candidates:
            results = [
                (w, self._p1(w))
                for w, _ in self.unigram.most_common(top_n * 3)
                if not w.startswith("<")
            ]
        else:
            results = [
                (w, self.interpolated(context, w))
                for w in candidates
                if not w.startswith("<")
            ]

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_n]

        if not results:
            return []

        max_p   = results[0][1]
        elapsed = time.perf_counter() - t0
        log.debug("next-word predict → %d candidates in %.1fms",
                  len(results), elapsed * 1000)
        return [
            {"word": w, "prob": round(p, 6), "pct": round(p / max_p * 100, 1)}
            for w, p in results
        ]

    def autocorrect(
        self,
        word: str,
        max_edit: int = 2,
        after_period: bool = False,
    ) -> str:
        """
        Correct spelling using a BK-tree.

        Capitalisation rules:
          after_period=True  → sentence start → capitalise result
          after_period=False → mid-sentence   → lowercase result
        """
        if not word:
            return word

        lower = word.lower()

        if lower in self.vocab:
            return lower.capitalize() if after_period else lower

        if self._bktree is None:
            return word

        hits = self._bktree.search(lower, max_edit)
        if not hits:
            return word

        best_dist = hits[0][0]
        best_word = max(
            (w for d, w in hits if d == best_dist),
            key=lambda w: self.unigram.get(w, 0),
        )

        return best_word.capitalize() if after_period else best_word

    def perplexity(self, text: str) -> float:
        """
        Compute interpolated trigram perplexity on text.
        Lower = model is less surprised by the text.
        Returns inf for very short inputs (< 2 tokens).
        """
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return float("inf")
        lp = sum(
            math.log(self.interpolated(tokens[max(0, i - 2):i], w) + 1e-12)
            for i, w in enumerate(tokens)
        )
        return round(math.exp(-lp / len(tokens)), 2)

    def stats(self) -> dict:
        """Return model statistics for the /api/stats endpoint."""
        return {
            "vocab_size":       self._V,
            "total_tokens":     self.unigram_total,
            "bigram_contexts":  len(self.bigram),
            "trigram_contexts": len(self.trigram),
        }

    def save_model(self, filepath: str):
        """Persist model to disk so rebuild is skipped on next run."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(filepath) / 1_000_000
        log.info("Model saved → %s  (%.1f MB)", filepath, size_mb)

    @staticmethod
    def load_model(filepath: str) -> "NGramModel":
        """Load a previously saved model from disk."""
        with open(filepath, "rb") as f:
            m = pickle.load(f)
        log.info("Model loaded ← %s", filepath)
        return m



def load_corpus() -> str:
    """Load corpus.txt. Exits with a clear message if file is missing."""
    path = os.path.join(os.path.dirname(__file__), CFG.corpus_file)
    if not os.path.exists(path):
        log.error("'%s' not found. Place your corpus file here: %s",
                  CFG.corpus_file, os.path.dirname(os.path.abspath(__file__)))
        sys.exit(1)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if len(text.strip()) < 100:
        log.error("'%s' is empty or too small.", CFG.corpus_file)
        sys.exit(1)
    log.info("Corpus loaded: %s  (%s chars)", path, f"{len(text):,}")
    return text



app   = Flask(__name__, static_folder="static")
model: Optional[NGramModel] = None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST { "text": str, "top_n": int }
    → { "predictions": [{"word", "prob", "pct"}, …] }
    """
    try:
        data  = request.get_json(force=True, silent=True) or {}
        text  = str(data.get("text", ""))
        top_n = min(int(data.get("top_n", CFG.default_top_n)), CFG.max_top_n)
        preds = model.predict(text, top_n=top_n)
        return jsonify({"predictions": preds})
    except Exception as e:
        log.exception("predict error")
        return jsonify({"predictions": [], "error": str(e)}), 500


@app.route("/api/autocorrect", methods=["POST"])
def api_autocorrect():
    """
    POST { "word": str, "after_period": bool }
    → { "original", "corrected", "changed" }
    """
    try:
        data         = request.get_json(force=True, silent=True) or {}
        word         = str(data.get("word", ""))
        after_period = bool(data.get("after_period", False))
        corrected    = model.autocorrect(word, after_period=after_period)
        return jsonify({
            "original":  word,
            "corrected": corrected,
            "changed":   corrected != word,
        })
    except Exception as e:
        log.exception("autocorrect error")
        return jsonify({"original": word, "corrected": word,
                        "changed": False, "error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """GET → model statistics."""
    try:
        return jsonify(model.stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/perplexity", methods=["POST"])
def api_perplexity():
    """
    POST { "text": str }
    → { "perplexity": float }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("text", ""))
        return jsonify({"perplexity": model.perplexity(text)})
    except Exception as e:
        return jsonify({"perplexity": 0.0, "error": str(e)}), 500



def _get_model() -> NGramModel:
    """
    Load cached model from disk if available and corpus hasn't changed
    (checked via file modification time). Otherwise rebuild and save.
    """
    model_path  = os.path.join(os.path.dirname(__file__), CFG.model_file)
    corpus_path = os.path.join(os.path.dirname(__file__), CFG.corpus_file)

    corpus_mtime = os.path.getmtime(corpus_path)

    if os.path.exists(model_path):
        model_mtime = os.path.getmtime(model_path)
        if model_mtime >= corpus_mtime:
            log.info("Loading cached model (corpus unchanged)…")
            try:
                return NGramModel.load_model(model_path)
            except Exception:
                log.warning("Cached model load failed — rebuilding.")

    log.info("Building model from corpus…")
    corpus = load_corpus()
    m      = NGramModel(corpus)
    m.save_model(model_path)
    return m


if __name__ == "__main__":
    model = _get_model()
    st    = model.stats()
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║         AutoKey  —  N-gram KB        ║")
    print("  ╠══════════════════════════════════════╣")
    print(f"  ║  Vocab:    {st['vocab_size']:>10,} words          ║")
    print(f"  ║  Tokens:   {st['total_tokens']:>10,}              ║")
    print(f"  ║  Bigrams:  {st['bigram_contexts']:>10,} contexts      ║")
    print(f"  ║  Trigrams: {st['trigram_contexts']:>10,} contexts      ║")
    print("  ╠══════════════════════════════════════╣")
    print("  ║  Open:  http://localhost:5000        ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    app.run(debug=False, port=5000)