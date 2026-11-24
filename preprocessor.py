import re
import html
import os
from typing import List, Dict, Any, Optional
import nltk

_LOCAL_NLTK_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "nltk_data"))
if _LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

CONTRACTIONS_DICT = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "u": "you",
    "ur": "your",
    "r": "are",
    "thx": "thanks",
    "pls": "please",
    "plz": "please",
    "idk": "i do not know",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "omg": "oh my god",
    "btw": "by the way",
    "w/": "with",
    "w/o": "without",
}

EMOJI_SENTIMENT_MAP = {
    "😍": " love adored ",
    "❤️": " love heart ",
    "💕": " love hearts ",
    "💖": " love sparkle ",
    "😊": " happy smile ",
    "😀": " happy grin ",
    "😃": " joyful grin ",
    "😄": " happy laughter ",
    "😁": " beaming smile ",
    "🎉": " celebration party ",
    "🥳": " celebrating thrilled ",
    "👏": " applause bravo ",
    "🔥": " awesome fire great ",
    "🚀": " rocket thriving ",
    "⭐": " star excellent ",
    "🌟": " glowing wonderful ",
    "✨": " sparkles brilliant ",
    "👍": " thumbs up approved ",
    "💯": " perfect hundred ",
    "🙌": " praise celebration ",
    "💪": " strong resilient ",
    "🥰": " loving affectionate ",
    "🤩": " starstruck amazed ",
    "😡": " angry furious ",
    "🤬": " rage cursing ",
    "😠": " annoyed angry ",
    "👿": " evil angry ",
    "🤮": " disgusted vomiting ",
    "🤢": " nauseated sick ",
    "👎": " thumbs down bad ",
    "😭": " crying tears weeping ",
    "😢": " sad crying ",
    "😞": " disappointed sad ",
    "😔": " sorrowful regret ",
    "💔": " broken heart devastated ",
    "😤": " frustrated upset ",
    "💩": " terrible awful crap ",
    "❌": " rejected bad failed ",
    "🚫": " banned forbidden ",
    "🤕": " hurt pain sick ",
    "😩": " exhausted distressed ",
    "😫": " tired miserable ",
    "😱": " shocked horrified ",
    "🥱": " bored tired ",
}

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@[\w_]+")
HASHTAG_REGEX = re.compile(r"#(\w+)")
REPEATED_CHARS_REGEX = re.compile(r"(.)\1{2,}")
NUMBERS_REGEX = re.compile(r"\b\d+\b")
WHITESPACE_REGEX = re.compile(r"\s+")
EMOJI_REGEX = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]"
)


class TweetPreprocessor:
    def __init__(self, spacy_model: Optional[str] = "en_core_web_sm"):
        self.tweet_tokenizer = TweetTokenizer(
            preserve_case=True, reduce_len=True, strip_handles=False
        )
        try:
            self.stop_words = set(stopwords.words("english"))
            self.sentiment_stopwords_to_keep = {
                "not", "no", "nor", "neither", "never", "none", "very", "too",
                "but", "against", "cannot", "couldn't", "didn't", "doesn't",
                "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't",
                "won't", "wouldn't"
            }
            self.stop_words = self.stop_words - self.sentiment_stopwords_to_keep
        except Exception:
            self.stop_words = set()

        self.nlp = None
        if spacy_model:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = None

    def expand_contractions(self, text: str) -> str:
        words = text.split()
        expanded_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in CONTRACTIONS_DICT:
                replacement = CONTRACTIONS_DICT[lower_word]
                if word.istitle():
                    replacement = replacement.capitalize()
                elif word.isupper():
                    replacement = replacement.upper()
                expanded_words.append(replacement)
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)

    def extract_metadata(self, raw_text: str) -> Dict[str, Any]:
        mentions = MENTION_REGEX.findall(raw_text)
        hashtags = HASHTAG_REGEX.findall(raw_text)
        urls = URL_REGEX.findall(raw_text)
        emojis = EMOJI_REGEX.findall(raw_text)

        return {
            "mentions": mentions,
            "hashtags": hashtags,
            "urls": urls,
            "emojis": emojis,
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()),
        }

    def clean_for_vader(self, text: str) -> str:
        if not text:
            return ""

        text = html.unescape(text)
        text = URL_REGEX.sub("", text)
        text = MENTION_REGEX.sub("", text)
        text = HASHTAG_REGEX.sub(r"\1", text)

        for emoji_char, replacement in EMOJI_SENTIMENT_MAP.items():
            if emoji_char in text:
                text = text.replace(emoji_char, replacement)

        text = WHITESPACE_REGEX.sub(" ", text).strip()
        return text

    def clean_for_ml(
        self,
        text: str,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ) -> str:
        if not text:
            return ""

        text = html.unescape(text)
        text = URL_REGEX.sub("", text)
        text = MENTION_REGEX.sub("", text)
        text = HASHTAG_REGEX.sub(r"\1", text)

        for emoji_char, replacement in EMOJI_SENTIMENT_MAP.items():
            if emoji_char in text:
                text = text.replace(emoji_char, replacement)

        text = self.expand_contractions(text)
        text = REPEATED_CHARS_REGEX.sub(r"\1\1", text)

        if lemmatize and self.nlp:
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                lemma = token.lemma_.lower().strip()
                if not lemma:
                    continue
                if remove_stopwords and lemma in self.stop_words:
                    continue
                tokens.append(lemma)
            return " ".join(tokens)

        tokens = self.tweet_tokenizer.tokenize(text.lower())
        filtered = []
        for token in tokens:
            clean_tok = re.sub(r"[^\w\s]", "", token)
            if not clean_tok or clean_tok.isnumeric():
                continue
            if remove_stopwords and clean_tok in self.stop_words:
                continue
            filtered.append(clean_tok)

        return " ".join(filtered)

    def spacy_linguistic_features(self, text: str) -> List[Dict[str, str]]:
        if not self.nlp:
            return []
        doc = self.nlp(text)
        features = []
        for token in doc:
            features.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_stop": str(token.is_stop),
            })
        return features
