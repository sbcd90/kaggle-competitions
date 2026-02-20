import re
import unicodedata
import pandas as pd

SUBSCRIPT_MAP = str.maketrans({
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "ₓ":"x",
})

VOWEL_MAP = {
    # a
    "a2": "á", "aₐ": "á", "a₂": "á",
    "a3": "à", "a₃": "à",

    # e
    "e2": "é", "e₂": "é",
    "e3": "è", "e₃": "è",

    # i
    "i2": "í", "i₂": "í",
    "i3": "ì", "i₃": "ì",

    # u
    "u2": "ú", "u₂": "ú",
    "u3": "ù", "u₃": "ù",
}

KNOWN_DETS = {
    # core
    "d", "mul", "ki", "lu2", "e2", "uru", "kur",

    # gender
    "mi", "m",

    # materials / objects
    "geš", "ĝeš", "tug2", "dub", "id2", "mušen", "na4", "kuš", "u2",
}

DET_ALIAS = {
    "lu₂": "lu2",
    "e₂": "e2",
    "tug₂": "tug2",
    "id₂": "id2",
    "na₄": "na4",
    "u₂": "u2",
    "ĝeš": "geš",
}

def load_and_generate_akkadian_text_file(akkadian_text_file: str, lexicon):
    final_texts = []
    df = pd.read_csv(akkadian_text_file)
    df = df[["transliteration"]]
    for row in df["transliteration"]:
        cleaned_text = clean_akkadian_translit(row)
        final = normalize_with_lexicon(cleaned_text, lexicon)
        final_texts.append(final)
    with open("../data/akkadian_output.txt", "w", encoding="utf-8") as f:
        for line in final_texts:
            f.write(line.strip() + "\n")

def load_and_generate_english_text_file(english_text_file: str):
    final_texts = []
    df = pd.read_csv(english_text_file)
    for row in df["translation"]:
        final_texts.append(row)
    with open("../data/english_output.txt", "w", encoding="utf-8") as f:
        for line in final_texts:
            f.write(line.strip() + "\n")

def load_lexicon(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["form", "norm"])
    mapping = dict(zip(df["form"].astype(str).str.strip(),
                       df["norm"].astype(str).str.strip()))
    return mapping

def normalize_with_lexicon(transliteration: str, lexicon: dict) -> str:
    if transliteration is None:
        return ""

    tokens = transliteration.split()
    out = []

    for token in tokens:
        if token in lexicon:
            out.append(lexicon[token])
            continue

        token_stripped = re.sub(r"\{[^}]+\}", "", token)

        if token_stripped in lexicon:
            norm = lexicon[token_stripped]
            dets = "".join(re.findall(r"\{[^}]+\}", token))
            out.append(dets + norm if dets else norm)
            continue
        out.append(token)
    return " ".join(out)

def clean_akkadian_translit(text: str) -> str:
    if text is None:
        return ""

    s = str(text)

    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")

    s = s.replace("Ḫ", "H").replace("ḫ", "h")

    s = re.sub(r"\bsz\b", "š", s)
    s = re.sub(r"\bSZ\b", "Š", s)

    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")

    s = s.translate(SUBSCRIPT_MAP)

    for src, tgt in VOWEL_MAP.items():
        s = re.sub(rf"(?<!\d){re.escape(src)}(?!\d)", tgt, s)

    def det_paren_to_curly(m):
        det = m.group(1)
        det = det.translate(SUBSCRIPT_MAP)
        det = DET_ALIAS.get(det, det)
        return "{" + det + "}"

    s = re.sub(r"\((d|mul|ki|lu2|lu₂|e2|e₂|uru|kur|mi|m|geš|ĝeš|tug2|dub|id2|mušen|na4|kuš|u2)\)",
               det_paren_to_curly, s)

    s = re.sub(r"…+", " <big_gap> ", s)
    s = re.sub(r"\[\s*[xX]\s*\]", " <gap> ", s)

    def bracket_handler(m):
        inside = m.group(1).strip()

        if inside == "":
            return " <big_gap> "

        if re.fullmatch(r"[xX.\s]+", inside):
            return " <big_gap> "

        return f" {inside} "
    s = re.sub(r"\[([^\]]*)\]", bracket_handler, s)

    s = s.replace("˹", "").replace("˺", "")
    s = s.replace("!", "").replace("?", "")
    s = s.replace("/", "")

    s = s.replace("<", "").replace(">", "")
    s = s.replace(":", " ")

    def starts_with_capital(token: str) -> bool:
        for ch in token:
            if ch.isalpha():
                return ch.isupper()
        return False

    def strip_determinatives(token: str) -> str:
        return re.sub(r"\{[^}]+\}", "", token)

    def is_all_caps_ignoring_dets(token: str) -> bool:
        token = strip_determinatives(token)
        letters = [ch for ch in token if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters)

    def dot_handler(m):
        left = m.group(1)
        right = m.group(2)

        if left.isdigit() and right.isdigit():
            return left + "." + right
        elif starts_with_capital(left):
            return left + "." + right
        elif is_all_caps_ignoring_dets(left) or is_all_caps_ignoring_dets(right):
            return left + "." + right

        return left + " " + right

    s = re.sub(r"(\S+)\.(\S+)", dot_handler, s)

    #s = re.sub(r"(?<!\s)(\{[^}]+\})", r" \1", s)

    s = re.sub(r"\bki\b", "{ki}", s, flags=re.IGNORECASE)
    s = s.replace("{lu₂}", "{lu2}").replace("{e₂}", "{e2}")

    s = re.sub(r"\s+", " ", s).strip()

    return s

def main():
    akkadian_csv_file = "../data/train.csv"
    lexicon = load_lexicon("../data/OA_Lexicon_eBL.csv")
    load_and_generate_akkadian_text_file(akkadian_csv_file, lexicon)
    load_and_generate_english_text_file(akkadian_csv_file)
    # text = "KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-(d)IM KIŠIB šu-(d)EN.LÍL DUMU ma-nu-ki-a-šur KIŠIB MAN-a-šur DUMU a-ta-a 0.33333 ma-na 2 GÍN KÙ.BABBAR SIG₅ i-ṣé-er PUZUR₄-a-šur DUMU a-ta-a a-lá-ḫu-um i-šu iš-tù ḫa-muš-tim ša ì-lí-dan ITU.KAM ša ke-na-tim li-mu-um e-na-sú-in a-na ITU 14 ḫa-am-ša-tim i-ša-qal šu-ma lá iš-qú-ul 1.5 GÍN.TA a-na 1 ma-na-im i-na ITU.1.KAM ṣí-ib-tám ú-ṣa-áb"
    # cleaned_text = clean_akkadian_translit(text)
    # final = normalize_with_lexicon(cleaned_text, lexicon)
    # print(final)

if __name__ == "__main__":
    main()

