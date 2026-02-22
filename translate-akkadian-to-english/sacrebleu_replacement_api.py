import math
from collections import Counter

class ScoreResult:
    def __init__(self, score):
        self.score = score

def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def get_char_ngrams(text, n):
    return Counter(text[i:i+n] for i in range(len(text)-n+1))

def corpus_bleu(hyps, refs_list, max_n=4):
    refs = refs_list[0]

    total_clipped = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hyps, refs):
        hyp_tokens = hyp.strip().split()
        ref_tokens = ref.strip().split()

        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n+1):
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            total_counts[n-1] += sum(hyp_ngrams.values())

            for ng in hyp_ngrams:
                total_clipped[n-1] += min(
                    hyp_ngrams[ng],
                    ref_ngrams.get(ng, 0),
                )

    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(0)
        else:
            precisions.append(total_clipped[i] / total_counts[i])

    if min(precisions) == 0:
        return ScoreResult(0.0)

    log_prec = sum((1 / max_n) * math.log(p) for p in precisions)
    if hyp_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    bleu = bp * math.exp(log_prec)
    return ScoreResult(bleu * 100)

def corpus_chrf(hyps, refs_list, char_order=6, word_order=2, beta=2):
    refs = refs_list[0]
    total_char_matches = [0] * char_order
    total_char_hyp = [0] * char_order
    total_char_ref = [0] * char_order

    total_word_matches = [0] * word_order
    total_word_hyp = [0] * word_order
    total_word_ref = [0] * word_order

    for hyp, ref in zip(hyps, refs):

        # ---- character n-grams ----
        for n in range(1, char_order + 1):
            hyp_ngrams = get_char_ngrams(hyp, n)
            ref_ngrams = get_char_ngrams(ref, n)

            total_char_hyp[n - 1] += sum(hyp_ngrams.values())
            total_char_ref[n - 1] += sum(ref_ngrams.values())

            for ng in hyp_ngrams:
                total_char_matches[n - 1] += min(
                    hyp_ngrams[ng],
                    ref_ngrams.get(ng, 0)
                )

        # ---- word n-grams (chrF++) ----
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()

        for n in range(1, word_order + 1):
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            total_word_hyp[n - 1] += sum(hyp_ngrams.values())
            total_word_ref[n - 1] += sum(ref_ngrams.values())

            for ng in hyp_ngrams:
                total_word_matches[n - 1] += min(
                    hyp_ngrams[ng],
                    ref_ngrams.get(ng, 0)
                )

    def f_score(matches, hyp_total, ref_total):
        if hyp_total == 0 or ref_total == 0:
            return 0.0

        precision = matches / hyp_total
        recall = matches / ref_total

        if precision + recall == 0:
            return 0.0

        beta2 = beta * beta
        return (1 + beta2) * precision * recall / (beta2 * precision + recall)

    char_f = [
        f_score(total_char_matches[i],
                total_char_hyp[i],
                total_char_ref[i])
        for i in range(char_order)
    ]

    word_f = [
        f_score(total_word_matches[i],
                total_word_hyp[i],
                total_word_ref[i])
        for i in range(word_order)
    ]

    score = (sum(char_f) / char_order + sum(word_f) / word_order) / 2

    return ScoreResult(score * 100)