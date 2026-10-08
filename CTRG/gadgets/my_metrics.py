import math
from collections import Counter

import numpy as np
import sklearn.metrics as sklm
import torch
from torchmetrics import Metric


def _tokenize(text):
    return str(text).lower().split()


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def compute_scores(gts, res):
    """Lightweight drop-in replacement for ``generation_api.metrics.compute_scores``.

    The original implementation pulled in ``pycocoevalcap`` (BLEU/METEOR/ROUGE/
    CIDEr) plus a stack of Java jars.  This standalone version only computes
    corpus-level BLEU-1..4 and ROUGE-L with pure Python; METEOR and CIDER are
    reported as 0.0 (they are not used by the pretraining task).

    :param gts: {key: reference(s)} -- a str or a list of str
    :param res: {key: hypothesis}   -- a str or a list of str
    """
    def _as_list(v):
        return [v] if isinstance(v, str) else list(v)

    matches = [0] * 4
    cand_totals = [0] * 4
    ref_len = 0
    cand_len = 0
    rouge_l = 0.0
    n_items = 0

    for key, refs in gts.items():
        hyp = _as_list(res.get(key, [""]))[0]
        hyp_tokens = _tokenize(hyp)
        ref_token_lists = [_tokenize(r) for r in _as_list(refs)]

        cand_len += len(hyp_tokens)
        ref_len += min(len(r) for r in ref_token_lists) if ref_token_lists else 0

        for n in range(1, 5):
            if len(hyp_tokens) < n:
                cand_totals[n - 1] += 0
                continue
            hyp_counts = _ngram_counts(hyp_tokens, n)
            cand_totals[n - 1] += max(len(hyp_tokens) - n + 1, 0)
            max_ref_counts = Counter()
            for r in ref_token_lists:
                for g, c in _ngram_counts(r, n).items():
                    max_ref_counts[g] = max(max_ref_counts[g], c)
            matches[n - 1] += sum(min(c, max_ref_counts[g]) for g, c in hyp_counts.items())

        best = 0.0
        for r in ref_token_lists:
            l = _lcs(hyp_tokens, r)
            if l == 0 or not hyp_tokens or not r:
                continue
            p, rec = l / len(hyp_tokens), l / len(r)
            best = max(best, (2 * p * rec / (p + rec)) if (p + rec) else 0.0)
        rouge_l += best
        n_items += 1

    n_items = max(n_items, 1)

    # brevity penalty (corpus level)
    if cand_len == 0:
        bp = 0.0
    elif ref_len and cand_len < ref_len:
        bp = math.exp(1 - ref_len / cand_len)
    else:
        bp = 1.0

    out = {}
    geo_sum = 0.0
    geo_cnt = 0
    for n in range(1, 5):
        if cand_totals[n - 1]:
            prec = matches[n - 1] / cand_totals[n - 1]
            geo_sum += math.log(prec) if prec > 0 else -math.inf
            geo_cnt += 1
        else:
            prec = 0.0
            geo_sum = -math.inf
            geo_cnt += 1
        out[f"BLEU_{n}"] = prec

    # cumulative BLEU-4 (standard "BLEU") reported on top of the per-order scores
    if geo_cnt and geo_sum != -math.inf:
        out["BLEU_4"] = bp * math.exp(geo_sum / geo_cnt)

    out["ROUGE_L"] = rouge_l / n_items
    out["METEOR"] = 0.0
    out["CIDER"] = 0.0
    return out


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total
    
class BLUE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("BLEU_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BLEU_2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BLEU_3", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("BLEU_4", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("METEOR", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ROUGE_L", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("CIDER", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, target, pred):
        eval_res = compute_scores(target, pred)

        self.BLEU_1 += eval_res["BLEU_1"]
        self.BLEU_2 += eval_res["BLEU_2"]
        self.BLEU_3 += eval_res["BLEU_3"]
        self.BLEU_4 += eval_res["BLEU_4"]

        # print(self.BLEU_1)
        #self.METEOR += eval_res["METEOR"]

        self.ROUGE_L += eval_res["ROUGE_L"]
        self.CIDER += eval_res["CIDER"]
        
        self.total += 1
        # print("updating", self.BLEU_4/self.total, eval_res["BLEU_4"], self.BLEU_1/self.total, self.ROUGE_L/self.total)

    def compute(self):
        # print(self.BLEU_4, self.total, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        return self.BLEU_4 / self.total, self.BLEU_1 / self.total, self.BLEU_3 / self.total, self.ROUGE_L / self.total # , self.CIDER / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class BinaryMicroF1(Metric):
    """Positive-label micro F1 accumulated over samples and label classes."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.as_tensor(preds, device=self.tp.device).bool()
        target = torch.as_tensor(target, device=self.tp.device).bool()
        if preds.shape != target.shape:
            raise ValueError(
                f"pred/target shape mismatch: {tuple(preds.shape)} != "
                f"{tuple(target.shape)}"
            )
        self.tp += torch.logical_and(preds, target).sum()
        self.fp += torch.logical_and(preds, ~target).sum()
        self.fn += torch.logical_and(~preds, target).sum()

    def compute(self):
        denominator = 2 * self.tp + self.fp + self.fn
        return torch.where(
            denominator > 0,
            2.0 * self.tp.float() / denominator.float(),
            torch.zeros_like(denominator, dtype=torch.float),
        )


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class VQARADScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.best_score = 0
        self.best_close_score = 0
        self.best_open_score = 0

    def update(self, logits, target, types=None):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        close_scores = scores[types == 0]
        open_scores = scores[types == 1]

        self.close_score += close_scores.sum()
        self.close_total += len(close_scores)
        self.open_score += open_scores.sum()
        self.open_total += len(open_scores)

        self.score += scores.sum()
        self.total += len(scores)

    def compute(self):
        score = self.score / self.total
        return score

    def get_best_score(self):
        self.sync()
        score = self.score / self.total
        if score > self.best_score:
            self.best_score = score
            self.best_close_score = self.close_score / self.close_total if self.close_total != 0 else 0
            self.best_open_score = self.open_score / self.open_total if self.open_total != 0 else 0
        self.unsync()
        return self.best_score

    def get_best_close_score(self):
        return self.best_close_score

    def get_best_open_score(self):
        return self.best_open_score


class ROCScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits))
        self.y_trues.append(y_true)
        self.y_scores.append(y_score)

    def compute(self):
        try:
            score = sklm.roc_auc_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                       np.concatenate([y_score.cpu().numpy() for y_score in self.y_scores], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score


class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_preds", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits)) > 0.5
        self.y_trues.append(y_true)
        self.y_preds.append(y_score)

    def compute(self):
        try:
            score = sklm.f1_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                  np.concatenate([y_pred.cpu().numpy() for y_pred in self.y_preds], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score
