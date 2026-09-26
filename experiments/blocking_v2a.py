
import os
import re
import time
import gc
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

# ============================================================
# Amazon ML Challenge 2026 — Blocking V2-A (patched)
# ============================================================
# This is TRAIN-first code. It produces candidate_pairs.tsv-style
# output for the training sources and evaluates it against GT.
#
# Run a smoke test first:
#   set SMOKE_ROWS=200000
#   python experiments/blocking_v2a_patched.py
#
# Full run:
#   set SMOKE_ROWS=0
#   python experiments/blocking_v2a_patched.py
#
# IMPORTANT: this script is deliberately conservative. It aborts
# before materializing a pass if its projected safe join is too large.
#
# Fixes applied 2026-09-26 (backup in _backup_pre_v2a_fixes/):
#   - name_token_pair keys are now country|NP|a|P|b on both sides
#     (previously side-prefixed S|NP| / T|NP|, so they never matched).
#   - name_token pass now includes alternate-name (DBA/parenthetical) tokens.
#   - fallback pairs carry fb=1 and a sentinel bp, and always rank after
#     genuine block pairs.
#   - rows of an oversized block lacking pc3/postcode get bounded fallback
#     candidates from that block instead of being silently dropped.

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "student_resource")

S1_PATH = os.path.join(BASE, "dataset", "train", "train_source1.tsv")
S2_PATH = os.path.join(BASE, "dataset", "train", "train_source2.tsv")
S3_PATH = os.path.join(BASE, "dataset", "train", "train_source3.tsv")
GT_PATH = os.path.join(BASE, "dataset", "train", "train_ground_truth.tsv")

OUTPUT_PATH = os.path.join(os.path.dirname(__file__),"blocking_v2a_candidates.tsv")

REPORT_PATH = os.path.join(os.path.dirname(__file__),"blocking_v2a_report.txt")

# ----------------------------
# Candidate/blocking parameters
# ----------------------------
JOINED_CAP = 20_000
TOKEN_CAP = 5_000
ADDRESS_CAP = 10_000
NO_GEO_ADDRESS_CAP = 2_000

PER_PASS_CAP = 100       # Claude's recommended first-run setting
PER_S1_CAP = 300
PER_TARGET_CAP = 50

# Hard safety budget: exact projected RAW pairs per pass (safe block
# pairs + no-geography fallback + unresolved-oversized fallback).
# If a pass exceeds this, stop before the expensive merge.
MAX_PAIRS_PER_PASS = 80_000_000

# Number of deterministic candidates to retain from an unresolved
# oversized block when no further geography exists.
FALLBACK_K = 10

# Block-size sentinel for fallback pairs. Larger than any real block
# product, so min(bp) always prefers a genuine block when both exist.
FALLBACK_BP = 10**15

SMOKE_ROWS = int(os.environ.get("SMOKE_ROWS", "0") or 0)

LEGAL = {
    "private", "pvt", "limited", "ltd", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llp", "lp", "plc",
    "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc"
}

GENERIC_ADDR = {
    "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
    "lane", "ln", "drive", "dr", "highway", "hwy", "way", "place", "pl",
    "building", "bldg", "floor", "fl", "unit", "suite", "ste", "apt",
    "apartment", "near", "opposite", "po", "post", "district", "county",
    "city", "state", "village", "town", "block", "sector", "area", "main"
}

COUNTRY_ALIASES = {
    "usa": "us",
    "u s a": "us",
    "united states": "us",
    "united states of america": "us",
    "us": "us",
    "fr": "france",
    "fra": "france",
    "france": "france",
    "in": "india",
    "ind": "india",
    "india": "india",
}

DBA_RE = re.compile(
    r"\b(?:dba|d/b/a|d\.b\.a\.?|aka|a/k/a|t/a|trading as|formerly|f/k/a)\b",
    re.I,
)


# ============================================================
# Normalization
# ============================================================

def fold(value):
    if pd.isna(value):
        return ""

    s = str(value).strip().casefold()
    s = unicodedata.normalize("NFKD", s)

    # Keep Unicode scripts. Do NOT transliterate here; native-script
    # equality remains useful and avoids destroying Indian/French text.
    replacements = {
        "ß": "ss", "ø": "o", "æ": "ae", "œ": "oe",
        "ł": "l", "đ": "d", "ı": "i",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)

    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("’", "'").replace("'", "")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def country_norm(value):
    x = fold(value)
    return COUNTRY_ALIASES.get(x, x)


def split_name_parts(value):
    """
    Return (main_part, DBA/trade-name part).
    We deliberately preserve the post-DBA portion instead of throwing it away.
    """
    if pd.isna(value):
        return "", ""

    raw = str(value).strip().casefold()

    # Parenthetical content is a useful alternate/trade-name signal.
    parenthetical = ""
    m = re.search(r"\(([^()]*)\)", raw)
    if m:
        parenthetical = m.group(1)

    m = DBA_RE.search(raw)
    if m:
        main = raw[:m.start()]
        alt = raw[m.end():]
    else:
        main = raw
        alt = ""

    # Remove parenthetical material from main only; retain it as alt.
    main = re.sub(r"\([^)]*\)", " ", main)
    return main, (alt or parenthetical)


def clean_name(value):
    main, _ = split_name_parts(value)
    x = fold(main)
    toks = x.split()

    # Strip legal suffix chains.
    while toks and toks[-1] in (LEGAL | {"and", "the", "of"}):
        toks.pop()

    if toks and toks[0] == "the":
        toks = toks[1:]

    return " ".join(toks) if toks else x


def clean_name_alt(value):
    _, alt = split_name_parts(value)
    x = fold(alt)
    toks = x.split()

    while toks and toks[-1] in (LEGAL | {"and", "the", "of"}):
        toks.pop()

    if toks and toks[0] == "the":
        toks = toks[1:]

    return " ".join(toks) if toks else x


def joined(value):
    return value.replace(" ", "") if value else ""


def extract_postcode(value):
    if not value:
        return "", ""

    # Handle Indian-style "400 001".
    m = re.findall(r"(?<!\d)(\d{3})\s+(\d{3})(?!\d)", value)
    if m:
        p = m[-1][0] + m[-1][1]
        return p, p[:3]

    m = re.findall(r"(?<!\d)(\d{5,6})(?!\d)", value)
    if m:
        p = m[-1]
        return p, p[:3]

    return "", ""


def extract_house(value, postcode_value):
    if not value:
        return ""

    toks = value.split()
    forbidden_prev = {
        "suite", "ste", "unit", "apt", "flat", "fl",
        "floor", "room", "rm", "no"
    }

    for i, tok in enumerate(toks[:12]):
        if tok == postcode_value:
            continue

        if re.fullmatch(r"\d{1,5}[a-z]?", tok):
            # Avoid 2nd / 3rd / 21st etc.
            if re.fullmatch(r"\d+(?:st|nd|rd|th)", tok):
                continue
            if i and toks[i - 1] in forbidden_prev:
                continue
            return tok

    return ""


def add_features(df):
    # No df.copy(): caller owns this frame.
    df["country_key"] = df["country"].map(country_norm)

    df["name_clean"] = df["business_name"].map(clean_name)
    df["name_alt"] = df["business_name"].map(clean_name_alt)
    df["name_joined"] = df["name_clean"].map(joined)
    df["name_alt_joined"] = df["name_alt"].map(joined)

    df["addr_clean"] = df["business_address"].map(fold)

    pcs = df["addr_clean"].map(extract_postcode)
    df["postcode"] = pcs.str[0]
    df["pc3"] = pcs.str[1]

    df["house"] = [
        extract_house(a, p)
        for a, p in zip(df["addr_clean"].to_numpy(), df["postcode"].to_numpy())
    ]

    df["name_tokens"] = df["name_clean"].str.split()
    df["name_alt_tokens"] = df["name_alt"].str.split()
    df["addr_tokens"] = df["addr_clean"].str.split()

    return df


# ============================================================
# Name keys
# ============================================================

def build_name_keys(s1, tgt):
    # Primary + DBA/trade-name joined keys.
    left_join = pd.concat([
        s1[["idx", "country_key", "name_joined", "pc3", "postcode"]]
        .rename(columns={"name_joined": "joined"}),
        s1[["idx", "country_key", "name_alt_joined", "pc3", "postcode"]]
        .rename(columns={"name_alt_joined": "joined"}),
    ], ignore_index=True)

    right_join = pd.concat([
        tgt[["idx", "country_key", "name_joined", "pc3", "postcode"]]
        .rename(columns={"name_joined": "joined"}),
        tgt[["idx", "country_key", "name_alt_joined", "pc3", "postcode"]]
        .rename(columns={"name_alt_joined": "joined"}),
    ], ignore_index=True)

    left_join = left_join[left_join.joined != ""].drop_duplicates(["idx", "joined"])
    right_join = right_join[right_join.joined != ""].drop_duplicates(["idx", "joined"])

    left_join["key"] = (
        left_join["country_key"] + "|J|" + left_join["joined"]
    )
    right_join["key"] = (
        right_join["country_key"] + "|J|" + right_join["joined"]
    )

    # Token explosion. This is deliberately only name-token data;
    # address token DF is handled separately with Counter to avoid a
    # ~100M-row address explode.
    # Alt-name tokens are renamed into the same column; otherwise they
    # land in a separate column and are dropped by the explode/dropna.
    alt = {"name_alt_tokens": "name_tokens"}
    tok = pd.concat([
        s1[["idx", "country_key", "name_tokens"]].assign(side="s"),
        s1[["idx", "country_key", "name_alt_tokens"]]
        .rename(columns=alt).assign(side="s"),
        tgt[["idx", "country_key", "name_tokens"]].assign(side="t"),
        tgt[["idx", "country_key", "name_alt_tokens"]]
        .rename(columns=alt).assign(side="t"),
    ], ignore_index=True)

    tok = tok.explode(
        "name_tokens", ignore_index=True
    ).rename(columns={"name_tokens": "token"})

    tok = tok.dropna(subset=["token"])
    tok = tok[tok.token.str.len() >= 2]
    tok = tok.drop_duplicates(["side", "idx", "country_key", "token"])

    name_counts = (
        tok.groupby(["country_key", "token"], sort=False)
        .size()
        .rename("df")
    )

    # Data-driven country-specific stopword set: top 50 tokens by DF.
    nc = name_counts.reset_index()
    stop_rows = (
        nc.sort_values(["country_key", "df"], ascending=[True, False])
        .groupby("country_key", sort=False)
        .head(50)
    )
    stop_pairs = set(zip(stop_rows.country_key, stop_rows.token))

    # Attach DF through a merge, avoiding the previous df_x/df_y crash.
    tok = tok.merge(
        name_counts.reset_index(),
        on=["country_key", "token"],
        how="left",
        validate="many_to_one",
    )

    # Geography lookup with arrays indexed by idx. idx is guaranteed to
    # be arange(len(frame)) for both S1 and target.
    is_s = tok.side.eq("s").to_numpy()
    ix = tok.idx.to_numpy(np.int64)

    geo = np.empty(len(tok), dtype=object)
    post = np.empty(len(tok), dtype=object)

    s_pc3 = s1["pc3"].to_numpy(object)
    t_pc3 = tgt["pc3"].to_numpy(object)
    s_post = s1["postcode"].to_numpy(object)
    t_post = tgt["postcode"].to_numpy(object)

    geo[is_s] = s_pc3[ix[is_s]]
    geo[~is_s] = t_pc3[ix[~is_s]]
    post[is_s] = s_post[ix[is_s]]
    post[~is_s] = t_post[ix[~is_s]]

    tok["geo"] = geo
    tok["post"] = post

    tok["base"] = tok.country_key + "|T|" + tok.token

    # Remove top-50 country stopwords completely from token blocking.
    tok = tok[
        ~tok.apply(
            lambda r: (r.country_key, r.token) in stop_pairs,
            axis=1
        )
    ]

    # Low/medium DF -> country+token.
    # High DF -> geography-refined token key only.
    tok["key"] = np.where(
        tok["df"] <= 50_000,
        tok["base"],
        np.where(
            tok["geo"] != "",
            tok["base"] + "|G|" + tok["geo"],
            ""
        ),
    )
    tok = tok[tok.key != ""]

    ts = (
        tok[tok.side == "s"][["idx", "key", "geo", "post"]]
        .rename(columns={"geo": "pc3", "post": "postcode"})
        .drop_duplicates(["idx", "key"])
    )

    tt = (
        tok[tok.side == "t"][["idx", "key", "geo", "post"]]
        .rename(columns={"geo": "pc3", "post": "postcode"})
        .drop_duplicates(["idx", "key"])
    )

    # Rare-token pair keys: top 2 tokens per record by DF.
    tok_rank = tok.copy()
    tok_rank["df_sort"] = tok_rank["df"].fillna(10**12)
    tok_rank = tok_rank.sort_values(
        ["side", "idx", "df_sort", "token"],
        ascending=[True, True, True, True],
    )

    # Each idx has exactly one country_key, so grouping by it is safe.
    top2 = (
        tok_rank.groupby(["side", "idx"], sort=False)
        .head(2)
        .groupby(["side", "idx", "country_key"], sort=False)["token"]
        .agg(list)
        .reset_index()
    )

    # Same key namespace on both sides (no side prefix), country-scoped.
    top2 = top2[top2["token"].map(len) >= 2]
    top2["key"] = (
        top2["country_key"]
        + "|NP|"
        + top2["token"].map(lambda x: "|P|".join(sorted(x)))
    )

    ps = (
        top2[top2.side == "s"][["idx", "key"]]
        .merge(
            s1[["idx", "pc3", "postcode"]],
            on="idx",
            how="left",
            validate="many_to_one",
        )
    )

    pt = (
        top2[top2.side == "t"][["idx", "key"]]
        .merge(
            tgt[["idx", "pc3", "postcode"]],
            on="idx",
            how="left",
            validate="many_to_one",
        )
    )

    # All returned frames use only idx/key, keeping block memory bounded.
    return {
        "joined": (
            left_join[["idx", "key", "pc3", "postcode"]],
            right_join[["idx", "key", "pc3", "postcode"]],
        ),
        "token": (ts, tt),
        "token_pair": (ps, pt),
        "name_counts": name_counts,
    }


# ============================================================
# Address keys
# ============================================================

def informative_addr_tokens(tokens):
    if not isinstance(tokens, list):
        return []
    return {
        t for t in tokens
        if len(t) >= 3
        and t not in GENERIC_ADDR
        and not t.isdigit()
    }


def build_address_frequency(s1, tgt):
    """
    Streaming Counter over per-record token SETS.
    This avoids exploding ~12M rows into ~100M+ address-token rows.
    """
    counts = Counter()

    for df in (s1, tgt):
        countries = df.country_key.to_numpy()
        token_lists = df.addr_tokens.to_numpy(object)

        for c, toks in zip(countries, token_lists):
            if not toks:
                continue
            for t in informative_addr_tokens(toks):
                counts[(c, t)] += 1

    return counts


def address_key_tables(s1, tgt, addr_freq):
    def make(df):
        ids = []
        keys = []
        pids = []
        pc3s = []
        posts = []

        for r in df.itertuples(index=False):
            toks = informative_addr_tokens(r.addr_tokens)

            toks = sorted(
                toks,
                key=lambda x: (addr_freq.get((r.country_key, x), 10**12), x)
            )
            top = toks[:3]

            if r.house and top and r.pc3:
                ids.append(r.idx)
                keys.append(
                    f"{r.country_key}|A1|{r.house}|{top[0]}|{r.pc3}"
                )
                pids.append(4)
                pc3s.append(r.pc3)
                posts.append(r.postcode)

            if r.postcode and top:
                for t in top:
                    ids.append(r.idx)
                    keys.append(
                        f"{r.country_key}|A2|{t}|{r.postcode}"
                    )
                    pids.append(5)
                    pc3s.append(r.pc3)
                    posts.append(r.postcode)

            # Country-level house+street-token fallback.
            # It is intentionally handled under a smaller cap later.
            if r.house and top:
                ids.append(r.idx)
                keys.append(
                    f"{r.country_key}|A3|{r.house}|{top[0]}"
                )
                pids.append(6)
                pc3s.append(r.pc3)
                posts.append(r.postcode)

        return pd.DataFrame({
            "idx": np.asarray(ids, dtype=np.int64),
            "key": np.asarray(keys, dtype=object),
            "pass_id": np.asarray(pids, dtype=np.int8),
            "pc3": np.asarray(pc3s, dtype=object),
            "postcode": np.asarray(posts, dtype=object),
        }).drop_duplicates(["idx", "key"])

    return make(s1), make(tgt)


# ============================================================
# Blocking
# ============================================================

def _head_pair_count(left_blk, right_blk, k):
    """Exact size of block_head_pairs(left, right, k) from block labels."""
    if len(left_blk) == 0 or len(right_blk) == 0:
        return 0
    rc = pd.Series(right_blk).value_counts().clip(upper=k)
    return int(
        pd.Series(left_blk).map(rc).fillna(0).to_numpy(np.int64).sum()
    )


def estimate_refined(k1, kt, cap, geo1, geot, levels=("pc3", "postcode"),
                     k=FALLBACK_K):
    """
    Exact count of the pairs block_join() will materialize, computed from
    block sizes only (no pair Cartesian product). Mirrors block_join step
    for step: safe block pairs, no-geography fallback pairs, and the final
    unresolved-oversized fallback pairs.
    """
    idx1 = k1.idx.to_numpy(np.int64)
    idxt = kt.idx.to_numpy(np.int64)
    key1 = k1.key.to_numpy(object)
    keyt = kt.key.to_numpy(object)
    blk1 = key1
    blkt = keyt

    est = {
        "safe": 0, "nogeo_fb": 0, "final_fb": 0,
        "oversized_pairs": 0, "oversized_blocks": 0,
    }

    for level in (None, *levels):
        if len(idx1) == 0 or len(idxt) == 0:
            break

        if level is None:
            j1, jt = key1, keyt
        else:
            g1 = geo1[level][idx1]
            gt = geot[level][idxt]
            m1 = g1 != ""
            mt = gt != ""

            est["nogeo_fb"] += _head_pair_count(blk1[~m1], blkt, k)
            est["nogeo_fb"] += _head_pair_count(blkt[~mt], blk1, k)

            idx1, key1, g1 = idx1[m1], key1[m1], g1[m1]
            idxt, keyt, gt = idxt[mt], keyt[mt], gt[mt]
            j1 = key1 + "|G|" + g1
            jt = keyt + "|G|" + gt

        codes, _ = pd.factorize(np.concatenate([j1, jt]), sort=False)
        c1 = codes[:len(j1)]
        ct = codes[len(j1):]
        n_codes = int(codes.max()) + 1 if len(codes) else 0
        p = (
            np.bincount(c1, minlength=n_codes).astype(np.int64)
            * np.bincount(ct, minlength=n_codes).astype(np.int64)
        )

        est["safe"] += int(p[(p > 0) & (p <= cap)].sum())
        big_codes = np.flatnonzero(p > cap)
        est["oversized_pairs"] += int(p[big_codes].sum())
        est["oversized_blocks"] += int(len(big_codes))

        if len(big_codes) == 0:
            idx1 = idx1[:0]
            idxt = idxt[:0]
            break

        keep1 = np.isin(c1, big_codes)
        keept = np.isin(ct, big_codes)
        idx1, key1, blk1 = idx1[keep1], key1[keep1], j1[keep1]
        idxt, keyt, blkt = idxt[keept], keyt[keept], jt[keept]

    if len(idx1) and len(idxt):
        est["final_fb"] = _head_pair_count(blk1, blkt, k)

    est["total"] = est["safe"] + est["nogeo_fb"] + est["final_fb"]
    return est


def fallback_pairs(rem1, remt, k=FALLBACK_K):
    """
    Bounded fallback for blocks still oversized after the last geography
    level. Each S1 row gets the first k targets of its own REFINED block
    (blk = key|G|pc3|G|postcode), not of the original unrefined key, so
    candidates stay within the S1 row's postcode. Never more pairs than
    the old base-key grouping (a refined block is a subset of its key).
    """
    return block_head_pairs(rem1, remt, k, left_is_s1=True)


def _empty_pairs():
    return pd.DataFrame({
        "s": np.array([], dtype=np.int64),
        "t": np.array([], dtype=np.int64),
        "bp": np.array([], dtype=np.int64),
        "fb": np.array([], dtype=np.int8),
    })


def block_head_pairs(left, right, k, left_is_s1):
    """
    Pair each row of `left` with the first k rows of `right` sharing the
    same oversized block `blk`. Bounded by k * len(left); never a
    Cartesian product. Used for rows lacking the next geography level and
    for blocks still oversized after the last level.
    """
    if left.empty or right.empty:
        return _empty_pairs()

    head = right[["idx", "blk"]].groupby("blk", sort=False).head(k)
    m = left[["idx", "blk"]].merge(
        head, on="blk", suffixes=("_l", "_r"), sort=False
    )
    if m.empty:
        return _empty_pairs()

    li = m.idx_l.to_numpy(np.int64)
    ri = m.idx_r.to_numpy(np.int64)
    return pd.DataFrame({
        "s": li if left_is_s1 else ri,
        "t": ri if left_is_s1 else li,
        "bp": np.full(len(m), FALLBACK_BP, dtype=np.int64),
        "fb": np.ones(len(m), dtype=np.int8),
    })


def block_join(k1, kt, cap, geo1, geot, levels=("pc3", "postcode")):
    """
    Vectorized refined block join.

    geo1/geot are dictionaries of arrays indexed directly by idx.
    This avoids the old merge/assign index-alignment bug.
    """
    cols = ["idx", "key", "pc3", "postcode", "blk"]

    # blk = the (possibly geo-refined) join key of the oversized block a
    # row currently belongs to; used to bound no-geography fallbacks.
    rem1 = k1[["idx", "key", "pc3", "postcode"]].copy()
    remt = kt[["idx", "key", "pc3", "postcode"]].copy()
    rem1["blk"] = rem1["key"]
    remt["blk"] = remt["key"]

    out = []
    n_nogeo_fb = 0

    for level in (None, *levels):
        if rem1.empty or remt.empty:
            break

        if level is None:
            j1 = rem1.key.to_numpy(object)
            jt = remt.key.to_numpy(object)
        else:
            g1_all = geo1[level][rem1.idx.to_numpy(np.int64)]
            gt_all = geot[level][remt.idx.to_numpy(np.int64)]

            m1 = g1_all != ""
            mt = gt_all != ""

            # Rows lacking this geography cannot be refined further.
            # Instead of discarding them, give each a bounded set of
            # fallback candidates from the whole oversized block.
            nogeo = [
                block_head_pairs(
                    rem1.iloc[np.flatnonzero(~m1)], remt,
                    FALLBACK_K, left_is_s1=True,
                ),
                block_head_pairs(
                    remt.iloc[np.flatnonzero(~mt)], rem1,
                    FALLBACK_K, left_is_s1=False,
                ),
            ]
            for fb in nogeo:
                if not fb.empty:
                    n_nogeo_fb += len(fb)
                    out.append(fb)

            rem1 = rem1.iloc[np.flatnonzero(m1)][cols]
            remt = remt.iloc[np.flatnonzero(mt)][cols]
            j1 = rem1.key.to_numpy(object) + "|G|" + g1_all[m1]
            jt = remt.key.to_numpy(object) + "|G|" + gt_all[mt]

        # Factorize both sides together: integer joins are much cheaper
        # than repeatedly hashing long Python strings.
        both = np.concatenate([j1, jt])
        codes, _ = pd.factorize(both, sort=False)

        c1 = codes[:len(j1)]
        ct = codes[len(j1):]

        n_codes = max(
            int(c1.max()) if len(c1) else -1,
            int(ct.max()) if len(ct) else -1,
        ) + 1

        a = np.bincount(c1, minlength=n_codes)
        b = np.bincount(ct, minlength=n_codes)
        p = a.astype(np.int64) * b.astype(np.int64)

        safe_codes = np.flatnonzero((p > 0) & (p <= cap))
        big_codes = np.flatnonzero(p > cap)

        if len(safe_codes):
            mask1 = np.isin(c1, safe_codes)
            maskt = np.isin(ct, safe_codes)

            a1 = pd.DataFrame({
                "idx": rem1.idx.to_numpy(np.int64)[mask1],
                "jcode": c1[mask1],
            })
            at = pd.DataFrame({
                "idx": remt.idx.to_numpy(np.int64)[maskt],
                "jcode": ct[maskt],
            })

            if len(a1) and len(at):
                m = a1.merge(
                    at,
                    on="jcode",
                    suffixes=("_s", "_t"),
                    sort=False,
                )

                out.append(pd.DataFrame({
                    "s": m.idx_s.to_numpy(np.int64),
                    "t": m.idx_t.to_numpy(np.int64),
                    "bp": p[m.jcode.to_numpy(np.int64)],
                    "fb": np.zeros(len(m), dtype=np.int8),
                }))

        if len(big_codes) == 0:
            rem1 = rem1.iloc[0:0]
            remt = remt.iloc[0:0]
            break

        keep1 = np.flatnonzero(np.isin(c1, big_codes))
        keept = np.flatnonzero(np.isin(ct, big_codes))

        rem1 = rem1.iloc[keep1][cols].assign(blk=j1[keep1])
        remt = remt.iloc[keept][cols].assign(blk=jt[keept])

    # Remaining oversized blocks get bounded fallback candidates.
    n_final_fb = 0
    if not rem1.empty and not remt.empty:
        fb = fallback_pairs(rem1, remt, FALLBACK_K)
        if not fb.empty:
            n_final_fb = len(fb)
            out.append(fb)

    print(
        f"  fallback pairs: no-geo={n_nogeo_fb:,} "
        f"unresolved-oversized={n_final_fb:,}"
    )

    if not out:
        return _empty_pairs()

    return pd.concat(out, ignore_index=True)


# ============================================================
# Candidate compaction/ranking
# ============================================================

# Ranking: genuine before fallback, then more supporting keys, then
# more specific (smaller) blocks.
RANK_COLS = ["s", "fb", "nk", "bp"]
RANK_ASC = [True, True, False, True]


def compact_pass(m, ntargets, per_s1=PER_PASS_CAP):
    if m.empty:
        return m

    m = m.copy()
    m["pid"] = (
        m.s.to_numpy(np.int64) * np.int64(ntargets)
        + m.t.to_numpy(np.int64)
    )

    # nk = number of GENUINE blocking keys/pieces supporting the pair.
    # fb = 1 only if every supporting row is a fallback row.
    m["genuine"] = (1 - m["fb"].to_numpy(np.int8)).astype(np.int64)
    g = (
        m.groupby("pid", sort=False)
        .agg(
            bp=("bp", "min"),
            nk=("genuine", "sum"),
            fb=("fb", "min"),
        )
        .reset_index()
    )

    g["s"] = g.pid // ntargets
    g["t"] = g.pid % ntargets

    g = g.sort_values(RANK_COLS, ascending=RANK_ASC, kind="stable")

    g = g[g.groupby("s", sort=False).cumcount() < per_s1]

    return g[["pid", "s", "t", "bp", "nk", "fb"]]


def final_compact(pass_frames, ntargets):
    allp = pd.concat(pass_frames, ignore_index=True)
    del pass_frames

    # Pair-level consolidation across passes.
    allp = (
        allp.groupby("pid", sort=False)
        .agg(
            s=("s", "first"),
            t=("t", "first"),
            bp=("bp", "min"),
            nk=("nk", "sum"),
            fb=("fb", "min"),
        )
        .reset_index(drop=True)
    )

    # Genuine before fallback; more supporting keys; more specific blocks.
    allp = allp.sort_values(RANK_COLS, ascending=RANK_ASC, kind="stable")
    allp = allp[
        allp.groupby("s", sort=False).cumcount() < PER_S1_CAP
    ]

    # Target-side cap. Same ranking principle.
    allp = allp.sort_values(
        ["t"] + RANK_COLS[1:], ascending=RANK_ASC, kind="stable"
    )
    allp = allp[
        allp.groupby("t", sort=False).cumcount() < PER_TARGET_CAP
    ]

    # Restore best-first S1 ordering.
    allp = allp.sort_values(RANK_COLS, ascending=RANK_ASC, kind="stable")

    return allp


# ============================================================
# Evaluation
# ============================================================

def evaluate(out, gt, s2_ids, s3_ids):
    """
    Candidate-set evaluation. Ground truth is aligned to `out` by
    source1_entity_id, never by row position (GT file order != S1 order).
    `gt` must be read with keep_default_na=False so empty cells stay "".
    """
    if gt["source1_entity_id"].duplicated().any():
        raise RuntimeError("Duplicate source1_entity_id in ground truth.")

    gt_map = gt.set_index("source1_entity_id")["matched_entity_ids"]
    missing = ~out["source1_entity_id"].isin(gt_map.index)
    if missing.any():
        raise RuntimeError(
            f"{int(missing.sum()):,} output S1 IDs are missing from GT, e.g. "
            f"{out.loc[missing, 'source1_entity_id'].head(5).tolist()}"
        )
    n_gt_unused = len(gt_map) - len(out)

    true_col = gt_map.reindex(out["source1_entity_id"]).to_numpy(object)
    pred_col = out["candidate_entity_ids"].to_numpy(object)

    total_true = total_tp = 0
    s2_true = s2_tp = s3_true = s3_tp = other_true = 0
    true_empty = fp_on_true_empty = 0
    matched_zero = matched_rows = 0
    macro_f = 0.0

    for true_s, pred_s in zip(true_col, pred_col):
        true = {x.strip() for x in str(true_s).split(",") if x.strip()}
        pred = {x.strip() for x in str(pred_s).split(",") if x.strip()}

        if not true:
            true_empty += 1
            # Correct empty prediction gets F=1 for singleton/no-match row.
            if pred:
                fp_on_true_empty += 1
            macro_f += 1.0 if not pred else 0.0
            continue

        matched_rows += 1
        if not pred:
            matched_zero += 1

        hit = true & pred
        total_true += len(true)
        total_tp += len(hit)

        for x in true:
            if x in s2_ids:
                s2_true += 1
                s2_tp += x in hit
            elif x in s3_ids:
                s3_true += 1
                s3_tp += x in hit
            else:
                other_true += 1

        precision = len(hit) / len(pred) if pred else 0.0
        recall = len(hit) / len(true)

        denom = 0.25 * precision + recall
        f05 = (
            1.25 * precision * recall / denom
            if denom > 0 else 0.0
        )
        macro_f += f05

    macro_f /= len(out)

    def ratio(a, b):
        return f"{a / b:.4%}" if b else "n/a"

    print("\n===== BLOCKING EVALUATION (GT aligned by source1_entity_id) =====")
    print(f"S1 rows evaluated:               {len(out):,}")
    print(f"GT rows not in output (ignored): {n_gt_unused:,}")
    print(f"True-empty S1 rows:              {true_empty:,}")
    print(f"False positives on true-empty:   {fp_on_true_empty:,}")
    print(f"Matched S1 rows:                 {matched_rows:,}")
    print(f"Matched S1 with zero candidates: {matched_zero:,}")
    print(f"GT true pairs:                   {total_true:,}")
    print(f"True pairs retrieved (TP):       {total_tp:,}")
    print(f"Candidate true-pair recall:      {ratio(total_tp, total_true)}")
    print(f"S2 pair recall:                  {ratio(s2_tp, s2_true)}"
          f"  ({s2_tp:,}/{s2_true:,})")
    print(f"S3 pair recall:                  {ratio(s3_tp, s3_true)}"
          f"  ({s3_tp:,}/{s3_true:,})")
    print(f"True IDs not in loaded S2/S3:    {other_true:,}")
    print(f"Macro F0.5 of candidate set:     {macro_f:.6f}")


# ============================================================
# Main
# ============================================================

def read_source(path, nrows=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )


def main():
    t0 = time.time()

    print("=" * 70)
    print("Amazon ML Challenge — Blocking V2-A PATCHED")
    print("=" * 70)
    print(f"SMOKE_ROWS = {SMOKE_ROWS:,}")
    print(f"PER_PASS_CAP = {PER_PASS_CAP}")
    print(f"PER_S1_CAP = {PER_S1_CAP}")
    print(f"PER_TARGET_CAP = {PER_TARGET_CAP}")

    # ----------------------------
    # Preflight: seconds, not hours
    # ----------------------------
    print("\n[1/8] Preflight")

    gt_head = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        nrows=5,
    )

    assert {
        "source1_entity_id",
        "matched_entity_ids",
    } <= set(gt_head.columns), (
        f"Unexpected GT columns: {gt_head.columns.tolist()}"
    )

    # ----------------------------
    # Load
    # ----------------------------
    print("\n[2/8] Loading sources")

    s1 = read_source(S1_PATH, SMOKE_ROWS or None)
    s2 = read_source(S2_PATH, SMOKE_ROWS or None)
    s3 = read_source(S3_PATH, SMOKE_ROWS or None)

    if set(s2.entity_id) & set(s3.entity_id):
        raise RuntimeError(
            "S2/S3 entity_id collision detected. "
            "Bare candidate IDs would be ambiguous."
        )

    tgt = pd.concat(
        [
            s2.assign(source="S2"),
            s3.assign(source="S3"),
        ],
        ignore_index=True,
    )

    del s2, s3
    gc.collect()

    # idx is a positional index by construction.
    s1["idx"] = np.arange(len(s1), dtype=np.int64)
    tgt["idx"] = np.arange(len(tgt), dtype=np.int64)

    print(f"S1 rows:      {len(s1):,}")
    print(f"Target rows:   {len(tgt):,}")

    # ----------------------------
    # Features
    # ----------------------------
    print("\n[3/8] Building features")
    s1 = add_features(s1)
    tgt = add_features(tgt)

    assert np.array_equal(
        s1.idx.to_numpy(),
        np.arange(len(s1)),
    )
    assert np.array_equal(
        tgt.idx.to_numpy(),
        np.arange(len(tgt)),
    )

    geo_s = {
        "pc3": s1["pc3"].to_numpy(object),
        "postcode": s1["postcode"].to_numpy(object),
    }
    geo_t = {
        "pc3": tgt["pc3"].to_numpy(object),
        "postcode": tgt["postcode"].to_numpy(object),
    }

    # ----------------------------
    # Name keys
    # ----------------------------
    print("\n[4/8] Name keys")
    nk = build_name_keys(s1, tgt)

    name_counts = nk.pop("name_counts")

    # ----------------------------
    # Address keys
    # ----------------------------
    print("\n[5/8] Address frequency + keys")
    addr_freq = build_address_frequency(s1, tgt)
    ak_s, ak_t = address_key_tables(s1, tgt, addr_freq)

    print(f"Address frequency entries: {len(addr_freq):,}")

    # ----------------------------
    # Passes
    # ----------------------------
    print("\n[6/8] Dry-run + materialization")

    pass_specs = [
        ("joined", nk["joined"], JOINED_CAP, 1),
        ("name_token", nk["token"], TOKEN_CAP, 2),
        ("name_token_pair", nk["token_pair"], TOKEN_CAP, 3),
        ("address", (ak_s, ak_t), ADDRESS_CAP, 4),
    ]

    pass_frames = []
    total_fallback_raw = 0

    for name, (k1, kt), cap, pass_id in pass_specs:
        if len(k1) == 0 or len(kt) == 0:
            print(f"{name}: empty")
            continue

        est = estimate_refined(k1, kt, cap, geo_s, geo_t)

        print(
            f"{name:18s} "
            f"projected-total={est['total']:,} "
            f"(safe={est['safe']:,} "
            f"nogeo-fb={est['nogeo_fb']:,} "
            f"final-fb={est['final_fb']:,}) "
            f"oversized-pairs={est['oversized_pairs']:,} "
            f"oversized-blocks={est['oversized_blocks']:,}"
        )

        # Budget covers EVERYTHING block_join will materialize,
        # fallback pairs included.
        if est["total"] > MAX_PAIRS_PER_PASS:
            raise RuntimeError(
                f"{name} projected {est['total']:,} raw pairs "
                f"(safe={est['safe']:,}, nogeo-fb={est['nogeo_fb']:,}, "
                f"final-fb={est['final_fb']:,}), exceeding "
                f"MAX_PAIRS_PER_PASS={MAX_PAIRS_PER_PASS:,}. "
                f"Do NOT continue blindly; lower/refine the pass cap first."
            )

        m = block_join(
            k1,
            kt,
            cap,
            geo_s,
            geo_t,
        )

        n_fb = int(m["fb"].sum()) if len(m) else 0
        total_fallback_raw += n_fb
        print(f"  materialized raw pairs: {len(m):,} (fallback {n_fb:,})")

        if len(m) != est["total"]:
            print(
                f"  WARNING: estimate {est['total']:,} != materialized "
                f"{len(m):,}; the safety budget is not exact for {name}."
            )

        if m.empty:
            continue

        m["pass_id"] = pass_id

        compact = compact_pass(
            m,
            ntargets=len(tgt),
            per_s1=PER_PASS_CAP,
        )

        print(f"  after per-pass compaction: {len(compact):,}")

        pass_frames.append(compact)

        del m, compact
        gc.collect()

    if not pass_frames:
        raise RuntimeError("No candidate pairs generated.")

    # ----------------------------
    # Final candidate set
    # ----------------------------
    print("\n[7/8] Final candidate compaction")

    allp = final_compact(
        pass_frames,
        ntargets=len(tgt),
    )

    n_fb_final = int((allp["fb"] == 1).sum())

    print(f"Final candidate pairs: {len(allp):,}")
    print(f"  fallback-only pairs kept: {n_fb_final:,} "
          f"(raw fallback materialized: {total_fallback_raw:,})")
    print(f"Average candidates/S1: {len(allp) / len(s1):.2f}")
    print(
        f"Zero-candidate S1: "
        f"{len(s1) - allp.s.nunique():,}"
    )

    # ----------------------------
    # Fast output construction
    # ----------------------------
    print("\n[8/8] Writing output")

    target_ids = tgt.entity_id.to_numpy(object)

    allp["tid"] = target_ids[
        allp.t.to_numpy(np.int64)
    ]

    candidate_series = (
        allp.sort_values(RANK_COLS, ascending=RANK_ASC, kind="stable")
        .groupby("s", sort=False)["tid"]
        .agg(",".join)
    )

    out = pd.DataFrame({
        "source1_entity_id": s1.entity_id.to_numpy(object),
        "candidate_entity_ids": candidate_series.reindex(
            np.arange(len(s1)),
            fill_value="",
        ).to_numpy(object),
    })

    out.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print(f"Wrote: {OUTPUT_PATH}")

    # Full training evaluation only if this is not a smoke test.
    if SMOKE_ROWS == 0:
        print("\nLoading full ground truth for evaluation...")
        gt = pd.read_csv(
            GT_PATH,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
        evaluate(
            out,
            gt,
            s2_ids=set(tgt.entity_id[tgt.source == "S2"]),
            s3_ids=set(tgt.entity_id[tgt.source == "S3"]),
        )

    runtime = (time.time() - t0) / 60
    print(f"\nRuntime: {runtime:.2f} minutes")

    # Lightweight report.
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Amazon ML Challenge — Blocking V2-A patched\n")
        f.write(f"S1 rows: {len(s1):,}\n")
        f.write(f"Target rows: {len(tgt):,}\n")
        f.write(f"Final candidate pairs: {len(allp):,}\n")
        f.write(f"Fallback-only pairs kept: {n_fb_final:,}\n")
        f.write(f"Raw fallback pairs materialized: {total_fallback_raw:,}\n")
        f.write(f"Average candidates/S1: {len(allp)/len(s1):.4f}\n")
        f.write(
            f"Zero-candidate S1: "
            f"{len(s1)-allp.s.nunique():,}\n"
        )
        f.write(f"Runtime minutes: {runtime:.2f}\n")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
