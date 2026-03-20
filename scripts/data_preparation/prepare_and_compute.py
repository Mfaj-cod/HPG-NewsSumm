import argparse
import json
import os
import re
import unicodedata
import hashlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

BOILERPLATE_PATTERNS = [
    r"subscribe",
    r"sign up",
    r"advertisement",
    r"advertising",
    r"cookie policy",
    r"privacy policy",
    r"terms of use",
    r"all rights reserved",
    r"copyright",
    r"follow us",
    r"share this",
    r"click here",
    r"read more",
    r"related articles",
    r"newsletter",
    r"contact us",
]

PUNCT_TRANSLATION = str.maketrans({
    "\u201c": "\"",
    "\u201d": "\"",
    "\u2018": "'",
    "\u2019": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2022": "-",
})


def normalize_col_name(name: str) -> str:
    name = re.sub(r"\s+", "", str(name).strip().lower())
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def resolve_columns(df: pd.DataFrame, args) -> dict:
    normalized = {normalize_col_name(c): c for c in df.columns}

    synonyms = {
        "article_text": [
            "article_text", "articletext", "article", "content", "story",
            "fulltext", "body", "articlebody", "text"
        ],
        "human_summary": [
            "human_summary", "humansummary", "summary", "abstract",
            "shortsummary", "highlights"
        ],
        "headline": ["headline", "title", "heading", "newsheadline"],
        "newspaper_name": ["newspaper_name", "newspaper", "source", "publisher", "newssource"],
        "published_date": ["published_date", "published", "date", "pubdate", "timestamp"],
        "news_category": ["news_category", "category", "section", "topic"],
    }

    resolved = {}
    for key, opts in synonyms.items():
        override = getattr(args, f"col_{key}", None)
        if override:
            if override not in df.columns:
                raise ValueError(f"Column override not found: {override}")
            resolved[key] = override
            continue

        found = None
        for opt in opts:
            opt_norm = normalize_col_name(opt)
            if opt_norm in normalized:
                found = normalized[opt_norm]
                break
        if found is None and key in ("article_text", "human_summary"):
            raise ValueError(f"Required column not found: {key}")
        resolved[key] = found

    return resolved


def clean_text(text: str, lowercase: bool = False) -> str:
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "lxml").get_text()
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(PUNCT_TRANSLATION)

    # Remove boilerplate fragments
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def word_count(text: str) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    return len(text.split())


def compute_basic_stats(df: pd.DataFrame, cols: dict, cluster_col=None, article_col_override=None, summary_col_override=None) -> dict:
    article_col = article_col_override or cols["article_text"]
    summary_col = summary_col_override or cols["human_summary"]
    source_col = cols.get("newspaper_name")
    category_col = cols.get("news_category")
    date_col = cols.get("published_date")

    stats = {
        "total_documents": int(len(df)),
        "missing_article_ratio": float(df[article_col].isna().mean()),
        "missing_summary_ratio": float(df[summary_col].isna().mean()),
    }

    article_lengths = df[article_col].fillna("").apply(word_count)
    summary_lengths = df[summary_col].fillna("").apply(word_count)

    stats["avg_article_length"] = float(article_lengths.mean()) if len(df) else 0.0
    stats["avg_summary_length"] = float(summary_lengths.mean()) if len(df) else 0.0
    stats["total_article_tokens"] = int(article_lengths.sum()) if len(df) else 0
    stats["total_summary_tokens"] = int(summary_lengths.sum()) if len(df) else 0

    if source_col:
        stats["num_sources"] = int(df[source_col].nunique(dropna=True))
        stats["source_distribution"] = (
            df[source_col].fillna("UNKNOWN").value_counts().head(20).to_dict()
        )

    if category_col:
        stats["category_distribution"] = (
            df[category_col].fillna("UNKNOWN").value_counts().to_dict()
        )

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        years = dates.dt.year.dropna()
        stats["year_distribution"] = years.value_counts().sort_index().to_dict()
        if len(years):
            stats["year_min"] = int(years.min())
            stats["year_max"] = int(years.max())

    # Exact duplicate ratio on raw article text
    raw_dups = df[article_col].fillna("").duplicated().mean() if len(df) else 0.0
    stats["duplicate_ratio_exact"] = float(raw_dups)

    if cluster_col and cluster_col in df.columns:
        stats["num_clusters"] = int(df[cluster_col].nunique())

    return stats


def save_report_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_report_md(path: str, title: str, payload: dict):
    lines = [f"# {title}", ""]
    for k, v in payload.items(): 
        if isinstance(v, dict):
            lines.append(f"- {k}:")
            for kk, vv in list(v.items())[:20]:
                lines.append(f"  - {kk}: {vv}")
        else:
            lines.append(f"- {k}: {v}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def filter_language(texts, lang="en", threshold=0.8):
    try:
        from langdetect import detect_langs
    except Exception as e:
        raise RuntimeError("langdetect is required for language filtering") from e

    keep = []
    for text in tqdm(texts, desc="Language filter"):
        if not text:
            keep.append(False)
            continue
        try:
            langs = detect_langs(text)
            if not langs:
                keep.append(False)
                continue
            top = langs[0]
            keep.append(top.lang == lang and top.prob >= threshold)
        except Exception:
            keep.append(False)
    return np.array(keep, dtype=bool)


def exact_hash_dedup(df: pd.DataFrame, text_col: str):
    seen = {}
    keep_mask = np.ones(len(df), dtype=bool)
    for idx, text in enumerate(df[text_col].fillna("").tolist()):
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            keep_mask[idx] = False
        else:
            seen[h] = idx
    return keep_mask


def build_minhash(text: str, num_perm=128, shingle_size=5):
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    words = text.split()
    if len(words) < shingle_size:
        shingles = [text]
    else:
        shingles = [" ".join(words[i:i + shingle_size]) for i in range(len(words) - shingle_size + 1)]

    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return mh


def minhash_candidates(texts, threshold=0.9, num_perm=128, shingle_size=5):
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    candidates = set()

    for idx, text in tqdm(enumerate(texts), total=len(texts), desc="MinHash indexing"):
        mh = build_minhash(text, num_perm=num_perm, shingle_size=shingle_size)
        matches = lsh.query(mh)
        for m in matches:
            if m != idx:
                pair = (min(idx, m), max(idx, m))
                candidates.add(pair)
        lsh.insert(idx, mh)
        minhashes[idx] = mh

    return minhashes, candidates


def minhash_dedup(candidates, minhashes, threshold=0.9):
    keep = {}
    to_drop = set()
    for i, j in tqdm(list(candidates), desc="MinHash filtering"):
        if j in to_drop:
            continue
        sim = minhashes[i].jaccard(minhashes[j])
        if sim >= threshold:
            to_drop.add(j)
    for idx in minhashes.keys():
        keep[idx] = idx not in to_drop
    return keep


def tfidf_dedup(texts, candidates, threshold=0.95, max_features=50000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if not candidates:
        return np.ones(len(texts), dtype=bool)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    X = normalize(X)

    to_drop = set()
    for i, j in tqdm(list(candidates), desc="TFIDF filtering"):
        if j in to_drop:
            continue
        sim = X[i].multiply(X[j]).sum()
        if sim >= threshold:
            to_drop.add(j)

    keep = np.ones(len(texts), dtype=bool)
    for idx in to_drop:
        keep[idx] = False
    return keep


def build_tfidf_features(texts, max_features=50000):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    return X


def ensure_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        raise RuntimeError(
            "NLTK punkt not found. Run: python -m nltk.downloader punkt"
        ) from e


def add_linguistic_features(df, text_col):
    ensure_nltk()
    import nltk
    import textstat

    token_counts = []
    sentence_counts = []
    lexical_div = []
    readability = []

    for text in tqdm(df[text_col].tolist(), desc="Linguistic features"):
        tokens = text.split()
        token_count = len(tokens)
        token_counts.append(token_count)
        try:
            sents = nltk.sent_tokenize(text)
            sentence_counts.append(len(sents))
        except Exception:
            sentence_counts.append(max(1, text.count(".") + text.count("!") + text.count("?")))

        if token_count:
            lexical_div.append(len(set(tokens)) / token_count)
        else:
            lexical_div.append(0.0)

        try:
            readability.append(float(textstat.flesch_reading_ease(text)))
        except Exception:
            readability.append(0.0)

    df["token_count"] = token_counts
    df["sentence_count"] = sentence_counts
    df["lexical_diversity"] = lexical_div
    df["readability"] = readability
    return df


def add_ner_features(df, text_col):
    try:
        import spacy
    except Exception as e:
        raise RuntimeError("spaCy is required for NER features") from e

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
    except Exception as e:
        raise RuntimeError(
            "spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm"
        ) from e

    ent_counts = []
    top_labels = []

    for doc in tqdm(nlp.pipe(df[text_col].tolist(), batch_size=32), total=len(df), desc="NER features"):
        labels = [ent.label_ for ent in doc.ents]
        ent_counts.append(len(labels))
        label_counts = Counter(labels).most_common(3)
        top_labels.append([f"{lab}:{cnt}" for lab, cnt in label_counts])

    df["ner_count"] = ent_counts
    df["top_entity_labels"] = top_labels
    return df


def cluster_quality_metrics(embeddings, labels, sample_size=2000):
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(labels)
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return {"silhouette": -1.0, "intra_sim": 0.0, "inter_dist": 0.0}

    if n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        emb = embeddings[idx]
        lbls = labels[idx]
    else:
        emb = embeddings
        lbls = labels

    try:
        sil = float(silhouette_score(emb, lbls, metric="cosine"))
    except Exception:
        sil = -1.0

    sim = cosine_similarity(emb)
    intra = []
    inter = []
    for i in range(len(lbls)):
        for j in range(i + 1, len(lbls)):
            if lbls[i] == lbls[j]:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])

    intra_sim = float(np.mean(intra)) if intra else 0.0
    inter_dist = float(1.0 - np.mean(inter)) if inter else 0.0

    return {"silhouette": sil, "intra_sim": intra_sim, "inter_dist": inter_dist}


def run_clustering(
    embeddings,
    seed=42,
    kmeans_k=None,
    dbscan_eps=0.5,
    dbscan_min_samples=5,
    max_agglomerative=50000,
    max_dbscan=50000,
    max_clusters=30,
):
    from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
    from scipy import sparse

    n = embeddings.shape[0]
    if kmeans_k is None:
        kmeans_k = max(2, min(max_clusters, int(np.sqrt(n))))
    else:
        kmeans_k = max(2, min(max_clusters, kmeans_k))

    results = {}

    is_sparse = sparse.issparse(embeddings)

    # Use MiniBatchKMeans for large datasets or sparse features
    if n > 50000 or is_sparse:
        km = MiniBatchKMeans(n_clusters=kmeans_k, random_state=seed, batch_size=2048, n_init="auto")
    else:
        km = KMeans(n_clusters=kmeans_k, random_state=seed, n_init="auto")
    labels_km = km.fit_predict(embeddings)
    results["kmeans"] = {"labels": labels_km, "metrics": cluster_quality_metrics(embeddings, labels_km)}

    if n <= max_agglomerative and not is_sparse:
        agg = AgglomerativeClustering(n_clusters=kmeans_k, linkage="average")
        labels_ag = agg.fit_predict(embeddings)
        results["agglomerative"] = {"labels": labels_ag, "metrics": cluster_quality_metrics(embeddings, labels_ag)}

    if n <= max_dbscan and not is_sparse:
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine")
        labels_db = db.fit_predict(embeddings)
        results["dbscan"] = {"labels": labels_db, "metrics": cluster_quality_metrics(embeddings, labels_db)}

    # Choose best by silhouette score
    best = None
    best_score = -1.0
    for name, payload in results.items():
        score = payload["metrics"]["silhouette"]
        if score > best_score:
            best_score = score
            best = name

    return best, results


def validate_summaries(df, article_col, summary_col, min_ratio=0.01, max_ratio=0.5,
                       min_overlap=0.2, min_rougeL=0.1, use_rouge=True):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if use_rouge else None

    keep = []
    ratios = []
    overlaps = []
    rouge_scores = []

    for art, summ in tqdm(list(zip(df[article_col], df[summary_col])), desc="Summary validation"):
        art_tokens = art.split()
        sum_tokens = summ.split()

        if not art_tokens or not sum_tokens:
            keep.append(False)
            ratios.append(0.0)
            overlaps.append(0.0)
            rouge_scores.append(0.0)
            continue

        ratio = len(sum_tokens) / max(1, len(art_tokens))
        overlap = len(set(sum_tokens) & set(art_tokens)) / max(1, len(set(sum_tokens)))

        rougeL = 0.0
        if scorer:
            rougeL = scorer.score(art, summ)["rougeL"].fmeasure

        ok = (min_ratio <= ratio <= max_ratio) and (overlap >= min_overlap)
        if scorer:
            ok = ok and (rougeL >= min_rougeL)

        keep.append(ok)
        ratios.append(float(ratio))
        overlaps.append(float(overlap))
        rouge_scores.append(float(rougeL))

    df["summary_ratio"] = ratios
    df["summary_overlap"] = overlaps
    df["summary_rougeL"] = rouge_scores

    return np.array(keep, dtype=bool)


def build_cluster_records(df, cols, cluster_col="cluster_id"):
    records = []
    group = df.groupby(cluster_col)

    for cluster_id, g in group:
        g = g.sort_values("published_date_parsed")
        documents = g["article_clean"].tolist()
        summaries = g[cols["human_summary"]].tolist()

        # Pick best summary based on quality metrics
        if "summary_rougeL" in g.columns:
            best_idx = g["summary_rougeL"].fillna(0).idxmax()
        else:
            best_idx = g["summary_ratio"].fillna(0).idxmax()

        summary = df.loc[best_idx, cols["human_summary"]]
        summary_source_doc_id = int(df.loc[best_idx, "doc_id"])

        meta = {
            "cluster_size": int(len(g)),
            "sources": g[cols["newspaper_name"]].fillna("UNKNOWN").tolist() if cols["newspaper_name"] else [],
            "dates": g["published_date_parsed"].dt.strftime("%Y-%m-%d").fillna("").tolist(),
            "categories": g[cols["news_category"]].fillna("UNKNOWN").tolist() if cols["news_category"] else [],
            "time_range": {
                "start": g["published_date_parsed"].min().strftime("%Y-%m-%d") if g["published_date_parsed"].notna().any() else "",
                "end": g["published_date_parsed"].max().strftime("%Y-%m-%d") if g["published_date_parsed"].notna().any() else "",
            },
            "summary_source_doc_id": summary_source_doc_id,
        }

        docs_meta = []
        for _, row in g.iterrows():
            docs_meta.append({
                "doc_id": int(row["doc_id"]),
                "orig_index": int(row["orig_index"]),
                "headline": row.get(cols["headline"], "") if cols["headline"] else "",
                "source": row.get(cols["newspaper_name"], "") if cols["newspaper_name"] else "",
                "published_date": row["published_date_parsed"].strftime("%Y-%m-%d") if pd.notna(row["published_date_parsed"]) else "",
                "category": row.get(cols["news_category"], "") if cols["news_category"] else "",
                "token_count": int(row.get("token_count", 0)),
                "sentence_count": int(row.get("sentence_count", 0)),
                "lexical_diversity": float(row.get("lexical_diversity", 0.0)),
                "readability": float(row.get("readability", 0.0)),
                "ner_count": int(row.get("ner_count", 0)),
                "top_entity_labels": row.get("top_entity_labels", []),
                "topic_id": int(row.get("topic_id", -1)),
                "topic_prob": float(row.get("topic_prob", 0.0)),
                "embedding_index": int(row.get("embedding_index", -1)),
            })

        records.append({
            "cluster_id": cluster_id,
            "documents": documents,
            "summary": summary,
            "metadata": meta,
            "documents_meta": docs_meta,
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/NewsSumm_Dataset.xlsx")
    parser.add_argument("--output_dir", default="data/enhanced")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--minhash_threshold", type=float, default=0.9)
    parser.add_argument("--cosine_threshold", type=float, default=0.95)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--lang_threshold", type=float, default=0.8)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--min_article_words", type=int, default=50)
    parser.add_argument("--min_summary_words", type=int, default=10)
    parser.add_argument("--summary_min_ratio", type=float, default=0.01)
    parser.add_argument("--summary_max_ratio", type=float, default=0.5)
    parser.add_argument("--summary_min_overlap", type=float, default=0.2)
    parser.add_argument("--summary_min_rougeL", type=float, default=0.1)
    parser.add_argument("--skip_topics", action="store_true")
    parser.add_argument("--skip_clustering", action="store_true")
    parser.add_argument("--skip_annotation_validation", action="store_true")
    parser.add_argument("--skip_ner", action="store_true")
    parser.add_argument("--export_csv", action="store_true")
    parser.add_argument("--export_xlsx", action="store_true")
    parser.add_argument("--kmeans_k", type=int, default=None)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)
    parser.add_argument("--max_agglomerative", type=int, default=50000)
    parser.add_argument("--max_dbscan", type=int, default=50000)
    parser.add_argument("--max_clusters", type=int, default=30)
    parser.add_argument("--cluster_tfidf_max_features", type=int, default=50000)

    # Column overrides
    parser.add_argument("--col_article_text", default=None)
    parser.add_argument("--col_human_summary", default=None)
    parser.add_argument("--col_headline", default=None)
    parser.add_argument("--col_newspaper_name", default=None)
    parser.add_argument("--col_published_date", default=None)
    parser.add_argument("--col_news_category", default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    print("Loading XLSX:", args.input)
    df = pd.read_excel(args.input, engine="openpyxl")
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)

    df["orig_index"] = df.index.astype(int)
    df = df.reset_index(drop=True)

    cols = resolve_columns(df, args)

    # Baseline stats
    baseline_stats = compute_basic_stats(df, cols)
    baseline_stats["sampled"] = bool(args.sample)
    save_report_json(os.path.join(args.reports_dir, "baseline_stats.json"), baseline_stats)
    save_report_md(os.path.join(args.reports_dir, "baseline_stats.md"), "Baseline Dataset Stats", baseline_stats)

    cleaning_log = {"initial_rows": int(len(df))}

    # Drop missing critical fields
    df = df.dropna(subset=[cols["article_text"], cols["human_summary"]]).reset_index(drop=True)
    cleaning_log["after_missing"] = int(len(df))

    # Clean and normalize text fields
    tqdm.pandas()
    df["article_clean"] = df[cols["article_text"]].progress_apply(lambda x: clean_text(x, args.lowercase))
    df[cols["human_summary"]] = df[cols["human_summary"]].progress_apply(lambda x: clean_text(x, args.lowercase))
    if cols["headline"]:
        df[cols["headline"]] = df[cols["headline"]].progress_apply(lambda x: clean_text(x, args.lowercase))

    # Remove very short / noisy entries
    df["article_words"] = df["article_clean"].apply(word_count)
    df["summary_words"] = df[cols["human_summary"]].apply(word_count)
    before_len = len(df)
    df = df[(df["article_words"] >= args.min_article_words) & (df["summary_words"] >= args.min_summary_words)].reset_index(drop=True)
    cleaning_log["noise_removed"] = int(before_len - len(df))
    cleaning_log["after_noise"] = int(len(df))

    # Language filter
    lang_keep = filter_language(df["article_clean"].tolist(), lang=args.lang, threshold=args.lang_threshold)
    before_lang = len(df)
    df = df[lang_keep].reset_index(drop=True)
    cleaning_log["language_removed"] = int(before_lang - len(df))
    cleaning_log["after_language"] = int(len(df))

    # Deduplication: exact hash
    keep_exact = exact_hash_dedup(df, "article_clean")
    before_exact = len(df)
    df = df[keep_exact].reset_index(drop=True)
    cleaning_log["exact_dups_removed"] = int(before_exact - len(df))
    cleaning_log["after_exact_dedup"] = int(len(df))

    # Deduplication: MinHash LSH
    minhashes, candidates = minhash_candidates(
        df["article_clean"].tolist(),
        threshold=args.minhash_threshold,
        num_perm=128,
        shingle_size=5,
    )
    keep_mh = minhash_dedup(candidates, minhashes, threshold=args.minhash_threshold)
    keep_mask = np.array([keep_mh[i] for i in range(len(df))], dtype=bool)
    before_mh = len(df)
    df = df[keep_mask].reset_index(drop=True)
    cleaning_log["minhash_dups_removed"] = int(before_mh - len(df))
    cleaning_log["after_minhash_dedup"] = int(len(df))

    # Deduplication: TFIDF cosine
    _, candidates_tfidf = minhash_candidates(
        df["article_clean"].tolist(),
        threshold=args.minhash_threshold,
        num_perm=128,
        shingle_size=5,
    )
    keep_tfidf = tfidf_dedup(df["article_clean"].tolist(), candidates_tfidf, threshold=args.cosine_threshold)
    before_tfidf = len(df)
    df = df[keep_tfidf].reset_index(drop=True)
    cleaning_log["tfidf_dups_removed"] = int(before_tfidf - len(df))
    cleaning_log["after_tfidf_dedup"] = int(len(df))

    # Feature engineering
    df = add_linguistic_features(df, "article_clean")
    if args.skip_ner:
        df["ner_count"] = 0
        df["top_entity_labels"] = [[] for _ in range(len(df))]
    else:
        df = add_ner_features(df, "article_clean")

    # Topic modeling (skipped)
    df["topic_id"] = -1
    df["topic_prob"] = 0.0

    # Clustering (TF-IDF based, capped at max_clusters)
    if args.skip_clustering:
        df["cluster_label"] = np.arange(len(df))
        cleaning_log["cluster_method"] = "none"
        cleaning_log["cluster_metrics"] = {}
        cleaning_log["cluster_quality"] = -1.0
    else:
        tfidf_features = build_tfidf_features(
            df["article_clean"].tolist(),
            max_features=args.cluster_tfidf_max_features,
        )
        best, all_results = run_clustering(
            tfidf_features,
            seed=42,
            kmeans_k=args.kmeans_k,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            max_agglomerative=args.max_agglomerative,
            max_dbscan=args.max_dbscan,
            max_clusters=args.max_clusters,
        )
        cluster_labels = all_results[best]["labels"]
        df["cluster_label"] = cluster_labels
        cluster_metrics = {name: res["metrics"] for name, res in all_results.items()}
        cleaning_log["cluster_method"] = best
        cleaning_log["cluster_metrics"] = cluster_metrics
        cleaning_log["cluster_quality"] = cluster_metrics[best]["silhouette"] if best else -1.0

    # Normalize cluster ids
    cluster_ids = []
    for idx, lab in enumerate(df["cluster_label"].tolist()):
        if lab == -1:
            cluster_ids.append(f"noise_{idx}")
        else:
            cluster_ids.append(f"cluster_{lab}")
    df["cluster_id"] = cluster_ids

    # Temporal structuring
    if cols["published_date"]:
        df["published_date_parsed"] = pd.to_datetime(df[cols["published_date"]], errors="coerce")
    else:
        df["published_date_parsed"] = pd.to_datetime(pd.Series([None] * len(df)))

    # Annotation validation
    keep_ann = None
    if not args.skip_annotation_validation:
        keep_ann = validate_summaries(
            df,
            "article_clean",
            cols["human_summary"],
            min_ratio=args.summary_min_ratio,
            max_ratio=args.summary_max_ratio,
            min_overlap=args.summary_min_overlap,
            min_rougeL=args.summary_min_rougeL,
            use_rouge=True,
        )
        before_ann = len(df)
        df = df[keep_ann].reset_index(drop=True)
        cleaning_log["annotation_removed"] = int(before_ann - len(df))
        cleaning_log["after_annotation"] = int(len(df))
    else:
        cleaning_log["annotation_removed"] = 0
        cleaning_log["after_annotation"] = int(len(df))

    # Assign ids after final filtering
    df["doc_id"] = np.arange(len(df))

    df["embedding_index"] = -1

    cleaning_log["final_rows"] = int(len(df))
    save_report_json(os.path.join(args.reports_dir, "cleaning_log.json"), cleaning_log)

    # Build enhanced dataset
    enhanced_records = build_cluster_records(df, cols, cluster_col="cluster_id")
    out_json = os.path.join(args.output_dir, "newssumm_enhanced.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(enhanced_records, f, indent=2, ensure_ascii=False)

    if args.export_csv:
        df.to_csv(os.path.join(args.output_dir, "enhanced_flat.csv"), index=False)
    if args.export_xlsx:
        df.to_excel(os.path.join(args.output_dir, "enhanced_flat.xlsx"), index=False)

    # Enhanced stats and comparison
    enhanced_stats = compute_basic_stats(
        df,
        cols,
        cluster_col="cluster_id",
        article_col_override="article_clean",
    )
    enhanced_stats["sampled"] = bool(args.sample)
    save_report_json(os.path.join(args.reports_dir, "enhanced_stats.json"), enhanced_stats)
    save_report_md(os.path.join(args.reports_dir, "enhanced_stats.md"), "Enhanced Dataset Stats", enhanced_stats)

    noise_pct = cleaning_log.get("noise_removed", 0) / max(1, cleaning_log.get("initial_rows", 1))
    comparison = {
        "Articles": [baseline_stats.get("total_documents", 0), enhanced_stats.get("total_documents", 0)],
        "Clusters": [baseline_stats.get("num_clusters", 0), enhanced_stats.get("num_clusters", 0)],
        "Sources": [baseline_stats.get("num_sources", 0), enhanced_stats.get("num_sources", 0)],
        "Avg Article Length": [baseline_stats.get("avg_article_length", 0), enhanced_stats.get("avg_article_length", 0)],
        "Avg Summary Length": [baseline_stats.get("avg_summary_length", 0), enhanced_stats.get("avg_summary_length", 0)],
        "Total Article Tokens": [baseline_stats.get("total_article_tokens", 0), enhanced_stats.get("total_article_tokens", 0)],
        "Total Summary Tokens": [baseline_stats.get("total_summary_tokens", 0), enhanced_stats.get("total_summary_tokens", 0)],
        "Duplicate Rate": [baseline_stats.get("duplicate_ratio_exact", 0), enhanced_stats.get("duplicate_ratio_exact", 0)],
        "Noise Percentage": [noise_pct, 0.0],
        "Cluster Quality Score": [np.nan, cleaning_log.get("cluster_quality", np.nan)],
        "Temporal Coverage": [
            f"{baseline_stats.get('year_min','')}-{baseline_stats.get('year_max','')}",
            f"{enhanced_stats.get('year_min','')}-{enhanced_stats.get('year_max','')}",
        ],
    }

    comp_df = pd.DataFrame(comparison, index=["Original", "Enhanced"])
    comp_df.to_csv(os.path.join(args.reports_dir, "comparison_table.csv"))

    print("Enhanced dataset saved to:", out_json)
    print("Reports saved to:", args.reports_dir)


if __name__ == "__main__":
    main()
