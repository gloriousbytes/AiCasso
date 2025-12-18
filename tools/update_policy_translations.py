from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


RTL_LANGS = {"ar", "fa", "he", "ur"}

_GOOGLE_BLOCKED = False

_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(r"^https?://\S+$")
_MYMEMORY_CACHE: dict[tuple[str, str], str] = {}
_MYMEMORY_MIN_INTERVAL_S = 1.0
_LAST_MYMEMORY_REQUEST_TS = 0.0

_ARGOS_PAIR_CACHE: dict[str, bool] = {}
_ARGOS_CACHE: dict[tuple[str, str], str] = {}

_NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
_NLLB_TOKENIZER = None
_NLLB_MODEL = None
_NLLB_CACHE: dict[str, str] = {}


def _ensure_nllb_loaded() -> None:
    global _NLLB_MODEL, _NLLB_TOKENIZER  # noqa: PLW0603 - lazy global init

    if _NLLB_MODEL is not None and _NLLB_TOKENIZER is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    _NLLB_TOKENIZER = AutoTokenizer.from_pretrained(_NLLB_MODEL_NAME)
    _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MODEL_NAME)


def _translate_en_to_nllb_batch(texts: list[str], target_lang: str) -> list[str]:
    if target_lang != "zu":
        raise RuntimeError("NLLB batch currently only wired for Zulu (zu)")  # noqa: TRY003

    _ensure_nllb_loaded()
    assert _NLLB_MODEL is not None
    assert _NLLB_TOKENIZER is not None

    to_translate: list[str] = []
    indices: list[int] = []
    out: list[str] = [""] * len(texts)
    for i, t in enumerate(texts):
        cached = _NLLB_CACHE.get(t)
        if cached is not None:
            out[i] = cached
        else:
            to_translate.append(t)
            indices.append(i)

    if not to_translate:
        return out

    import torch  # type: ignore

    _NLLB_TOKENIZER.src_lang = "eng_Latn"
    inputs = _NLLB_TOKENIZER(to_translate, return_tensors="pt", padding=True, truncation=True)
    forced_bos_token_id = _NLLB_TOKENIZER.convert_tokens_to_ids("zul_Latn")

    with torch.inference_mode():
        generated = _NLLB_MODEL.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=3,
        )
    decoded = _NLLB_TOKENIZER.batch_decode(generated, skip_special_tokens=True)
    for idx, translated in zip(indices, decoded, strict=False):
        out[idx] = translated
        _NLLB_CACHE[texts[idx]] = translated

    return out


def _translate_en_to_nllb(text: str, target_lang: str) -> str:
    return _translate_en_to_nllb_batch([text], target_lang)[0]


def _argos_can_translate(target_lang: str) -> bool:
    cached = _ARGOS_PAIR_CACHE.get(target_lang)
    if cached is not None:
        return cached
    try:
        import argostranslate.translate as argos_translate  # type: ignore

        translation = argos_translate.get_translation_from_codes("en", target_lang)
        ok = translation is not None
    except Exception:
        ok = False
    _ARGOS_PAIR_CACHE[target_lang] = ok
    return ok


def _translate_en_to_argos(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    key = (target_lang, text)
    cached = _ARGOS_CACHE.get(key)
    if cached is not None:
        return cached
    import argostranslate.translate as argos_translate  # type: ignore

    out = argos_translate.translate(text, "en", target_lang)
    _ARGOS_CACHE[key] = out
    return out


def _translate_en_to(text: str, target_lang: str, *, timeout_s: int = 30) -> str:
    global _GOOGLE_BLOCKED  # noqa: PLW0603 - module-level cache flag

    if target_lang == "en":
        return text

    base_url = (
        "https://translate.googleapis.com/translate_a/single"
        f"?client=gtx&sl=en&tl={urllib.parse.quote(target_lang)}&dt=t"
    )
    payload = urllib.parse.urlencode({"q": text}).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}

    if _GOOGLE_BLOCKED:
        raise RuntimeError("Google Translate is rate-limited (429); skipping")  # noqa: TRY003

    last_error: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(base_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            translated = "".join(seg[0] for seg in parsed[0] if seg and seg[0])
            if not translated.strip():
                raise RuntimeError("Empty translation response")
            return translated
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                _GOOGLE_BLOCKED = True
                raise
            time.sleep(0.6 * (2**attempt))
        except Exception as exc:  # noqa: BLE001 - small script; retry on any failure
            last_error = exc
            time.sleep(0.6 * (2**attempt))

    raise RuntimeError(f"Translation failed after retries: {last_error}")  # noqa: TRY003


def _translate_en_to_mymemory_small(text: str, target_lang: str, *, timeout_s: int = 30) -> str:
    global _LAST_MYMEMORY_REQUEST_TS  # noqa: PLW0603 - module-level throttle

    if target_lang == "en":
        return text

    key = (target_lang, text)
    cached = _MYMEMORY_CACHE.get(key)
    if cached is not None:
        return cached

    url = "https://api.mymemory.translated.net/get"
    payload = urllib.parse.urlencode(
        {"q": text, "langpair": f"en|{target_lang}", "de": "support@aicassoapp.com"}
    ).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}

    last_error: Exception | None = None
    for attempt in range(6):
        try:
            now = time.monotonic()
            elapsed = now - _LAST_MYMEMORY_REQUEST_TS
            if elapsed < _MYMEMORY_MIN_INTERVAL_S:
                time.sleep(_MYMEMORY_MIN_INTERVAL_S - elapsed)

            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            _LAST_MYMEMORY_REQUEST_TS = time.monotonic()

            parsed = json.loads(raw)
            translated = parsed.get("responseData", {}).get("translatedText", "")
            if not translated:
                raise RuntimeError("Empty MyMemory translation response")  # noqa: TRY003
            if translated.startswith("QUERY LENGTH LIMIT EXCEEDED"):
                raise RuntimeError(translated)  # noqa: TRY003
            _MYMEMORY_CACHE[key] = translated
            return translated
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                        continue
                    except ValueError:
                        pass
                time.sleep(2.0 * (2**attempt))
                continue
            time.sleep(0.6 * (2**attempt))
        except Exception as exc:  # noqa: BLE001 - small script; retry on any failure
            last_error = exc
            time.sleep(0.6 * (2**attempt))

    raise RuntimeError(f"MyMemory translation failed after retries: {last_error}")  # noqa: TRY003


def _chunk_text(text: str, max_len: int) -> list[str]:
    text = text.strip()
    if not text:
        return [""]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        cut = remaining.rfind(" ", 0, max_len + 1)
        if cut <= 0:
            cut = max_len
        chunk = remaining[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _translate_text_fallback(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text

    if not text.strip():
        return text

    leading_ws = re.match(r"^\s+", text)
    trailing_ws = re.search(r"\s+$", text)
    prefix = leading_ws.group(0) if leading_ws else ""
    suffix = trailing_ws.group(0) if trailing_ws else ""

    core = text.strip()
    if not core:
        return text

    if core == "AiCasso" or _EMAIL_RE.fullmatch(core) or _URL_RE.fullmatch(core):
        return prefix + core + suffix

    core = re.sub(r"\s+", " ", core)

    if _argos_can_translate(target_lang):
        translated = _translate_en_to_argos(core, target_lang).strip()
        return prefix + translated + suffix

    if target_lang == "zu":
        translated = _translate_en_to_nllb(core, target_lang).strip()
        return prefix + translated + suffix

    chunks = _chunk_text(core, 500)
    translated_chunks: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        translated_chunks.append(_translate_en_to_mymemory_small(chunk, target_lang))

    translated = " ".join(translated_chunks).strip()
    return prefix + translated + suffix


def _translate_html_fragment_fallback(fragment: str, target_lang: str) -> str:
    if target_lang == "en":
        return fragment

    if _argos_can_translate(target_lang):
        parts = _TAG_SPLIT_RE.split(fragment)
        out_parts: list[str] = []
        for part in parts:
            if not part:
                continue
            if part.startswith("<") and part.endswith(">"):
                out_parts.append(part)
            else:
                out_parts.append(_translate_text_fallback(part, target_lang))
        return "".join(out_parts)

    if target_lang == "zu":
        parts = _TAG_SPLIT_RE.split(fragment)
        out_parts = parts[:]

        batch_indices: list[int] = []
        batch_texts: list[str] = []
        whitespace_map: dict[int, tuple[str, str]] = {}

        def flush_batch() -> None:
            nonlocal batch_indices, batch_texts
            if not batch_texts:
                return
            translated_texts = _translate_en_to_nllb_batch(batch_texts, target_lang)
            for idx, translated in zip(batch_indices, translated_texts, strict=False):
                prefix, suffix = whitespace_map.get(idx, ("", ""))
                out_parts[idx] = prefix + translated + suffix
            batch_indices = []
            batch_texts = []

        for i, part in enumerate(parts):
            if not part or part.isspace() or (part.startswith("<") and part.endswith(">")):
                continue

            leading_ws = re.match(r"^\\s+", part)
            trailing_ws = re.search(r"\\s+$", part)
            prefix = leading_ws.group(0) if leading_ws else ""
            suffix = trailing_ws.group(0) if trailing_ws else ""
            core = part.strip()
            if not core:
                continue

            if core == "AiCasso" or _EMAIL_RE.fullmatch(core) or _URL_RE.fullmatch(core):
                out_parts[i] = prefix + core + suffix
                continue

            core = re.sub(r"\\s+", " ", core)
            whitespace_map[i] = (prefix, suffix)
            batch_indices.append(i)
            batch_texts.append(core)

            if len(batch_texts) >= 12:
                flush_batch()

        flush_batch()
        return "".join(out_parts)

    marker = "AICASSOSPLITMARK12345"
    max_query_len = 480
    sep_len = len(marker) + 2  # " " + marker + " "

    def batch_translate_texts(texts: list[str]) -> list[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return [_translate_en_to_mymemory_small(texts[0], target_lang)]

        combined = f" {marker} ".join(texts)
        translated = _translate_en_to_mymemory_small(combined, target_lang)
        split = [s.strip() for s in re.split(rf"\\s*{re.escape(marker)}\\s*", translated) if s is not None]
        if len(split) != len(texts):
            return [_translate_en_to_mymemory_small(t, target_lang) for t in texts]
        return split

    parts = _TAG_SPLIT_RE.split(fragment)
    out_parts = parts[:]

    batch_indices: list[int] = []
    batch_texts: list[str] = []
    batch_len = 0
    whitespace_map: dict[int, tuple[str, str]] = {}

    def flush_batch() -> None:
        nonlocal batch_indices, batch_texts, batch_len
        if not batch_texts:
            return
        translated_texts = batch_translate_texts(batch_texts)
        for idx, translated in zip(batch_indices, translated_texts, strict=False):
            prefix, suffix = whitespace_map.get(idx, ("", ""))
            out_parts[idx] = prefix + translated + suffix
        batch_indices = []
        batch_texts = []
        batch_len = 0

    for i, part in enumerate(parts):
        if not part or part.isspace() or (part.startswith("<") and part.endswith(">")):
            continue

        leading_ws = re.match(r"^\\s+", part)
        trailing_ws = re.search(r"\\s+$", part)
        prefix = leading_ws.group(0) if leading_ws else ""
        suffix = trailing_ws.group(0) if trailing_ws else ""
        core = part.strip()
        if not core:
            continue

        if core == "AiCasso" or _EMAIL_RE.fullmatch(core) or _URL_RE.fullmatch(core):
            out_parts[i] = prefix + core + suffix
            continue

        core = re.sub(r"\\s+", " ", core)
        whitespace_map[i] = (prefix, suffix)

        if len(core) > max_query_len:
            flush_batch()
            out_parts[i] = _translate_text_fallback(part, target_lang)
            continue

        proposed_len = len(core) if not batch_texts else batch_len + sep_len + len(core)
        if proposed_len > max_query_len:
            flush_batch()

        batch_indices.append(i)
        batch_texts.append(core)
        batch_len = len(core) if len(batch_texts) == 1 else (batch_len + sep_len + len(core))

    flush_batch()
    return "".join(out_parts)


def _extract_inner(html: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find <{tag}>...</{tag}> block")  # noqa: TRY003
    return match.group(1)


def _replace_inner(html: str, tag: str, inner: str) -> str:
    pattern = rf"(<{tag}[^>]*>)(.*?)(</{tag}>)"
    replaced, count = re.subn(
        pattern,
        lambda m: f"{m.group(1)}{inner}{m.group(3)}",
        html,
        flags=re.IGNORECASE | re.DOTALL,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not replace <{tag}>...</{tag}> block")  # noqa: TRY003
    return replaced


def _replace_html_tag(html: str, lang: str) -> str:
    dir_attr = ' dir="rtl"' if lang in RTL_LANGS else ""
    replaced, count = re.subn(
        r'<html\s+lang="[^"]+"\s*>',
        f'<html lang="{lang}"{dir_attr}>',
        html,
        flags=re.IGNORECASE,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace <html lang=...> tag")  # noqa: TRY003
    return replaced


def _replace_title(html: str, title_text: str) -> str:
    safe_title = title_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    replaced, count = re.subn(
        r"<title>.*?</title>",
        f"<title>{safe_title}</title>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace <title>...</title>")  # noqa: TRY003
    return replaced


def _inject_rtl_css(html: str) -> str:
    rtl_css = """
        html[dir="rtl"] body {
            direction: rtl;
            text-align: right;
        }
        html[dir="rtl"] ul {
            margin-left: 0;
            margin-right: 22px;
        }
"""
    replaced, count = re.subn(r"</style>", rtl_css + "\n</style>", html, flags=re.IGNORECASE, count=1)
    if count != 1:
        raise RuntimeError("Could not inject RTL CSS")  # noqa: TRY003
    return replaced


def _translate_page(source_html: str, target_lang: str) -> str:
    title_en_match = re.search(r"<title>(.*?)</title>", source_html, flags=re.IGNORECASE | re.DOTALL)
    if not title_en_match:
        raise RuntimeError("Could not find <title>")  # noqa: TRY003
    title_en = title_en_match.group(1).strip()

    header_en = _extract_inner(source_html, "header")
    main_en = _extract_inner(source_html, "main")
    footer_en = _extract_inner(source_html, "footer")

    if target_lang == "en":
        out = source_html
        out = _replace_html_tag(out, target_lang)
        return out

    title_part: str
    header_part: str
    main_part: str
    footer_part: str

    try:
        section_0 = "AICASSOSECTIONBREAK0"
        section_1 = "AICASSOSECTIONBREAK1"
        section_2 = "AICASSOSECTIONBREAK2"

        hr_token = "AICASSO_HR_TOKEN_0"
        br_token = "AICASSO_BR_TOKEN_0"

        main_pre, hr_count = re.subn(r"<hr\s*/?>", hr_token, main_en, flags=re.IGNORECASE)
        main_pre, br_count = re.subn(r"<br\s*/?>", br_token, main_pre, flags=re.IGNORECASE)

        combined = "\n\n".join(
            [
                title_en,
                section_0,
                header_en,
                section_1,
                main_pre,
                section_2,
                footer_en,
            ]
        )

        translated = _translate_en_to(combined, target_lang)

        has_markers = (
            translated.count(section_0) == 1
            and translated.count(section_1) == 1
            and translated.count(section_2) == 1
        )

        if has_markers:
            title_part, rest = translated.split(section_0, 1)
            header_part, rest = rest.split(section_1, 1)
            main_part, footer_part = rest.split(section_2, 1)
        else:
            title_part = _translate_en_to(title_en, target_lang)
            header_part = _translate_en_to(header_en, target_lang)
            main_part = _translate_en_to(main_pre, target_lang)
            footer_part = _translate_en_to(footer_en, target_lang)

        title_part = title_part.strip()
        header_part = header_part.strip()
        main_part = main_part.strip()
        footer_part = footer_part.strip()

        if hr_count and main_part.count(hr_token) != hr_count:
            raise RuntimeError("HR marker missing from translated main block")  # noqa: TRY003
        if br_count and main_part.count(br_token) != br_count:
            raise RuntimeError("BR marker missing from translated main block")  # noqa: TRY003

        main_part = re.sub(rf"\b{re.escape(hr_token)}\b", "<hr>", main_part)
        main_part = re.sub(rf"\b{re.escape(br_token)}\b", "<br>", main_part)
    except Exception:
        title_part = _translate_text_fallback(title_en, target_lang).strip()
        header_part = _translate_html_fragment_fallback(header_en, target_lang).strip()
        main_part = _translate_html_fragment_fallback(main_en, target_lang).strip()
        footer_part = _translate_text_fallback(footer_en, target_lang).strip()

    out = source_html
    out = _replace_html_tag(out, target_lang)
    out = _replace_title(out, title_part)
    out = _replace_inner(out, "header", header_part)
    out = _replace_inner(out, "main", main_part)
    out = _replace_inner(out, "footer", footer_part)

    if target_lang in RTL_LANGS:
        out = _inject_rtl_css(out)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate updated policy pages into all docs languages.")
    parser.add_argument(
        "--docs-dir",
        default=str(Path(__file__).resolve().parents[1] / "docs"),
        help="Path to docs directory (default: repo/docs).",
    )
    parser.add_argument(
        "--langs",
        default="",
        help="Comma-separated language codes to update (default: all under docs/, excluding images and en).",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.is_dir():
        raise RuntimeError(f"Docs dir not found: {docs_dir}")  # noqa: TRY003

    source_terms = docs_dir / "en" / "TermsOfUse.html"
    source_privacy = docs_dir / "en" / "PrivacyPolicy.html"
    if not source_terms.is_file() or not source_privacy.is_file():
        raise RuntimeError("Missing English source pages under docs/en")  # noqa: TRY003

    terms_html = source_terms.read_text(encoding="utf-8")
    privacy_html = source_privacy.read_text(encoding="utf-8")

    all_langs = sorted(
        [
            p.name
            for p in docs_dir.iterdir()
            if p.is_dir() and p.name not in {"images"} and (p / "TermsOfUse.html").is_file()
        ]
    )
    if args.langs.strip():
        target_langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    else:
        target_langs = [l for l in all_langs if l != "en"]

    for lang in target_langs:
        target_dir = docs_dir / lang
        if not target_dir.is_dir():
            raise RuntimeError(f"Language dir not found: {target_dir}")  # noqa: TRY003

        print(f"Translating policy pages -> {lang}")

        translated_terms = _translate_page(terms_html, lang)
        time.sleep(0.25)
        translated_privacy = _translate_page(privacy_html, lang)
        time.sleep(0.25)

        (target_dir / "TermsOfUse.html").write_text(translated_terms, encoding="utf-8", newline="\n")
        (target_dir / "PrivacyPolicy.html").write_text(translated_privacy, encoding="utf-8", newline="\n")

        time.sleep(0.25)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
