"""
Populate Utility
================
Shared RAG population logic for the final ablations.

Given:
  - An index directory
  - A final evaluation dataset
  - A retrieval_mode and top_k

Produces a populated JSON file with generated_answer and retrieved_chunks
for every question. Supports resume from partial progress.

This module embeds a lean GeminiKeyRotator rather than importing from
evaluationV2/gemini_key_rotator.py so it picks up is_invalid_key_error.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Set

import google.generativeai as genai

# ── Resolve backend ──────────────────────────────────────────────────────────
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _EXPERIMENTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from shared.vector_store_experiment import VectorStoreExperiment


def index_manifest_fingerprint(index_dir: str) -> str:
    """Fingerprint the exact index build used for a populated result."""
    manifest_path = Path(index_dir) / "faiss_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def dataset_fingerprint(eval_data_path: str) -> str:
    """Fingerprint the immutable question/label file used for a run."""
    path = Path(eval_data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieval_implementation_fingerprint() -> str:
    """Fingerprint code that changes retrieval results without rebuilding FAISS."""
    paths = [
        _EXPERIMENTS_DIR / "shared" / "vector_store_experiment.py",
        _EXPERIMENTS_DIR / "shared" / "source_router.py",
        _BACKEND_DIR / "app" / "rag" / "vector_store_improved.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(_BACKEND_DIR)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def populated_matches_index(
    output_path: str,
    index_dir: str,
    eval_data_path: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    top_k: Optional[int] = None,
) -> bool:
    """Return True only if completed items cite the same immutable run inputs."""
    path = Path(output_path)
    if not path.is_file():
        return False
    expected = index_manifest_fingerprint(index_dir)
    expected_dataset = dataset_fingerprint(eval_data_path) if eval_data_path else None
    expected_retrieval = retrieval_implementation_fingerprint()
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    completed = [item for item in items if item.get("generated_answer")]
    def matches(item: Dict) -> bool:
        metadata = item.get("run_metadata", {})
        return (
            metadata.get("index_fingerprint") == expected
            and metadata.get("retrieval_implementation_fingerprint") == expected_retrieval
            and (expected_dataset is None or metadata.get("dataset_fingerprint") == expected_dataset)
            and (retrieval_mode is None or metadata.get("retrieval_mode") == retrieval_mode)
            and (top_k is None or metadata.get("top_k") == top_k)
        )
    return bool(completed) and all(matches(item) for item in completed)


# ── Inline GeminiKeyRotator (matches populate_eval_data.py variant) ──────────
class _GeminiKeyRotator:
    """Manage multiple Gemini API keys with rotation on quota / invalid-key errors."""

    def __init__(self, env_file: Optional[Path] = None):
        if env_file is None:
            env_file = _BACKEND_DIR / ".env"
        self.api_keys = self._load_api_keys(Path(env_file))
        self.current_index = 0
        self.failed_keys: Set[int] = set()
        if not self.api_keys:
            raise ValueError("No Gemini API keys found in .env file")
        print(f"🔑 Loaded {len(self.api_keys)} Gemini API key(s)")
        self._configure()

    def _load_api_keys(self, env_file: Path) -> List[str]:
        if not env_file.exists():
            raise FileNotFoundError(f".env not found: {env_file}")
        keys: List[str] = []
        seen: Set[str] = set()
        for line in env_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r'GEMINI_API_KEYS\s*=\s*(.+)', line)
            if m:
                for k in m.group(1).split(','):
                    k = k.strip().strip('"\'')
                    if len(k) > 20 and k not in seen:
                        keys.append(k); seen.add(k)
                continue
            m = re.match(r'GEMINI_API_KEY\w*\s*=\s*(\S+)', line)
            if m:
                k = m.group(1).strip().strip('"\'')
                if len(k) > 20 and k not in seen:
                    keys.append(k); seen.add(k)
        return keys

    def _configure(self):
        genai.configure(api_key=self.api_keys[self.current_index])

    def get_model(self, name: str = 'gemini-3.5-flash') -> genai.GenerativeModel:
        return genai.GenerativeModel(name)

    def rotate_key(self) -> bool:
        self.failed_keys.add(self.current_index)
        for i in range(len(self.api_keys)):
            nxt = (self.current_index + 1 + i) % len(self.api_keys)
            if nxt not in self.failed_keys:
                self.current_index = nxt
                self._configure()
                print(f"🔄 Rotated to key {self.current_index + 1}/{len(self.api_keys)}")
                return True
        print("❌ All Gemini keys exhausted")
        return False

    def reset_failed_keys(self):
        self.failed_keys.clear()
        print("🔄 Reset failed keys")

    def is_rate_limit_error(self, e: Exception) -> bool:
        s = str(e).lower()
        return any(t in s for t in [
            'rate limit', 'quota', 'resource exhausted',
            'too many requests', '429', 'resourceexhausted',
        ])

    def is_invalid_key_error(self, e: Exception) -> bool:
        s = str(e).lower()
        return any(t in s for t in [
            'api key not valid', 'api key not found', 'api_key_invalid',
            'invalid api key', 'permission_denied', 'unauthenticated',
            'account_state_invalid', 'deleted or disabled',
            '400', '401', '403',
        ])


# ── Prompt builder ─────────────────────────────────────────────────────────────
def _build_prompt(query: str, context_chunks: List) -> str:
    chunks_by_key = defaultdict(list)

    for chunk in context_chunks:
        if isinstance(chunk, dict):
            dafa      = chunk.get('dafa', 'Unknown')
            text      = chunk.get('text', '')
            metadata  = chunk.get('metadata', {})
            act_name  = metadata.get('act_name', chunk.get('act_name', ''))
            bhag      = metadata.get('bhag', chunk.get('bhag', ''))
            parichhed = metadata.get('parichhed', chunk.get('parichhed', ''))
        else:
            dafa      = getattr(chunk, 'dafa', 'Unknown')
            text      = getattr(chunk, 'text', '')
            meta      = getattr(chunk, 'metadata', {})
            act_name  = meta.get('act_name', '')
            bhag      = meta.get('bhag', '')
            parichhed = meta.get('parichhed', '')

        chunks_by_key[(act_name, bhag, parichhed, dafa)].append(text)

    context_parts = []
    for (act_name, bhag, parichhed, dafa), texts in sorted(chunks_by_key.items()):
        combined = "\n".join(texts)
        label_parts = [p for p in [act_name, bhag, parichhed] if p]
        label_parts.append(f"दफा {dafa}")
        label = " | ".join(label_parts)
        context_parts.append(f"[{label}]\n{combined}")

    context = "\n\n".join(context_parts)

    return f"""तपाईं एक नेपाली कानूनी सल्लाहकार हुनुहुन्छ जसले **केवल र केवल** तल दिइएको कानूनी सन्दर्भबाट मात्र जवाफ दिन्छ।

**अनिवार्य नियमहरू (CRITICAL RULES):**

⚠️ **तपाईंको आफ्नै ज्ञान प्रयोग गर्न निषेध छ** - केवल तलका सन्दर्भहरूबाट मात्र उत्तर दिनुहोस्
⚠️ **प्रत्येक तथ्यसँग दफा नम्बर र ऐनको नाम अनिवार्य छ**
⚠️ **सन्दर्भमा सीधै जवाफ नभएमा** - तार्किक निष्कर्ष निकाल्न सकिन्छ भने सो स्पष्ट पारेर उत्तर दिनुहोस्। कुनै सान्दर्भिक जानकारी नभएमा भन्नुहोस्: \"माफ गर्नुहोस्, यो जानकारी उपलब्ध कानूनी सन्दर्भमा छैन।\"

**प्रश्न:** {query}

**कानूनी सन्दर्भहरू:**
{context}

**उत्तर:**"""


# ── Generator with key rotation ───────────────────────────────────────────────
class RotatingGenerator:
    def __init__(self, key_rotator: _GeminiKeyRotator, model_name: str = 'gemini-3.5-flash'):
        self.key_rotator = key_rotator
        self.model_name  = model_name
        self.model       = key_rotator.get_model(model_name)

    async def generate(
        self,
        query: str,
        context_chunks: List,
        max_retries: int = 1,
        retry_delay: float = 2.0,
        request_timeout: float = 60.0,
    ) -> str:
        prompt = _build_prompt(query, context_chunks)

        while True:
            for attempt in range(max_retries):
                try:
                    response = await asyncio.wait_for(
                        self.model.generate_content_async(prompt),
                        timeout=request_timeout,
                    )
                    return response.text
                except asyncio.TimeoutError:
                    print(
                        f"   Request timed out after {request_timeout:g}s "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    elif self.key_rotator.rotate_key():
                        self.model = self.key_rotator.get_model(self.model_name)
                        break
                    else:
                        raise RuntimeError("All Gemini keys timed out")
                except Exception as e:
                    if self.key_rotator.is_rate_limit_error(e):
                        print(f"   ⚠️ Rate limit (attempt {attempt+1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                        else:
                            if self.key_rotator.rotate_key():
                                self.model = self.key_rotator.get_model(self.model_name)
                                break
                            else:
                                print("   ⏳ All keys exhausted, waiting 60s…")
                                await asyncio.sleep(60)
                                self.key_rotator.reset_failed_keys()
                                self.model = self.key_rotator.get_model(self.model_name)
                                break
                    elif self.key_rotator.is_invalid_key_error(e):
                        print(f"   ❌ Invalid key, rotating…")
                        if self.key_rotator.rotate_key():
                            self.model = self.key_rotator.get_model(self.model_name)
                            break
                        else:
                            raise
                    else:
                        raise
            else:
                continue


# ── Main populate function ─────────────────────────────────────────────────────
async def populate(
    eval_data_path: str,
    output_path: str,
    index_dir: str,
    retrieval_mode: str = 'hybrid',
    top_k: int = 5,
    skip_oos: bool = False,
    dry_run: int = 0,
    restart: bool = False,
) -> List[Dict]:
    """
    Populate an eval dataset with generated answers and retrieved chunks.

    Args:
        eval_data_path : Path to the selected LangSmith-compatible V2 dataset.
        output_path    : Path where the populated JSON will be saved.
        index_dir      : Directory with the strategy-specific index artifacts.
        retrieval_mode : 'hybrid' | 'faiss_only' | 'bm25_only' |
                         'routed_hybrid'
        top_k          : Number of chunks to retrieve per question.
        skip_oos       : Skip out_of_scope questions (default True).
        dry_run        : If > 0, only process that many questions (for testing).
        restart        : Ignore an existing output and start from source data.

    Returns:
        Populated eval data list.
    """
    out_path = Path(output_path)
    in_path  = Path(eval_data_path)

    print(f"📂 Loading eval data: {in_path}")
    with open(in_path, 'r', encoding='utf-8') as f:
        eval_data: List[Dict] = json.load(f)
    print(f"✅ {len(eval_data)} questions loaded")
    index_fingerprint = index_manifest_fingerprint(index_dir)
    eval_fingerprint = dataset_fingerprint(str(in_path))
    retrieval_fingerprint = retrieval_implementation_fingerprint()

    # Resume from partial progress
    if out_path.exists() and not restart:
        print("📂 Found existing output, checking for resume…")
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        already_done = sum(
            1 for item in existing
            if item.get("generated_answer") and not str(item["generated_answer"]).startswith("ERROR")
        )
        if already_done > 0:
            stale = any(
                item.get("generated_answer") and (
                    item.get("run_metadata", {}).get("index_fingerprint") != index_fingerprint
                    or item.get("run_metadata", {}).get("dataset_fingerprint") != eval_fingerprint
                    or item.get("run_metadata", {}).get("retrieval_implementation_fingerprint") != retrieval_fingerprint
                    or item.get("run_metadata", {}).get("retrieval_mode") != retrieval_mode
                    or item.get("run_metadata", {}).get("top_k") != top_k
                )
                for item in existing
            )
            if stale:
                raise RuntimeError(
                    "Existing populated output was produced from a different "
                    "or unrecorded index build. Re-run this command with "
                    "--restart; stale answers cannot be resumed safely."
                )
            print(f"   {already_done} items already populated — resuming")
            eval_data = existing
    elif out_path.exists():
        print("♻️  Restart requested; existing populated output will be replaced")

    # Load index
    print(f"\n🔄 Loading index from: {index_dir}")
    vs = VectorStoreExperiment(index_dir)
    vs.load()

    # Init generator
    key_rotator = _GeminiKeyRotator(env_file=_BACKEND_DIR / ".env")
    generator   = RotatingGenerator(key_rotator)
    print(f"✅ Generator ready  (retrieval_mode={retrieval_mode}, top_k={top_k})")

    processed = 0
    for i, item in enumerate(eval_data):
        qid      = item.get('id', f'q_{i+1}')
        category = item.get('category', '')
        question = item.get('question', '')

        # Skip OOS
        if skip_oos and category == 'out_of_scope':
            print(f"[{i+1}] ⏭️ Skipping OOS: {qid}")
            continue

        # Skip already done
        existing_ans = item.get('generated_answer', '')
        if existing_ans and not str(existing_ans).startswith('ERROR'):
            print(f"[{i+1}] ⏭️ Already populated: {qid}")
            continue

        # Dry-run limit
        if dry_run > 0 and processed >= dry_run:
            print(f"\n🛑 Dry-run limit ({dry_run}) reached — stopping")
            break

        print(f"\n[{i+1}/{len(eval_data)}] Processing: {question[:55]}…")

        try:
            results = await vs.search(question, top_k=top_k,
                                      retrieval_mode=retrieval_mode)

            retrieved_chunks = []
            for r in results:
                if isinstance(r, dict):
                    retrieved_chunks.append(r)
                else:
                    retrieved_chunks.append({
                        'text':     getattr(r, 'text', ''),
                        'dafa':     getattr(r, 'dafa', ''),
                        'score':    float(getattr(r, 'score', 0)),
                        'metadata': getattr(r, 'metadata', {}),
                        # also store dafa_list for metrics
                        'dafa_list': getattr(
                            getattr(r, 'metadata', {}), 'get', lambda k,d=None: d
                        )('dafa_list') or [],
                    })

            # Enrich chunk dicts with dafa_list from the index for metrics
            for rc, orig_result in zip(retrieved_chunks, results):
                if not isinstance(orig_result, dict):
                    cid_str = getattr(orig_result, 'chunk_id', None)
                    if cid_str is not None:
                        try:
                            cid_int = int(cid_str) - 1
                            if 0 <= cid_int < len(vs.chunks):
                                chunk_obj = vs.chunks[cid_int]
                                rc['dafa_list'] = chunk_obj.get('dafa_list', [])
                                rc['source']     = chunk_obj.get('source', '')
                                rc['act_name']  = chunk_obj.get('act_name', '')
                                rc['bhag']      = chunk_obj.get('bhag', '')
                                rc['parichhed'] = chunk_obj.get('parichhed', '')
                        except (ValueError, IndexError):
                            pass

            generated = await generator.generate(question, results)

            item['generated_answer']   = generated
            item['retrieved_chunks']   = retrieved_chunks
            item['retrieval_mode']     = retrieval_mode
            item['run_metadata'] = {
                'num_chunks_retrieved': len(retrieved_chunks),
                'top_k':                top_k,
                'retrieval_mode':       retrieval_mode,
                'index_dir':            str(index_dir),
                'index_fingerprint':    index_fingerprint,
                'dataset_fingerprint':  eval_fingerprint,
                'retrieval_implementation_fingerprint': retrieval_fingerprint,
            }
            print(f"   ✅ {len(generated)} chars, {len(retrieved_chunks)} chunks")
            processed += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")
            item['generated_answer'] = f"ERROR: {e}"
            item['retrieved_chunks'] = []
            item['run_metadata']     = {
                'error': str(e),
                'index_fingerprint': index_fingerprint,
                'dataset_fingerprint': eval_fingerprint,
                'retrieval_implementation_fingerprint': retrieval_fingerprint,
                'retrieval_mode': retrieval_mode,
                'top_k': top_k,
            }

        # Save progress every 5 items
        if (i + 1) % 5 == 0:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)
            print(f"   💾 Progress saved ({i+1}/{len(eval_data)})")

    # Final save
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    success = sum(1 for item in eval_data
                  if item.get('generated_answer')
                  and not str(item['generated_answer']).startswith('ERROR'))
    print(f"\n✅ Population complete — {success}/{len(eval_data)} successful")
    print(f"💾 Output: {out_path}")
    return eval_data
