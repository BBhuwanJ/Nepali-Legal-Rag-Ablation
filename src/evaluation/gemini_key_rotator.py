#!/usr/bin/env python
"""
Gemini API Key Rotator for Evaluations
======================================
Manages multiple Gemini API keys with automatic rotation on rate limit errors.
Uses synchronous calls for evaluation (not async like populate_eval_data.py).
"""
import os
import re
import time
import threading
from pathlib import Path
from typing import List, Optional, Set
import google.generativeai as genai


class GeminiKeyRotator:
    """
    Manages multiple Gemini API keys with automatic rotation on rate limit errors.
    
    Keys are loaded from .env file in formats:
        GEMINI_API_KEY=AIzaSy...
        AIzaSy...  (bare keys after the main one)
    """
    _instance = None
    _lock = threading.Lock()
    DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 12.0

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GeminiKeyRotator, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize the key rotator.
        
        Args:
            env_file: Path to .env file. Defaults to backend/.env
        """
        if self._initialized:
            return
            
        if env_file is None:
            env_file = Path(__file__).parent.parent / ".env"
        
        self.api_keys = self._load_api_keys(env_file)
        self.current_index = 0
        self.failed_keys: Set[int] = set()
        self._request_lock = threading.Lock()
        self._last_successful_request_started_at: Optional[float] = None
        try:
            configured_interval = float(
                os.getenv(
                    "GEMINI_MIN_REQUEST_INTERVAL_SECONDS",
                    str(self.DEFAULT_MIN_REQUEST_INTERVAL_SECONDS),
                )
            )
        except ValueError as exc:
            raise ValueError(
                "GEMINI_MIN_REQUEST_INTERVAL_SECONDS must be a number"
            ) from exc
        self.min_request_interval_seconds = max(0.0, configured_interval)
        
        if not self.api_keys:
            raise ValueError("No Gemini API keys found in .env file")
        
        print(f"🔑 Loaded {len(self.api_keys)} Gemini API key(s) for evaluation")
        self._configure_current_key()
        self._initialized = True
    
    def _load_api_keys(self, env_file: Path) -> List[str]:
        """Load all Gemini API keys from .env file.
        
        Supports:
            GEMINI_API_KEYS=key1,key2,...   (comma-separated list)
            GEMINI_API_KEY=singlekey        (single key)
            GEMINI_API_KEY_2=key            (numbered variants)
        """
        if not env_file.exists():
            return []
        
        api_keys = []
        seen = set()
        
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Handle GEMINI_API_KEYS=key1,key2,...  (comma-separated list)
            match = re.match(r'GEMINI_API_KEYS\s*=\s*(.+)', line)
            if match:
                for key in match.group(1).split(','):
                    key = key.strip().strip('"\'')
                    if len(key) > 20 and key not in seen:
                        api_keys.append(key)
                        seen.add(key)
                continue
            # Handle GEMINI_API_KEY=singlekey or GEMINI_API_KEY_N=key
            match = re.match(r'GEMINI_API_KEY\w*\s*=\s*(\S+)', line)
            if match:
                key = match.group(1).strip().strip('"\'')
                if len(key) > 20 and key not in seen:
                    api_keys.append(key)
                    seen.add(key)
        
        return api_keys
    
    def _configure_current_key(self):
        """Configure genai with the current API key."""
        current_key = self.api_keys[self.current_index]
        genai.configure(api_key=current_key)
    
    def get_current_key(self) -> str:
        """Get the current API key (for debugging)."""
        return self.api_keys[self.current_index]
    
    def get_model(self, model_name: str = "gemini-3.5-flash") -> genai.GenerativeModel:
        """Get a GenerativeModel instance."""
        return genai.GenerativeModel(model_name)

    def generate_content(
        self,
        prompt: str,
        model_name: str = "gemini-3.5-flash",
        retry_reset_delay: float = 60.0,
        **generate_kwargs,
    ):
        """Generate content while enforcing the shared free-tier RPM limit.

        ``google.generativeai.configure`` is process-global, and this rotator is a
        singleton shared by all four evaluators.  The lock therefore protects both
        API-key configuration and the request itself. After a successful request,
        the next request starts at least 12 seconds after that successful request
        started (5 RPM by default). Rate-limit and quota failures rotate to the next
        available key immediately without waiting or retrying the failed key.
        """
        # Keep one worker in control through key rotation. This prevents
        # queued combined-evaluation workers from issuing requests with a key that
        # has just failed but has not yet been rotated.
        with self._request_lock:
            while True:
                if self._last_successful_request_started_at is not None:
                    elapsed = (
                        time.monotonic()
                        - self._last_successful_request_started_at
                    )
                    wait_seconds = self.min_request_interval_seconds - elapsed
                    if wait_seconds > 0:
                        print(
                            f"   ⏳ Waiting {wait_seconds:.1f}s "
                            f"(Gemini free tier: 5 RPM)...",
                            flush=True,
                        )
                        time.sleep(wait_seconds)

                with self._lock:
                    current_key_index = self.current_index
                    # Reconfigure and create the model while protected from other
                    # evaluator threads; the SDK configuration is process-global.
                    self._configure_current_key()
                    model = self.get_model(model_name)

                print(
                    f"   🔑 Using key {current_key_index + 1}/"
                    f"{len(self.api_keys)}",
                    end="",
                    flush=True,
                )
                request_started_at = time.monotonic()

                try:
                    response = model.generate_content(prompt, **generate_kwargs)
                    self._last_successful_request_started_at = request_started_at
                    print(" ✓", flush=True)
                    return response
                except Exception as exc:
                    print(" ✗", flush=True)
                    if not self.is_rate_limit_error(exc):
                        raise

                    print(
                        "   ⚠️ Rate limit/quota error. "
                        "Rotating key immediately...",
                        flush=True,
                    )

                    if self.rotate_key(failed_index=current_key_index):
                        continue

                    print(
                        f"   ⏳ All keys exhausted. Waiting "
                        f"{retry_reset_delay:g}s...",
                        flush=True,
                    )
                    time.sleep(retry_reset_delay)
                    self.reset_failed_keys()
    
    def rotate_key(self, failed_index: Optional[int] = None) -> bool:
        """
        Rotate to the next available API key.
        
        Args:
            failed_index: The index of the key that failed. If the current index has
                          already been rotated by another thread, this will be a no-op.
        
        Returns:
            True if successfully rotated to a new key, False if no more keys available
        """
        with self._lock:
            # If another thread already rotated the key after this thread failed, just return True
            if failed_index is not None and failed_index != self.current_index:
                return True
                
            self.failed_keys.add(self.current_index)
            
            # Find next available key
            for i in range(len(self.api_keys)):
                next_index = (self.current_index + 1 + i) % len(self.api_keys)
                if next_index not in self.failed_keys:
                    self.current_index = next_index
                    self._configure_current_key()
                    print(f"\n🔄 Rotated to Gemini API key {self.current_index + 1}/{len(self.api_keys)}")
                    return True
            
            print("\n❌ All Gemini API keys exhausted!")
            return False
    
    def reset_failed_keys(self):
        """Reset failed keys (useful after waiting for quota reset)."""
        with self._lock:
            self.failed_keys.clear()
            print("\n🔄 Reset failed keys - all Gemini keys available again")
    
    def has_available_keys(self) -> bool:
        """Check if there are any available keys left."""
        return len(self.failed_keys) < len(self.api_keys)
    
    def get_key_status(self) -> str:
        """Get a status string showing key usage."""
        with self._lock:
            available = len(self.api_keys) - len(self.failed_keys)
            return f"{available}/{len(self.api_keys)} keys available"
    
    def is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit, quota, or invalid key error."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit',
            'rate_limit',
            'quota exceeded',
            'quota',
            'resource exhausted',
            'resourceexhausted',
            'too many requests',
            '429',
            'api_key_invalid',
            'api key not found',
            'invalid api key',
            'unauthenticated',
            'account_state_invalid',
            'deleted or disabled',
            'permission_denied',
            '401',
            '403',
            'timeout',
            'deadline',
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)


class GeminiEvaluator:
    """
    Base class for Gemini-based evaluation with automatic key rotation.
    """
    
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        """
        Initialize the Gemini evaluator.
        
        Args:
            model_name: Gemini model to use for evaluation
        """
        self.key_rotator = GeminiKeyRotator()
        self.model_name = model_name
        self.model = self.key_rotator.get_model(model_name)
        self._last_key_index = self.key_rotator.current_index
    
    def _make_api_call(self, prompt: str, max_retries: int = 2, retry_delay: float = 2.0) -> str:
        """
        Make API call with automatic key rotation on quota errors.
        
        Args:
            prompt: The prompt to send
            max_retries: Deprecated compatibility argument; rate-limit and quota
                         failures now rotate immediately
            retry_delay: Deprecated compatibility argument; successful-request
                         pacing is controlled by
                         GEMINI_MIN_REQUEST_INTERVAL_SECONDS
            
        Returns:
            Generated text response
        """
        # Compatibility arguments are retained for external callers. The shared
        # rotator owns successful-request pacing and immediate key rotation.
        response = self.key_rotator.generate_content(
            prompt,
            model_name=self.model_name,
        )
        return response.text


if __name__ == "__main__":
    # Test the rotator
    rotator = GeminiKeyRotator()
    print(f"Status: {rotator.get_key_status()}")
    
    # Test a simple call
    model = rotator.get_model()
    try:
        response = model.generate_content("Say 'Hello, evaluation!' in exactly those words.")
        print(f"✅ Test response: {response.text[:50]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")
