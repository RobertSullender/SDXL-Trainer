"""Offline spell checking with Ubuntu's Enchant dictionaries, without Python bindings."""
from __future__ import annotations

import ctypes
import ctypes.util
import re
import threading


class SpellCheckUnavailable(RuntimeError):
    pass


class SpellChecker:
    """Use the installed Enchant/Hunspell dictionary; never download dictionaries.

    ``available`` and ``reason`` let the editor show a clear unavailable status.
    Correctness/suggestions raise if unavailable instead of silently accepting words.
    ``ignore`` applies only to this app instance, useful for a dataset's trigger words.
    """

    def __init__(self, language: str = "en_US"):
        self.language = language
        self.available = False
        self.reason = ""
        self._lib = None
        self._broker = None
        self._dictionary = None
        self._lock = threading.RLock()
        self._ignored: set[str] = set()
        path = ctypes.util.find_library("enchant-2")
        if not path:
            self.reason = "Offline spell checking needs Ubuntu's libenchant-2-2 and a Hunspell dictionary such as hunspell-en-us."
            return
        try:
            lib = ctypes.CDLL(path)
            self._lib = lib
            lib.enchant_broker_init.argtypes = []
            lib.enchant_broker_init.restype = ctypes.c_void_p
            lib.enchant_broker_free.argtypes = [ctypes.c_void_p]
            lib.enchant_broker_free.restype = None
            lib.enchant_broker_request_dict.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.enchant_broker_request_dict.restype = ctypes.c_void_p
            lib.enchant_broker_free_dict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.enchant_broker_free_dict.restype = None
            lib.enchant_dict_check.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ssize_t]
            lib.enchant_dict_check.restype = ctypes.c_int
            lib.enchant_dict_suggest.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ssize_t, ctypes.POINTER(ctypes.c_size_t)]
            lib.enchant_dict_suggest.restype = ctypes.POINTER(ctypes.c_char_p)
            lib.enchant_dict_free_string_list.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
            lib.enchant_dict_free_string_list.restype = None
            self._broker = lib.enchant_broker_init()
            if not self._broker:
                self.reason = "The system spell-check broker could not start."
                return
            self._dictionary = lib.enchant_broker_request_dict(self._broker, language.encode("utf-8"))
            if not self._dictionary:
                self.reason = f"The offline {language} dictionary is not installed. Install its Ubuntu Hunspell dictionary."
                self.close()
                return
            self.available = True
        except (OSError, AttributeError, UnicodeError) as exc:
            self.reason = f"The system offline spell checker is unavailable: {exc}"
            self.close()

    def _require(self):
        if not self.available:
            raise SpellCheckUnavailable(self.reason or "The offline spell checker is closed.")

    def check(self, word: str) -> bool:
        with self._lock:
            self._require()
            if not word or word.casefold() in self._ignored:
                return True
            encoded = word.encode("utf-8")
            result = self._lib.enchant_dict_check(self._dictionary, encoded, len(encoded))
            if result < 0:
                raise RuntimeError("The offline dictionary could not check this word.")
            return result == 0

    def suggest(self, word: str) -> list[str]:
        with self._lock:
            self._require()
            if not word:
                return []
            encoded = word.encode("utf-8")
            count = ctypes.c_size_t(0)
            suggestions = self._lib.enchant_dict_suggest(self._dictionary, encoded, len(encoded), ctypes.byref(count))
            if not suggestions:
                return []
            try:
                return [suggestions[i].decode("utf-8") for i in range(min(count.value, 30))]
            finally:
                self._lib.enchant_dict_free_string_list(self._dictionary, suggestions)

    def ignore(self, word: str):
        self._ignored.add(word.casefold())

    def misspellings(self, text: str) -> list[tuple[int, int, str]]:
        """Return Python string offsets; Qt callers convert offsets to UTF-16 when needed."""
        with self._lock:
            self._require()
            matches = re.finditer(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, re.UNICODE)
            return [(match.start(), len(match.group()), match.group())
                    for match in matches if not self.check(match.group())]

    def close(self):
        with self._lock:
            self.available = False
            if self._dictionary and self._lib and self._broker:
                self._lib.enchant_broker_free_dict(self._broker, self._dictionary)
                self._dictionary = None
            if self._broker and self._lib:
                self._lib.enchant_broker_free(self._broker)
                self._broker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self._require()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
