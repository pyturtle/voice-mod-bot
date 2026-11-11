import sys
import os
from pathlib import Path
import io
import shlex

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']
    
    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value

set_cuda_paths()

import re
import json
import asyncio
import time
import logging
from collections import defaultdict
import numpy as np
import soundfile as sf
import librosa

import discord
from discord.ext import commands
from dotenv import load_dotenv

from faster_whisper import WhisperModel

# whisper_online pulls in torch; imported lazily below only when USE_ONLINE_VAD is enabled.
VACOnlineASRProcessor = None

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in .env")

ENABLE_SOUNDBOARD_SFX = os.getenv("ENABLE_SOUNDBOARD_SFX", "true").lower() in {"1", "true", "yes", "on"}
GROUP1_SOUNDBOARD_SOUND_ID = os.getenv("GROUP1_SOUNDBOARD_SOUND_ID")
GROUP2_SOUNDBOARD_SOUND_ID = os.getenv("GROUP2_SOUNDBOARD_SOUND_ID")
GROUP3_SOUNDBOARD_SOUND_ID = os.getenv("GROUP3_SOUNDBOARD_SOUND_ID")
USE_ONLINE_VAD = os.getenv("USE_ONLINE_VAD", "false").lower() in {"1", "true", "yes", "on"}
WHISPER_LANGUAGE = "en"  # set to None to auto-detect, or language code like "en"

STATE_PATH = Path("state.json")
TERMS_PATH = Path("terms.txt")
TERMS1_PATH = Path("terms1.txt")
TERMS2_PATH = Path("terms2.txt")
TERMS3_PATH = Path("terms3.txt")
BIAS_TERMS_PATH = Path("bias_terms.txt")
SOUNDBOARD_CACHE = {}
SOUND_ON_WORDS = {"sound", "sounds", "sfx", "on", "enable", "enabled", "true", "yes", "1"}
SOUND_OFF_WORDS = {"nosound", "mute", "muted", "off", "disable", "disabled", "false", "no", "0", "silent"}

# -------- Logging --------
DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("voice-mod-bot")
# Quiet Opus decode warnings that originate from malformed/empty RTP packets.
logging.getLogger("discord.opus").setLevel(logging.WARNING)


def _install_dave_decrypt_fix():
    """Keep the transport-decrypted audio when DAVE decryption doesn't apply.

    py-cord 2.8's PacketDecryptor.decrypt_rtp decrypts the RTP transport layer into
    plain Opus, then tries a DAVE (E2EE) decrypt on top. When that second step raises
    -- which it does constantly with DecryptionFailed(UnencryptedWhenPassthroughDisabled),
    because the packets simply aren't E2EE -- the handler replaces the payload with
    OPUS_SILENCE, throwing away the good audio it already had. It also never assigns
    decrypted_data at all when there's no DAVE session.

    Decoding that empty payload makes Opus synthesise packet-loss concealment noise,
    which is why captured audio measured as noise (zero-crossing rate ~0.31 vs ~0.12
    for real speech). This restores the fallback their own commented-out version had.
    """
    from discord.voice.receive import reader as _reader

    cls = getattr(_reader, "PacketDecryptor", None)
    if cls is None or getattr(cls, "_voicemod_dave_fixed", False):
        return
    try:
        import davey
    except ImportError:
        davey = None

    def decrypt_rtp(self, packet):
        state = self.client._connection
        dave = state.dave_session

        data = self._decryptor_rtp(packet)

        if davey is not None and dave is not None and dave.ready:
            uid = state.ssrc_user_map.get(packet.ssrc)
            if uid:
                try:
                    data = dave.decrypt(uid, davey.MediaType.audio, data)
                except Exception:
                    # Unencrypted / passthrough-disabled packets land here. The
                    # transport-decrypted payload is already plain Opus, so keep it.
                    pass

        # No extension-header handling here: _decryptor_rtp already calls
        # update_extended_header and returns the payload past it. Stripping again
        # would cut real Opus bytes off the front of every packet.
        packet.decrypted_data = data
        return data

    cls.decrypt_rtp = decrypt_rtp
    cls._voicemod_dave_fixed = True
    logger.info("Installed RTP decrypt fallback fix")


def _install_decoder_teardown_fix():
    """Iterate over a copy when destroying decoders.

    py-cord's PacketRouter.destroy_all_decoders loops over self.decoders.keys() while
    destroy_decoder pops from that same dict, so it raises "dictionary changed size
    during iteration" whenever more than one decoder exists. It runs from
    on_voice_state_update, so the crash aborts the reconnect handler and a recoverable
    voice drop ends the monitoring session instead.
    """
    from discord.voice.receive import router as _router

    cls = getattr(_router, "PacketRouter", None)
    if cls is None or getattr(cls, "_voicemod_teardown_fixed", False):
        return

    def destroy_all_decoders(self) -> None:
        with self._lock:
            for ssrc in list(self.decoders.keys()):
                self.destroy_decoder(ssrc)

    cls.destroy_all_decoders = destroy_all_decoders
    cls._voicemod_teardown_fixed = True


def _install_router_teardown_fix():
    """Stop PacketRouter's exit path from tearing down a *newer* recording.

    py-cord's PacketRouter.run ends with `self.reader.client.stop_recording()`. When we
    deliberately stop a recording (to recycle the sink buffers), that call runs a second
    time on the way out: it raises RecordingException, killing the router thread, and --
    worse -- it unregisters the socket listener that the replacement recording just
    registered. Audio reception then dies silently while the voice connection stays up.

    The exit path only needs to release the waiter; ending the recording is the caller's
    job and is already handled.
    """
    from discord.voice.receive import router as _router

    cls = getattr(_router, "PacketRouter", None)
    if cls is None or getattr(cls, "_voicemod_run_fixed", False):
        return

    def run(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            logger.debug("PacketRouter loop ended: %r", exc)
            try:
                self.reader.error = exc
            except Exception:
                pass
        finally:
            try:
                self.waiter.clear()
            except Exception:
                pass

    cls.run = run
    cls._voicemod_run_fixed = True
    logger.info("Installed packet-router teardown fix")


def _install_opus_decode_guard():
    """Keep one bad Opus frame from ending the whole recording.

    py-cord's PacketRouter loop has no per-packet error handling: an OpusError
    propagates out of the thread's run loop, which then calls stop_recording, so a
    single undecodable frame discards the entire chunk. Substituting silence for a
    failed frame keeps the stream alive and preserves the audio around it.
    """
    import discord.opus as _opus

    if getattr(_opus.Decoder, "_voicemod_guarded", False):
        return
    original_decode = _opus.Decoder.decode

    def decode(self, data, *, fec=True):
        try:
            return original_decode(self, data, fec=fec)
        except _opus.OpusError as exc:
            logger.debug("Opus decode failed (%s); substituting silence", exc)
            return bytes(self.SAMPLES_PER_FRAME * self.CHANNELS * 2)

    _opus.Decoder.decode = decode
    _opus.Decoder._voicemod_guarded = True


_install_dave_decrypt_fix()
_install_decoder_teardown_fix()
_install_router_teardown_fix()
_install_opus_decode_guard()


class LenientWaveSink(discord.sinks.WaveSink):
    """Wave sink that ignores late writes after finalize to avoid SinkException spam.

    py-cord 2.8's receive pipeline expects sink hooks that the legacy
    ``discord.sinks.Sink`` base never defines, so recording dies with AttributeError
    without these shims:

    * ``__sink_listeners__`` / ``walk_children`` - SinkEventRouter.register_events.
      Empty is correct; they only carry speaking/connect notifications we don't use.
    * ``is_opus`` - PacketDecoder asks whether to hand us raw Opus. Returning False
      makes it decode to PCM, which is what WaveSink expects to write.

    Audio itself still arrives through the classic ``write()`` call in PacketRouter.
    """

    __sink_listeners__: list = []

    def walk_children(self):
        return []

    @property
    def root(self):
        return self

    def is_opus(self) -> bool:
        return False

    def write(self, data, user):
        # py-cord 2.8 hands the sink a VoiceData object; the legacy WaveSink expects
        # raw PCM bytes. Unwrap it, and key audio by user id (source can be None
        # because members intent is off, and we process audio anonymously anyway).
        pcm = getattr(data, "pcm", None)
        if pcm is not None:
            source = getattr(data, "source", None)
            user = getattr(source, "id", None) if source is not None else user
            if user is None:
                user = 0
            data = pcm
        if not data:
            return
        try:
            return super().write(data, user)
        except discord.sinks.SinkException:
            # Occurs when packets arrive after sink has been finalized; safe to ignore.
            logger.debug("Ignoring late audio write after sink finalized")
            return

# -------- Persistence --------
def load_state():
    def _empty():
        return {
            "guilds": {},
            "total_incidents": 0,
            "total_major_incidents": 0,
            "total_group1_incidents": 0,
            "total_group2_incidents": 0,
            "total_group3_incidents": 0,
        }

    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read primary state; trying backup")

    backup_path = STATE_PATH.with_suffix(".json.bak")
    if backup_path.exists():
        try:
            logger.warning("Loading state from backup %s", backup_path)
            return json.loads(backup_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read state backup")

    return _empty()

def _atomic_write_text(path: Path, data: str):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(data, encoding="utf-8")
    os.replace(tmp_path, path)

def save_state():
    data = json.dumps(state, indent=2)
    _atomic_write_text(STATE_PATH, data)
    try:
        _atomic_write_text(STATE_PATH.with_suffix(".json.bak"), data)
    except Exception:
        logger.exception("Failed writing state backup")

state = load_state()
state.setdefault("guilds", {})
state.setdefault("total_incidents", 0)

# Back-compat for older `state.json`
state.setdefault("total_major_incidents", 0)
state.setdefault("total_group1_incidents", 0)
state.setdefault("total_group2_incidents", 0)
state.setdefault("total_group3_incidents", 0)

def guild_cfg(guild_id: int):
    defaults = {
        "mod_log_channel_id": None,
        "counter_message_id": None,
        "counter_channel_id": None,
        "call_seconds": 0.0,
        "call_started_at": None,
        "incidents": 0,
        "major_incidents": 0,
        "group1_incidents": 0,
        "group2_incidents": 0,
        "group3_incidents": 0,
        # Per-guild soundboard overrides set via /sounds set; fall back to .env.
        "group_sound_ids": {},
        # Channel that receives the live transcription feed (/live on).
        "live_channel_id": None,
    }
    g = state["guilds"].setdefault(str(guild_id), defaults.copy())
    for key, value in defaults.items():
        g.setdefault(key, value)
    return g


def get_group_sound_id(guild_id: int, group: int):
    """Sound configured for group N: per-guild override first, then .env default."""
    cfg = guild_cfg(guild_id)
    override = (cfg.get("group_sound_ids") or {}).get(str(group))
    if override:
        return override
    return [GROUP1_SOUNDBOARD_SOUND_ID, GROUP2_SOUNDBOARD_SOUND_ID, GROUP3_SOUNDBOARD_SOUND_ID][group - 1]


def set_group_sound_id(guild_id: int, group: int, sound_id) -> None:
    cfg = guild_cfg(guild_id)
    cfg.setdefault("group_sound_ids", {})[str(group)] = str(sound_id) if sound_id else None
    save_state()


def start_call_timer(guild_id: int):
    cfg = guild_cfg(guild_id)
    if cfg.get("call_started_at"):
        return
    cfg["call_started_at"] = time.time()
    save_state()


def stop_call_timer(guild_id: int) -> float:
    cfg = guild_cfg(guild_id)
    started = cfg.get("call_started_at")
    if not started:
        return 0.0
    elapsed = max(0.0, time.time() - started)
    cfg["call_seconds"] += elapsed
    cfg["call_started_at"] = None
    save_state()
    return elapsed


def total_monitored_seconds(guild_id: int) -> tuple[float, bool]:
    cfg = guild_cfg(guild_id)
    total = float(cfg.get("call_seconds", 0.0))
    running = False
    started = cfg.get("call_started_at")
    if started:
        running = True
        total += max(0.0, time.time() - started)
    return total, running


def compute_per_minute(avg_count: int, seconds: float) -> float:
    minutes = seconds / 60.0
    if minutes <= 0:
        return 0.0
    return avg_count / minutes


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def parse_monitor_options(options: tuple[str, ...]) -> tuple[float, bool]:
    # Caps how long one utterance may run before being transcribed anyway.
    chunk_seconds: float = 15.0
    sound_enabled = False
    for raw in options:
        opt = raw.strip().lower()
        if not opt:
            continue
        try:
            # Accept ints or floats for chunk size (e.g., "0.5", "1", "2.5").
            chunk_seconds = float(opt)
            continue
        except ValueError:
            pass

        if opt in SOUND_ON_WORDS:
            sound_enabled = True
            continue
        if opt in SOUND_OFF_WORDS:
            sound_enabled = False
            continue
        raise ValueError(opt)
    return chunk_seconds, sound_enabled


# Discord voice delivers 48 kHz stereo signed 16-bit PCM.
DISCORD_PCM_SR = 48000
DISCORD_PCM_CHANNELS = 2

# Diagnostic: set DUMP_AUDIO_DIR to a folder to save each raw segment for offline analysis.
_dump_dir = os.getenv("DUMP_AUDIO_DIR", "").strip()
DUMP_AUDIO_DIR = Path(_dump_dir) if _dump_dir else None
if DUMP_AUDIO_DIR:
    DUMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _finish_mono(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # Pad very short clips so the model has minimal context (helps with homophones).
    min_seconds = 1.0
    if len(data) < target_sr * min_seconds:
        pad_len = int(target_sr * min_seconds) - len(data)
        # pad half in front, half at end
        pre = pad_len // 2
        post = pad_len - pre
        data = np.pad(data, (pre, post), mode="constant", constant_values=0.0)

    return data.astype(np.float32, copy=False)


def decode_wav_to_float_mono(wav_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode WAV bytes to mono float32 at target sample rate."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    # mixdown to mono
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, dtype=np.float32)
    else:
        data = data[:, 0]
    return _finish_mono(data, sr, target_sr)


def decode_pcm_to_float_mono(
    raw: bytes,
    target_sr: int = 16000,
    src_sr: int = DISCORD_PCM_SR,
    channels: int = DISCORD_PCM_CHANNELS,
) -> np.ndarray:
    """Convert raw interleaved s16le PCM to mono float32 at target sample rate.

    py-cord 2.8 never calls ``Sink.cleanup()`` (the call is commented out in its
    AudioReader), so the sink holds bare PCM with no WAV header. Converting it here
    avoids depending on their formatting path entirely.
    """
    audio = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        usable = (len(audio) // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    data = audio.astype(np.float32) / 32768.0
    return _finish_mono(data, src_sr, target_sr)


def audio_bytes_to_float_mono(raw: bytes, target_sr: int = 16000) -> np.ndarray:
    """Accept either a RIFF/WAV container or bare Discord PCM."""
    if raw[:4] == b"RIFF":
        return decode_wav_to_float_mono(raw, target_sr=target_sr)
    return decode_pcm_to_float_mono(raw, target_sr=target_sr)

# -------- Terms / matching --------
def load_terms_file(path: Path, *, treat_last_two_as_major: bool = False):
    if not path.exists():
        return [], [], set()
    single_words: list[str] = []
    phrases: list[str] = []
    ordered_terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ordered_terms.append(line.lower())
        if " " in line:
            phrases.append(line.lower())
        else:
            single_words.append(line.lower())

    major_terms = set()
    if treat_last_two_as_major and len(ordered_terms) >= 2:
        major_terms = set(ordered_terms[-2:])
    return single_words, phrases, major_terms


single_words, phrases, major_terms = load_terms_file(TERMS_PATH, treat_last_two_as_major=True)

def load_group_terms():
    group_data = []
    for path in (TERMS1_PATH, TERMS2_PATH, TERMS3_PATH):
        g_words, g_phrases, _ = load_terms_file(path, treat_last_two_as_major=False)
        group_data.append({
            "path": path,
            "words": g_words,
            "phrases": g_phrases,
        })
    return group_data

group_terms = load_group_terms()
GROUP_PATHS = (TERMS1_PATH, TERMS2_PATH, TERMS3_PATH)


def reload_group_terms():
    """Re-read the group term files and rebuild their regexes."""
    global group_terms, GROUP_REGEX
    group_terms = load_group_terms()
    GROUP_REGEX = [build_regex(g["words"]) for g in group_terms]


def read_group_lines(group: int) -> list[str]:
    """Current terms for group N, in file order, comments and blanks dropped."""
    path = GROUP_PATHS[group - 1]
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def add_group_term(group: int, term: str) -> tuple[bool, str]:
    """Append a term to group N's file. Returns (changed, message)."""
    term = " ".join(term.strip().lower().split())
    if not term:
        return False, "That term is empty."
    if len(term) > 100:
        return False, "That term is too long (100 characters max)."
    # normalize() strips punctuation before matching, so a term containing any would never hit.
    if normalize(term) != term:
        return False, (
            f"`{term}` contains characters that are stripped before matching. "
            f"Use letters, numbers and spaces only (it would need to be `{normalize(term)}`)."
        )

    existing = read_group_lines(group)
    if term in {e.lower() for e in existing}:
        return False, f"`{term}` is already in group {group}."

    path = GROUP_PATHS[group - 1]
    body = path.read_text(encoding="utf-8") if path.exists() else ""
    if body and not body.endswith("\n"):
        body += "\n"
    _atomic_write_text(path, body + term + "\n")
    reload_group_terms()
    kind = "phrase" if " " in term else "word"
    return True, f"✅ Added {kind} `{term}` to group {group} (`terms{group}.txt`). Now live."


def remove_group_term(group: int, term: str) -> tuple[bool, str]:
    """Remove every occurrence of a term from group N's file."""
    term_l = " ".join(term.strip().lower().split())
    path = GROUP_PATHS[group - 1]
    if not path.exists():
        return False, f"`terms{group}.txt` doesn't exist yet."

    kept, removed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower() == term_l and not line.strip().startswith("#"):
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return False, f"`{term_l}` isn't in group {group}."

    _atomic_write_text(path, "\n".join(kept).rstrip("\n") + ("\n" if kept else ""))
    reload_group_terms()
    plural = "entry" if removed == 1 else "entries"
    return True, f"✅ Removed {removed} {plural} of `{term_l}` from group {group}. Now live."


def load_bias_terms():
    if not BIAS_TERMS_PATH.exists():
        return []
    terms: list[str] = []
    for line in BIAS_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    # dedupe while preserving order
    seen: set[str] = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


BIAS_TERMS = load_bias_terms()
BIAS_PROMPT = None
BIAS_HOTWORDS = None
if BIAS_TERMS:
    # Keep prompt concise so decoding isn’t slowed down.
    limited = BIAS_TERMS[:50]
    BIAS_PROMPT = "Key terms that may appear: " + ", ".join(limited)
    # Hotwords: space-separated; faster-whisper boosts these terms when no prefix is set.
    BIAS_HOTWORDS = " ".join(limited)

def build_regex(words: list[str]) -> re.Pattern | None:
    if not words:
        return None
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

WORD_REGEX = build_regex(single_words)
GROUP_REGEX = [build_regex(g["words"]) for g in group_terms]

def normalize(text: str) -> str:
    # keep letters/numbers/spaces, squash whitespace
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t

def find_flagged(text: str) -> list[str]:
    if not text:
        return []
    norm = normalize(text)
    hits = []

    if WORD_REGEX:
        hits.extend([m.group(0) for m in WORD_REGEX.finditer(norm)])

    for ph in phrases:
        if ph in norm:
            hits.append(ph)

    # unique while preserving order-ish
    seen = set()
    out = []
    for h in hits:
        hh = h.lower()
        if hh not in seen:
            seen.add(hh)
            out.append(h)
    return out


def count_flagged_instances(text: str) -> tuple[int, dict[str, int]]:
    """Return (total_occurrences, per_term_counts) for the given text.

    Counts every match found in normalized text.
    """
    if not text:
        return 0, {}

    norm = normalize(text)
    counts: dict[str, int] = defaultdict(int)

    if WORD_REGEX:
        for m in WORD_REGEX.finditer(norm):
            counts[m.group(0).lower()] += 1

    for ph in phrases:
        if ph in norm:
            counts[ph] += norm.count(ph)

    total = sum(counts.values())
    return total, dict(counts)


def count_group_instances(text: str, words: list[str], phrases_list: list[str], compiled: re.Pattern | None) -> int:
    if not text:
        return 0
    norm = normalize(text)
    count = 0
    if compiled:
        count += sum(1 for _ in compiled.finditer(norm))
    for ph in phrases_list:
        if ph in norm:
            count += norm.count(ph)
    return count

# -------- Discord / bot setup --------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True
intents.members = False  # not needed; audio is processed anonymously

# auto_sync_commands=False: we register per-guild in on_ready instead, so commands show
# up immediately and aren't duplicated by a parallel global registration.
bot = commands.Bot(command_prefix="!", intents=intents, auto_sync_commands=False)

# -------- ASR backend --------
# "parakeet" (default): NVIDIA Parakeet TDT 0.6B via onnxruntime. Faster than whisper even
# on CPU, uses no VRAM, and emits no text at all on silence.
# "whisper": faster-whisper on CUDA. Supports term biasing (initial_prompt/hotwords).
ASR_BACKEND = os.getenv("ASR_BACKEND", "parakeet").strip().lower()
PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v2")
# "cpu" is the default and is fast enough; "cuda" additionally needs `pip install onnxruntime-gpu`.
PARAKEET_DEVICE = os.getenv("PARAKEET_DEVICE", "cpu").strip().lower()
# 0 = let onnxruntime decide; set e.g. 4 to leave cores free for a game.
try:
    PARAKEET_THREADS = int(os.getenv("PARAKEET_THREADS", "0"))
except ValueError:
    PARAKEET_THREADS = 0
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "deepdml/faster-distil-whisper-large-v3.5")

whisper = None
parakeet = None


def _load_parakeet():
    import onnx_asr
    import onnxruntime

    providers = None
    if PARAKEET_DEVICE == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess_options = None
    if PARAKEET_THREADS > 0:
        # Cap CPU usage so transcription doesn't fight a game for cores.
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = PARAKEET_THREADS
        sess_options.inter_op_num_threads = 1

    model = onnx_asr.load_model(PARAKEET_MODEL, providers=providers, sess_options=sess_options)
    # Warm up so the first real chunk isn't slowed by lazy graph init.
    model.recognize(np.zeros(16000, dtype=np.float32))
    return model


def _load_whisper():
    return WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")


if ASR_BACKEND == "parakeet":
    try:
        parakeet = _load_parakeet()
        logger.info("ASR backend: parakeet (%s on %s)", PARAKEET_MODEL, PARAKEET_DEVICE)
    except Exception:
        logger.exception("Failed loading Parakeet; falling back to faster-whisper")
        ASR_BACKEND = "whisper"

if ASR_BACKEND != "parakeet":
    whisper = _load_whisper()
    logger.info("ASR backend: whisper (%s on cuda)", WHISPER_MODEL)

# ASR_COMPARE=true also runs the other engine on every segment and logs both, so the
# two can be judged on the same real audio. Counting still uses ASR_BACKEND.
ASR_COMPARE = os.getenv("ASR_COMPARE", "false").strip().lower() in {"1", "true", "yes", "on"}
if ASR_COMPARE:
    try:
        if whisper is None:
            whisper = _load_whisper()
        if parakeet is None:
            parakeet = _load_parakeet()
        logger.info("ASR compare mode ON: every segment goes through both engines")
    except Exception:
        logger.exception("Failed loading the second engine; compare mode off")
        ASR_COMPARE = False

# Optional streaming VAD+ASR (from whisper_online) for lower-latency detection
online_asr = None
if USE_ONLINE_VAD:
    try:
        from whisper_online import FasterWhisperASR, VACOnlineASRProcessor  # noqa: F811

        online_asr = FasterWhisperASR(WHISPER_LANGUAGE, modelsize=WHISPER_MODEL, logfile=sys.stderr)
        online_asr.use_vad()
    except Exception:
        logger.exception("Failed initializing online ASR; falling back to chunked mode")
        online_asr = None

# Active monitoring sessions per guild
sessions = {}

def log_console(msg: str):
    print(msg, flush=True)


async def fetch_soundboard_sound(guild: discord.Guild, sound_id: int):
    """Return a cached discord.SoundboardSound if soundboard APIs are available."""
    key = (guild.id, sound_id)
    cached = SOUNDBOARD_CACHE.get(key)
    if cached is not None:
        return cached

    # py-cord 2.7: cached lookup first, then a single-sound fetch. Both avoid pulling
    # the guild's entire soundboard on every cache miss.
    get_cached = getattr(guild, "get_sound", None)
    if callable(get_cached):
        try:
            sound = get_cached(sound_id)
            if sound is not None:
                SOUNDBOARD_CACHE[key] = sound
                return sound
        except Exception:
            logger.debug("get_sound failed for id=%s", sound_id, exc_info=True)

    fetch_single = getattr(guild, "fetch_sound", None)
    if callable(fetch_single):
        try:
            sound = await fetch_single(sound_id)
            if sound is not None:
                SOUNDBOARD_CACHE[key] = sound
            return sound
        except Exception:
            logger.exception("fetch_sound failed for id=%s", sound_id)
            return None

    fetch_one = getattr(guild, "fetch_soundboard_sound", None)
    fetch_all = getattr(guild, "fetch_soundboard_sounds", None)
    fetch_pycord = getattr(guild, "fetch_sounds", None)  # Pycord v2.7+
    sound = None
    try:
        if callable(fetch_one):
            sound = await fetch_one(sound_id)
        elif callable(fetch_all):
            sounds = await fetch_all()
            sound = next((s for s in sounds if s.id == sound_id), None)
        elif callable(fetch_pycord):
            sounds = await fetch_pycord()
            sound = next((s for s in sounds if getattr(s, "id", None) == sound_id), None)
        else:
            logger.debug("Soundboard APIs unavailable; skipping sound fetch.")
            return None
    except Exception:
        logger.exception("Failed fetching soundboard sound id=%s", sound_id)
        return None

    if sound is not None:
        SOUNDBOARD_CACHE[key] = sound
    return sound


async def play_soundboard_custom(guild: discord.Guild, sound_id_raw: str | int | None, label: str = "custom"):
    if not ENABLE_SOUNDBOARD_SFX:
        logger.debug("Soundboard disabled; skipping %s incident sound.", label)
        return
    if not sound_id_raw:
        logger.debug("No soundboard ID configured for %s", label)
        return
    try:
        sound_id = int(sound_id_raw)
    except (TypeError, ValueError):
        logger.warning("Invalid soundboard id configured for %s: %r", label, sound_id_raw)
        return

    sess = sessions.get(guild.id)
    if not sess:
        logger.debug("Soundboard skipped (%s): no active monitoring session for guild=%s", label, guild.id)
        return

    if not sess.get("play_sounds"):
        logger.debug("Soundboard skipped (%s): sounds disabled for this session in guild=%s", label, guild.id)
        return

    vc = sess.get("vc")
    if not vc:
        logger.info("Soundboard skipped (%s): bot is not connected to voice for guild=%s", label, guild.id)
        return

    sound = await fetch_soundboard_sound(guild, sound_id)
    if sound is None:
        logger.warning("Could not resolve soundboard sound id=%s for %s in guild=%s", sound_id, label, guild.id)
        return

    channel = getattr(vc, "channel", None)

    play_cb = getattr(vc, "play_soundboard_sound", None)
    if not callable(play_cb):
        play_cb = getattr(vc, "send_soundboard_sound", None)
    if not callable(play_cb) and channel:
        play_cb = getattr(channel, "play_soundboard_sound", None)
    if not callable(play_cb) and channel:
        play_cb = getattr(channel, "send_soundboard_sound", None)
    if not callable(play_cb):
        logger.warning("Soundboard playback unsupported (%s) in this discord.py/py-cord build; guild=%s", label, guild.id)
        return

    try:
        await play_cb(sound)
        logger.info("Played %s soundboard clip id=%s in guild=%s", label, sound_id, guild.id)
    except Exception:
        logger.exception("Failed playing soundboard sound id=%s for %s", sound_id, label)


def is_mod(ctx: commands.Context) -> bool:
    # Only let mods/admins run moderation commands
    perms = ctx.author.guild_permissions
    return perms.moderate_members or perms.administrator

async def get_mod_log_channel(guild: discord.Guild):
    cfg = guild_cfg(guild.id)
    ch_id = cfg.get("mod_log_channel_id")
    if ch_id:
        ch = guild.get_channel(int(ch_id))
        if ch:
            return ch

    # fallback: channel named mod-log
    for ch in guild.text_channels:
        if ch.name.lower() in {"mod-log", "modlogs", "mod-logs"}:
            return ch
    return None

async def update_counter_message(guild: discord.Guild):
    cfg = guild_cfg(guild.id)
    ch_id = cfg.get("counter_channel_id")
    if not ch_id:
        return

    ch = guild.get_channel(int(ch_id))
    if not ch:
        return

    total = state["total_incidents"]
    major_total = state.get("total_major_incidents", 0)
    content = (
        f"📊 **Voice Moderation Counter (running total)**\n"
        f"All-time count: **{total}**\n"
        f"Major incidents: **{major_total}**\n"
        f"_Counts every occurrence of configured terms._"
    )

    msg_id = cfg.get("counter_message_id")
    if msg_id:
        try:
            msg = await ch.fetch_message(int(msg_id))
            await msg.edit(content=content)
            return
        except Exception:
            logger.exception("Failed updating existing counter message")
            cfg["counter_message_id"] = None
            save_state()

    try:
        msg = await ch.send(content)
        cfg["counter_message_id"] = msg.id
        cfg["counter_channel_id"] = ch.id
        save_state()
    except Exception:
        logger.exception("Failed sending counter message")


async def send_counter_incident_message(guild: discord.Guild, total_after: int):
    """Send a new message in the configured counter channel."""
    cfg = guild_cfg(guild.id)
    ch_id = cfg.get("counter_channel_id")
    if not ch_id:
        logger.warning("Incident detected but counter channel is not set. Run !counter in the channel you want.")
        return

    ch = guild.get_channel(int(ch_id))
    if not ch:
        logger.warning("Counter channel id %s not found in guild", ch_id)
        return

    try:
        await ch.send(f"📊 Incident detected. All-time: **{total_after}**")
    except Exception:
        logger.exception("Failed sending per-incident counter message")


# One message style per term group. {total} is the all-time count after this hit,
# {group_total} the running count for that group alone.
GROUP_INCIDENT_TEMPLATES: dict[int, str] = {
    1: "📊 Incident detected. All-time: **{total}**",
    2: "📊 Incident detected. All-time: **{total}**",
    3: "📊 Incident detected. Group 3: **{group_total}** · all-time: **{total}**",
}
DEFAULT_INCIDENT_TEMPLATE = "📊 Incident detected. All-time: **{total}**"


async def send_group_incident_message(
    guild: discord.Guild, group: int | None, total_after: int, group_total_after: int
):
    """Announce one occurrence using that group's message style."""
    cfg = guild_cfg(guild.id)
    ch_id = cfg.get("counter_channel_id")
    if not ch_id:
        logger.warning("Incident detected but counter channel is not set. Run /counter in the channel you want.")
        return
    ch = guild.get_channel(int(ch_id))
    if not ch:
        logger.warning("Counter channel id %s not found in guild", ch_id)
        return

    template = GROUP_INCIDENT_TEMPLATES.get(group or 0, DEFAULT_INCIDENT_TEMPLATE)
    try:
        await ch.send(template.format(total=total_after, group_total=group_total_after))
    except Exception:
        logger.exception("Failed sending incident message for group %s", group)


async def send_major_incident_message(guild: discord.Guild, major_total_after: int):
    """Send a special message for major incidents (the last two terms of the base list)."""
    cfg = guild_cfg(guild.id)
    ch_id = cfg.get("counter_channel_id")
    if not ch_id:
        logger.warning("Major incident detected but counter channel is not set. Run !counter in the channel you want.")
        return

    ch = guild.get_channel(int(ch_id))
    if not ch:
        logger.warning("Counter channel id %s not found in guild", ch_id)
        return

    try:
        await ch.send(f"📊 Major incident detected. Major all-time: **{major_total_after}**")
    except Exception:
        logger.exception("Failed sending major-incident message")

async def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribe 16 kHz mono float32 audio with the active backend."""

    def _run_parakeet() -> str:
        return (parakeet.recognize(audio) or "").strip()

    def _run_whisper() -> str:
        # faster-whisper accepts a float32 array directly, so no re-encoding is needed.
        segments, _info = whisper.transcribe(
            audio,
            vad_filter=False,
            language=WHISPER_LANGUAGE,
            beam_size=5,
            temperature=0.0,
            initial_prompt=BIAS_PROMPT,
            hotwords=BIAS_HOTWORDS,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    primary_is_parakeet = ASR_BACKEND == "parakeet" and parakeet is not None
    _run = _run_parakeet if primary_is_parakeet else _run_whisper

    start = time.perf_counter()
    text = await asyncio.to_thread(_run)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("%s done: %.1fs audio in %.0fms", ASR_BACKEND, len(audio) / 16000, elapsed_ms)

    if ASR_COMPARE:
        other = _run_whisper if primary_is_parakeet else _run_parakeet
        other_name = "whisper" if primary_is_parakeet else "parakeet"
        o_start = time.perf_counter()
        try:
            other_text = await asyncio.to_thread(other)
        except Exception:
            logger.exception("Compare engine failed")
            other_text = "<error>"
        o_ms = (time.perf_counter() - o_start) * 1000

        def _hits(t: str) -> list[str]:
            found: list[str] = []
            for terms in group_hit_terms(t):
                found.extend(terms)
            return found

        p_text, w_text = (text, other_text) if primary_is_parakeet else (other_text, text)
        p_ms, w_ms = (elapsed_ms, o_ms) if primary_is_parakeet else (o_ms, elapsed_ms)
        p_hits, w_hits = _hits(p_text), _hits(w_text)
        marker = " <<< DIFFERENT HITS" if set(p_hits) != set(w_hits) else ""
        logger.info(
            "[COMPARE %.1fs]%s\n    parakeet [%4.0fms] %r  hits=%s\n    whisper  [%4.0fms] %r  hits=%s",
            len(audio) / 16000, marker, p_ms, p_text, p_hits, w_ms, w_text, w_hits,
        )
        _ = other_name

    return text


async def transcribe_audio_batch(clips: list[np.ndarray]) -> list[str]:
    """Transcribe several speakers' audio in one pass.

    With three or more people talking at once, transcribing them one after another made
    each detection cycle N times longer, which forced a slower scan interval. Parakeet
    accepts a list, so one batched call keeps the cycle roughly flat as speakers are
    added.
    """
    if not clips:
        return []
    if len(clips) == 1:
        return [await transcribe_audio(clips[0])]

    def _run() -> list[str]:
        if parakeet is not None:
            out = parakeet.recognize(clips)
            if isinstance(out, str):
                out = [out]
            return [(t or "").strip() for t in out]
        results = []
        for c in clips:
            segments, _info = whisper.transcribe(
                c, vad_filter=False, language=WHISPER_LANGUAGE, beam_size=5,
                temperature=0.0, initial_prompt=BIAS_PROMPT, hotwords=BIAS_HOTWORDS,
            )
            results.append(" ".join(seg.text.strip() for seg in segments).strip())
        return results

    start = time.perf_counter()
    try:
        texts = await asyncio.to_thread(_run)
    except Exception:
        logger.exception("Batched transcription failed; falling back to one at a time")
        return [await transcribe_audio(c) for c in clips]
    elapsed_ms = (time.perf_counter() - start) * 1000
    total_s = sum(len(c) for c in clips) / 16000
    logger.debug(
        "%s batch: %d clips (%.1fs audio) in %.0fms", ASR_BACKEND, len(clips), total_s, elapsed_ms
    )
    return texts


# -------- Live transcription feed --------
# Chunks arrive every few seconds per speaker, which would blow past Discord's
# ~5 messages / 5s per-channel limit. Lines are buffered and flushed in batches.
LIVE_BUFFERS: dict[int, list[str]] = defaultdict(list)
LIVE_FLUSH_TASKS: dict[int, asyncio.Task] = {}
LIVE_FLUSH_SECONDS = 4.0


def group_hit_terms(text: str) -> list[list[str]]:
    """Which terms of each group appear in the text (for the live feed)."""
    norm = normalize(text)
    per_group: list[list[str]] = []
    for idx, g in enumerate(group_terms):
        found = []
        rx = GROUP_REGEX[idx]
        if rx:
            found.extend(sorted({m.group(0).lower() for m in rx.finditer(norm)}))
        found.extend(ph for ph in g["phrases"] if ph in norm)
        per_group.append(found)
    return per_group


def format_live_line(text: str, speaker: str = "") -> str:
    hits = group_hit_terms(text)
    snippet = text.strip().replace("`", "'")[:300]
    tags = [
        f"G{i + 1}: {', '.join(terms)}"
        for i, terms in enumerate(hits)
        if terms
    ]
    who = f"**{speaker}**: " if speaker else ""
    if tags:
        return f"🔴 {who}`{snippet}` — **{' · '.join(tags)}**"
    return f"⚪ {who}`{snippet}`"


async def _flush_live(guild: discord.Guild):
    await asyncio.sleep(LIVE_FLUSH_SECONDS)
    lines = LIVE_BUFFERS.pop(guild.id, [])
    LIVE_FLUSH_TASKS.pop(guild.id, None)
    if not lines:
        return

    cfg = guild_cfg(guild.id)
    ch_id = cfg.get("live_channel_id")
    if not ch_id:
        return
    ch = guild.get_channel(int(ch_id))
    if not ch:
        return

    body = "\n".join(lines)
    if len(body) > 1900:
        body = body[-1900:]
    try:
        await ch.send(body)
    except Exception:
        logger.exception("Failed sending live transcription batch")


def queue_live_transcript(guild: discord.Guild, text: str, speaker: str = ""):
    """Buffer a transcript line for the live feed, if enabled for this guild."""
    if not text:
        return
    cfg = guild_cfg(guild.id)
    if not cfg.get("live_channel_id"):
        return

    LIVE_BUFFERS[guild.id].append(format_live_line(text, speaker))
    task = LIVE_FLUSH_TASKS.get(guild.id)
    if task is None or task.done():
        LIVE_FLUSH_TASKS[guild.id] = asyncio.create_task(_flush_live(guild))


SPEAKER_NAME_CACHE: dict[int, str] = {}


async def resolve_speaker(guild: discord.Guild, user_id) -> str:
    """Best-effort display name for a speaker id.

    The members intent is off, so guild.get_member is usually empty; fetch_user is a
    plain REST lookup that works without it. Results are cached so this costs one call
    per person per run.
    """
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return "unknown"
    if uid in SPEAKER_NAME_CACHE:
        return SPEAKER_NAME_CACHE[uid]

    name = None
    member = guild.get_member(uid)
    if member is not None:
        name = member.display_name
    else:
        user = bot.get_user(uid)
        if user is not None:
            name = user.display_name
        else:
            try:
                fetched = await bot.fetch_user(uid)
                name = fetched.display_name
            except Exception:
                logger.debug("Could not resolve speaker %s", uid, exc_info=True)

    name = name or f"user{str(uid)[-4:]}"
    SPEAKER_NAME_CACHE[uid] = name
    return name


def prepare_segment(raw_bytes: bytes) -> np.ndarray | None:
    """Decode a raw segment for transcription, or None if it isn't worth running."""
    if len(raw_bytes) < 10_000:
        return None
    try:
        audio_16k = audio_bytes_to_float_mono(raw_bytes, target_sr=16000)
    except Exception:
        logger.exception("Failed decoding audio segment (%d bytes)", len(raw_bytes))
        return None
    if audio_16k.size == 0 or float(np.max(np.abs(audio_16k))) < 0.002:
        return None
    return audio_16k


async def process_audio_segment(
    guild: discord.Guild,
    raw_bytes: bytes,
    previous_text: str = "",
    already_counted: dict[str, int] | None = None,
    quiet: bool = False,
    pretranscribed: str | None = None,
    speaker: str = "",
) -> tuple[str, dict[str, int]]:
    """Transcribe one speaker's audio and count any terms not already counted.

    ``already_counted`` lets the same (growing) utterance be re-scanned while the person
    is still talking: only occurrences beyond what was previously credited are counted,
    so a word is never counted twice. Returns the transcript plus the updated tally.

    Audio is handled anonymously; the speaker is never identified or recorded.
    """
    counted = dict(already_counted or {})
    # 10 KB of 48 kHz stereo PCM is only ~50 ms, so this just drops blips.
    if len(raw_bytes) < 10_000:
        logger.debug("skip tiny segment: %d bytes", len(raw_bytes))
        return "", counted

    try:
        audio_16k = audio_bytes_to_float_mono(raw_bytes, target_sr=16000)
    except Exception:
        logger.exception("Failed decoding audio segment (%d bytes)", len(raw_bytes))
        return "", counted

    peak = float(np.max(np.abs(audio_16k))) if audio_16k.size else 0.0
    logger.debug(
        "segment decoded: %d bytes -> %.2fs, peak=%.4f", len(raw_bytes), len(audio_16k) / 16000, peak
    )

    if DUMP_AUDIO_DIR:
        try:
            path = DUMP_AUDIO_DIR / f"seg_{int(time.time() * 1000)}.bin"
            path.write_bytes(raw_bytes)
        except Exception:
            logger.exception("Failed dumping raw segment")
    if peak < 0.002:
        # Effectively silence; skip the model entirely.
        logger.debug("skip near-silent segment (peak=%.4f)", peak)
        return "", counted

    text = ""
    online_used = False
    online_logged = False
    # Preferred: online VAD + ASR pipeline if available. We only consume committed text.
    sess = sessions.get(guild.id)
    proc = sess.get("online_processor") if sess else None

    if USE_ONLINE_VAD and proc is not None:
        try:
            audio_f32 = audio_bytes_to_float_mono(raw_bytes, target_sr=proc.SAMPLING_RATE)
            proc.insert_audio_chunk(audio_f32)
            _b, _e, text = proc.process_iter()
            online_used = True
            if text:
                logger.info("[ONLINE TRANSCRIPTION] %r", text)
                online_logged = True
            else:
                logger.debug("Online ASR returned no committed text yet")
        except Exception:
            logger.exception("Online VAD/ASR failed; falling back to chunked transcribe")
            text = ""
            online_used = False

    if not text:
        if online_used:
            # No committed text yet; skip this chunk to avoid double-counting partials.
            return "", counted
        if pretranscribed is not None:
            text = pretranscribed
        else:
            try:
                text = await transcribe_audio(audio_16k)
            except Exception:
                logger.exception("Transcription failed")
                return "", counted

    if not text:
        logger.debug("empty transcription")
        return "", counted

    # Always show what the ASR produced (skip if already logged for online commit).
    if not online_logged:
        if speaker:
            logger.info("[TRANSCRIPTION] %s: %r", speaker, text)
        else:
            logger.info("[TRANSCRIPTION] %r", text)

    # The window deliberately overlaps the previous one, so drop the speech that was
    # already counted last time. Everything below sees only genuinely new words.
    full_text = text
    text = drop_repeated_prefix(previous_text, text)
    if not text:
        logger.debug("segment was entirely overlap; nothing new")
        return full_text, counted
    if text != full_text:
        logger.debug("overlap trimmed: %r -> %r", full_text, text)

    # Mirror every transcript to the live feed, including ones that matched nothing:
    # those are exactly the mishears worth adding as new terms.
    if not quiet:
        queue_live_transcript(guild, text, speaker)

    # Count every occurrence (not just "any hit").
    increment, per_term = count_flagged_instances(text)
    group_increments = []
    for idx, g in enumerate(group_terms):
        inc = count_group_instances(text, g["words"], g["phrases"], GROUP_REGEX[idx])
        group_increments.append(inc)

    # This transcript may re-cover speech counted a moment ago (the utterance is
    # re-scanned as it grows), so credit only what is new.
    def _take(key: str, total: int) -> int:
        prior = counted.get(key, 0)
        counted[key] = max(prior, total)
        return max(0, total - prior)

    increment = _take("main", increment)
    per_term = {t: _take(f"term:{t}", c) for t, c in per_term.items()}
    per_term = {t: c for t, c in per_term.items() if c > 0}
    group_increments = [_take(f"group{i}", v) for i, v in enumerate(group_increments)]

    if increment <= 0 and all(v <= 0 for v in group_increments):
        logger.debug("no new hits in text snippet=%r", text[:120])
        return full_text, counted

    major_increment = 0
    if per_term and major_terms:
        for term in major_terms:
            major_increment += int(per_term.get(term, 0))
    group_total = sum(group_increments)
    effective_increment = increment + group_total
    normal_increment = effective_increment - major_increment

    # Update totals one-by-one so the counter channel gets one message per occurrence.
    start_total = state["total_incidents"]
    start_major_total = state.get("total_major_incidents", 0)

    state["total_incidents"] += effective_increment
    state["total_major_incidents"] = start_major_total + major_increment
    cfg = guild_cfg(guild.id)
    cfg["incidents"] = cfg.get("incidents", 0) + effective_increment
    cfg["major_incidents"] = cfg.get("major_incidents", 0) + major_increment

    group_keys_total = ["total_group1_incidents", "total_group2_incidents", "total_group3_incidents"]
    group_keys_guild = ["group1_incidents", "group2_incidents", "group3_incidents"]
    for idx, inc in enumerate(group_increments):
        if inc <= 0:
            continue
        state[group_keys_total[idx]] = state.get(group_keys_total[idx], 0) + inc
        cfg[group_keys_guild[idx]] = cfg.get(group_keys_guild[idx], 0) + inc
        # Play once per detected occurrence to mirror counter increments.
        sound_id = get_group_sound_id(guild.id, idx + 1)
        for _ in range(inc):
            await play_soundboard_custom(guild, sound_id, label=f"group{idx+1}")
    save_state()

    # Print to terminal (anonymous)
    logger.info(
        "[INCIDENT] %s+%d total=%d distinct_terms=%d snippet=%r",
        f"{speaker} " if speaker else "",
        effective_increment,
        state["total_incidents"],
        len(per_term),
        text[:140],
    )

    # One message per occurrence, in that group's style. Major incidents keep their
    # own separate announcement and counter.
    for i in range(major_increment):
        await send_major_incident_message(guild, start_major_total + i + 1)

    running = start_total + major_increment
    # Occurrences from terms.txt itself have no group; anything from terms1/2/3 does.
    occurrences: list[int | None] = [None] * max(0, increment - major_increment)
    for idx, inc in enumerate(group_increments):
        occurrences.extend([idx + 1] * max(0, inc))

    group_running = {
        g: state.get(f"total_group{g}_incidents", 0) - group_increments[g - 1]
        for g in (1, 2, 3)
    }
    for group in occurrences:
        running += 1
        if group is not None:
            group_running[group] += 1
        await send_group_incident_message(
            guild, group, running, group_running.get(group or 0, 0)
        )

    # Also keep the dashboard message (from !counter) updated.
    await update_counter_message(guild)

    return full_text, counted


# Recording is restarted occasionally so the sinks' in-memory buffers don't grow
# without bound (48 kHz stereo PCM is ~190 KB/s per speaker).
RECORDING_RESET_SECONDS = 300.0

# --- Utterance segmentation ---
# Discord only sends packets while someone is actually speaking, so a speaker's buffer
# growing means "talking" and standing still means "paused". Waiting for the pause and
# transcribing the whole utterance beats slicing on a fixed timer: measured segments
# averaged 1.9s that way, which is too little audio for any model to decode reliably.
UTTERANCE_POLL_SECONDS = 0.15
# How long a speaker must be silent before their utterance is considered finished.
UTTERANCE_SILENCE_HOLD = 0.7
# Force a flush on someone who never pauses, so detection can't stall indefinitely.
UTTERANCE_MAX_SECONDS = 15.0
# Ignore blips shorter than this.
UTTERANCE_MIN_SECONDS = 0.35
# How often the audio so far is re-scanned while someone is still talking. Waiting for
# the pause meant a word said early in a long rant wasn't caught until the rant ended;
# re-scanning keeps detection within about a second of the word being spoken. Only
# occurrences beyond those already credited are counted, so nothing counts twice.
# With this many speakers talking at once, interim scans are transcribed as one batch
# instead of one after another, so the detection cycle stays fast as people pile on.
BATCH_SCAN_MIN_SPEAKERS = 3
# Warn if no audio arrives for this long while still connected to voice.
AUDIO_STALL_WARN_SECONDS = 90.0

try:
    DETECT_INTERVAL_SECONDS = float(os.getenv("DETECT_INTERVAL_SECONDS", "0.3"))
except ValueError:
    DETECT_INTERVAL_SECONDS = 0.3
# Audio re-read from before the utterance start. A pause long enough to trigger a flush
# can fall mid-sentence, so replaying the tail of the previous utterance keeps a word
# spanning that split intact and gives the model lead-in context. Words repeated by the
# overlap are stripped afterwards so nothing is counted twice.
UTTERANCE_PAD_SECONDS = 0.5

# How long to tolerate a dropped voice connection before giving up. Discord migrates
# voice servers periodically ("Disconnecting from voice by force... potentially
# reconnecting"), and py-cord reconnects on its own; bailing out immediately turned a
# routine blip into a dead session.
RECONNECT_GRACE_SECONDS = 45.0


def drop_repeated_prefix(previous: str, current: str) -> str:
    """Remove the part of `current` already reported in `previous`.

    Each window re-transcribes a little of the preceding audio, so consecutive
    transcripts overlap. Whatever tail of the previous transcript opens this one is
    the same speech heard twice and must not be counted again.
    """
    if not previous or not current:
        return current
    prev_words = previous.split()
    cur_words = current.split()
    max_overlap = min(len(prev_words), len(cur_words))
    for size in range(max_overlap, 0, -1):
        if [w.lower() for w in prev_words[-size:]] == [w.lower() for w in cur_words[:size]]:
            return " ".join(cur_words[size:])
    return current


async def monitoring_loop(guild: discord.Guild, vc: discord.VoiceClient, chunk_seconds: float = 15.0):
    """Record continuously and transcribe one whole utterance at a time.

    Two things this avoids. Stopping and restarting the recording per chunk made
    py-cord discard the decoder's buffered packets every time, shredding speech into
    fragments. And slicing on a fixed timer produced ~1.9s pieces, too little audio to
    decode reliably. Instead the recording runs once and each speaker's audio is held
    until they stop talking, then transcribed whole.
    """
    bytes_per_second = DISCORD_PCM_SR * DISCORD_PCM_CHANNELS * 2
    # `chunk_seconds` now caps how long one utterance may run before being transcribed
    # anyway, so a non-stop talker still gets counted promptly.
    max_utterance = max(3.0, float(chunk_seconds))

    def _start(sink) -> bool:
        try:
            # py-cord 2.8 no longer calls sink.init(vc) (the line is commented out in
            # its AudioReader), leaving sink.client None and tripping an assert in the
            # opus decoder on the first packet.
            if getattr(sink, "vc", None) is None:
                sink.init(vc)
            vc.start_recording(sink, lambda error=None: (
                logger.error("Recording ended with error: %r", error) if error else None
            ))
            return True
        except Exception:
            logger.exception("start_recording failed for guild=%s", guild.id)
            return False

    sink = LenientWaveSink()
    if not _start(sink):
        sessions.pop(guild.id, None)
        stop_call_timer(guild.id)
        try:
            await vc.disconnect(force=True)
        except Exception:
            pass
        return

    # Per speaker: bytes already transcribed, last observed buffer size, and when it
    # last grew (i.e. when they were last actually speaking).
    offsets: dict[object, int] = {}          # start of the current utterance
    last_size: dict[object, int] = {}
    last_growth: dict[object, float] = {}
    last_text: dict[object, str] = {}        # transcript of the previous utterance
    counted: dict[object, dict[str, int]] = {}   # credited so far in this utterance
    last_scan: dict[object, float] = {}
    scanning: set = set()
    started_at = time.monotonic()
    disconnected_since: float | None = None
    pad_bytes = int(UTTERANCE_PAD_SECONDS * bytes_per_second)
    last_audio_at = time.monotonic()
    last_total_buffered = 0
    stall_warned = False

    async def _scan(user, buffered: bytes, reason: str, final: bool):
        """Transcribe the current utterance so far and count anything new."""
        if user in scanning:
            return
        consumed = offsets.get(user, 0)
        new_seconds = (len(buffered) - consumed) / bytes_per_second
        if new_seconds < UTTERANCE_MIN_SECONDS:
            if final:
                offsets[user] = len(buffered)
                counted.pop(user, None)
            return
        # Replay a little of the previous utterance so a word split at the boundary
        # stays whole; duplicated words are stripped from the transcript.
        start = max(0, consumed - pad_bytes)
        segment = buffered[start:]
        scanning.add(user)
        try:
            logger.debug(
                "%s scan (%s): %.2fs + %.2fs lead-in",
                "final" if final else "interim", reason, new_seconds,
                (consumed - start) / bytes_per_second,
            )
            text, tally = await process_audio_segment(
                guild, segment, last_text.get(user, ""),
                already_counted=counted.get(user),
                # Interim passes re-cover the same speech; only mirror the final one to
                # the live feed so the channel isn't spammed with partial transcripts.
                quiet=not final,
                speaker=await resolve_speaker(guild, user),
            )
        finally:
            scanning.discard(user)
        counted[user] = tally
        if final:
            if text:
                last_text[user] = text
            offsets[user] = len(buffered)
            counted.pop(user, None)

    async def _scan_many(items):
        """Interim-scan several speakers in one batched transcription pass.

        Sequential scans made the cycle N times longer with N talkers, which is what
        forced a slower detect interval. Batching keeps it roughly flat.
        """
        prepared = []
        for user, buffered in items:
            if user in scanning:
                continue
            consumed = offsets.get(user, 0)
            new_seconds = (len(buffered) - consumed) / bytes_per_second
            if new_seconds < UTTERANCE_MIN_SECONDS:
                continue
            start = max(0, consumed - pad_bytes)
            segment = buffered[start:]
            audio = prepare_segment(segment)
            if audio is None:
                continue
            prepared.append((user, buffered, segment, audio))

        if not prepared:
            return
        if len(prepared) < BATCH_SCAN_MIN_SPEAKERS:
            for user, buffered, _seg, _a in prepared:
                await _scan(user, buffered, "mid-speech", final=False)
            return

        for user, _b, _s, _a in prepared:
            scanning.add(user)
        try:
            logger.debug("batched interim scan: %d speakers", len(prepared))
            texts = await transcribe_audio_batch([a for _u, _b, _s, a in prepared])
        except Exception:
            logger.exception("Batched scan failed")
            texts = [None] * len(prepared)
        finally:
            for user, _b, _s, _a in prepared:
                scanning.discard(user)

        for (user, buffered, segment, _audio), text in zip(prepared, texts):
            if text is None:
                continue
            _full, tally = await process_audio_segment(
                guild, segment, last_text.get(user, ""),
                already_counted=counted.get(user), quiet=True, pretranscribed=text,
                speaker=await resolve_speaker(guild, user),
            )
            counted[user] = tally

    async def _flush(user, buffered: bytes, reason: str):
        await _scan(user, buffered, reason, final=True)

    try:
        while sessions.get(guild.id, {}).get("running"):
            await asyncio.sleep(UTTERANCE_POLL_SECONDS)
            now = time.monotonic()

            if not vc.is_connected():
                # Discord migrates voice servers periodically and py-cord reconnects on
                # its own; only give up if it stays down.
                if disconnected_since is None:
                    disconnected_since = now
                    logger.warning("Voice connection dropped; waiting up to %.0fs to recover (guild=%s)",
                                   RECONNECT_GRACE_SECONDS, guild.id)
                elif now - disconnected_since >= RECONNECT_GRACE_SECONDS:
                    logger.warning("Voice did not recover after %.0fs; ending monitoring for guild=%s",
                                   RECONNECT_GRACE_SECONDS, guild.id)
                    break
                continue

            if disconnected_since is not None:
                # Reconnected: the old reader is gone, so recording must be restarted.
                logger.info("Voice reconnected after %.1fs; resuming recording for guild=%s",
                            now - disconnected_since, guild.id)
                disconnected_since = None
                sink = LenientWaveSink()
                offsets.clear(); last_size.clear(); last_growth.clear(); counted.clear(); last_scan.clear()
                if not _start(sink):
                    break
                started_at = now
                continue

            due_scans: list = []
            for user, audio in list(getattr(sink, "audio_data", {}).items()):
                try:
                    buffered = audio.file.getvalue()
                except Exception:
                    logger.exception("Failed reading audio buffer")
                    continue

                size = len(buffered)
                grew = size - last_size.get(user, 0)
                if grew > 0:
                    last_size[user] = size
                    # Growing isn't the same as speaking: if py-cord backfills gaps with
                    # concealment frames the buffer keeps growing through silence, so
                    # judge by the level of the newly written audio, not its existence.
                    speaking = True
                    try:
                        fresh = decode_pcm_to_float_mono(buffered[size - grew:], target_sr=16000)
                        speaking = fresh.size > 0 and float(np.max(np.abs(fresh))) >= 0.004
                    except Exception:
                        pass
                    if speaking:
                        last_growth[user] = now
                        pending = (size - offsets.get(user, 0)) / bytes_per_second
                        if pending >= max_utterance:
                            await _flush(user, buffered, "max length")
                        elif now - last_scan.get(user, 0.0) >= DETECT_INTERVAL_SECONDS:
                            # Catch flagged words mid-sentence rather than waiting for
                            # the speaker to stop. Collected and dispatched together
                            # below so several talkers share one transcription pass.
                            last_scan[user] = now
                            due_scans.append((user, buffered))
                        continue

                # Nothing new, or only silence: they've paused.
                pending = (size - offsets.get(user, 0)) / bytes_per_second
                if pending <= 0:
                    continue
                if now - last_growth.get(user, now) >= UTTERANCE_SILENCE_HOLD:
                    await _flush(user, buffered, "pause")

            if due_scans:
                await _scan_many(due_scans)

            # Watchdog: audio reception can die while the voice socket stays healthy
            # (a py-cord teardown race). Surface it instead of failing silently.
            total_buffered = sum(last_size.values())
            if total_buffered > last_total_buffered:
                last_total_buffered = total_buffered
                last_audio_at = now
            elif now - last_audio_at >= AUDIO_STALL_WARN_SECONDS and not stall_warned:
                stall_warned = True
                logger.warning(
                    "No audio received for %.0fs while still connected to voice "
                    "(guild=%s) - reception may have stalled.",
                    now - last_audio_at, guild.id,
                )

            # Restart periodically so the sink buffers don't grow without bound.
            if now - started_at >= RECORDING_RESET_SECONDS:
                logger.debug("cycling recording buffers for guild=%s", guild.id)
                for user, audio in list(getattr(sink, "audio_data", {}).items()):
                    try:
                        await _flush(user, audio.file.getvalue(), "buffer cycle")
                    except Exception:
                        logger.exception("Failed flushing before buffer cycle")
                try:
                    vc.stop_recording()
                except Exception:
                    logger.exception("stop_recording failed during buffer cycle")
                sink = LenientWaveSink()
                offsets.clear(); last_size.clear(); last_growth.clear(); counted.clear(); last_scan.clear()
                await asyncio.sleep(0.2)   # let the old reader finish tearing down
                if not _start(sink):
                    break
                if not getattr(vc, "is_recording", lambda: True)():
                    logger.warning("Recording did not resume after buffer cycle; retrying")
                    await asyncio.sleep(0.5)
                    if not _start(sink):
                        break
                started_at = time.monotonic()
    finally:
        # Runs on !monitor stop (task cancel) and on external disconnect alike,
        # so a dead session can never linger and block a future !monitor start.
        sessions.pop(guild.id, None)
        stop_call_timer(guild.id)
        try:
            # Flush the in-flight recording (no-op if none); salvages the last chunk on cancel.
            vc.stop_recording()
        except Exception:
            pass
        try:
            await vc.disconnect(force=True)
        except Exception:
            pass


# -------- Command core --------
# Each core_* function holds the real logic and returns a message string, so the
# prefix commands and the slash commands stay in lockstep.

GROUP_CHOICES = [
    discord.OptionChoice(name="Group 1 - terms1.txt", value=1),
    discord.OptionChoice(name="Group 2 - terms2.txt", value=2),
    discord.OptionChoice(name="Group 3 - terms3.txt", value=3),
]


def terms_configured() -> bool:
    if single_words or phrases:
        return True
    return any(g.get("words") or g.get("phrases") for g in group_terms)


async def core_start_monitor(guild, author, channel, chunk_seconds: float, sound_enabled: bool) -> str:
    if not terms_configured():
        return "No terms configured. Add terms to `terms.txt` or the group files (`terms1/2/3.txt`) and restart."

    if guild.id in sessions:
        return "Already monitoring here. Stop it first with `/monitor stop`."

    if channel is None:
        if not getattr(author, "voice", None) or not author.voice.channel:
            return "Join a voice channel first, or pass one with the `channel` option."
        channel = author.voice.channel

    try:
        existing_vc = guild.voice_client
        if existing_vc and existing_vc.is_connected():
            # Reuse a leftover client (e.g. after a crashed session) instead of erroring.
            if existing_vc.channel != channel:
                await existing_vc.move_to(channel)
            vc = existing_vc
        else:
            if existing_vc:
                await existing_vc.disconnect(force=True)
            vc = await channel.connect()
    except Exception:
        logger.exception("Failed connecting to voice channel %s", channel)
        return f"Couldn't connect to **{channel.name}**. Check my Connect/Speak permissions there."

    # py-cord 2.8 negotiates Discord's DAVE (E2EE) session after connect() returns;
    # recording before it completes would drop the first chunk.
    try:
        # Synchronous in py-cord 2.8 (returns bool), so it must not be awaited.
        ready = await asyncio.to_thread(vc.wait_until_connected, 30.0)
        if not ready:
            logger.warning("Voice session not ready within 30s in guild=%s", guild.id)
        else:
            logger.info(
                "Voice ready in %s (dave=%s)", channel.name, getattr(vc, "is_dave_connection", "?")
            )
    except Exception:
        logger.exception("wait_until_connected failed; continuing anyway")

    # Clamp to reasonable bounds for offline chunked transcription.
    chunk_seconds = max(3.0, min(float(chunk_seconds), 30.0))

    processor = None
    if USE_ONLINE_VAD and online_asr is not None:
        try:
            processor = VACOnlineASRProcessor(
                online_chunk_size=0.5,
                asr=online_asr,
                tokenizer=None,
                buffer_trimming=("segment", 15),
                logfile=sys.stderr,
            )
        except Exception:
            logger.exception("Failed creating online VAD processor; continuing without it")
            processor = None

    sessions[guild.id] = {
        "running": True,
        "vc": vc,
        "task": None,
        "lock": asyncio.Lock(),
        "chunk_seconds": chunk_seconds,
        "play_sounds": bool(sound_enabled),
        "online_processor": processor,
    }
    sessions[guild.id]["task"] = asyncio.create_task(
        monitoring_loop(guild, vc, chunk_seconds=chunk_seconds)
    )
    start_call_timer(guild.id)

    cfg = guild_cfg(guild.id)
    warn = ""
    if not cfg.get("counter_channel_id"):
        warn = "\n⚠️ No counter channel set — detections won't be posted anywhere. Run `/counter` in the channel you want."

    return (
        f"✅ Monitoring **{channel.name}** · max utterance **{chunk_seconds:g}s** · "
        f"sounds **{'on' if sound_enabled else 'off'}**{warn}"
    )


def core_stop_monitor(guild) -> str:
    sess = sessions.get(guild.id)
    if not sess:
        return "No active monitoring session."
    sess["running"] = False
    elapsed = stop_call_timer(guild.id)
    try:
        sess["task"].cancel()
    except Exception:
        pass
    sessions.pop(guild.id, None)
    msg = "🛑 Stopped monitoring and left voice."
    if elapsed:
        msg += f" Logged +{format_duration(elapsed)} this session."
    return msg


def core_status(guild) -> str:
    sess = sessions.get(guild.id)
    cfg = guild_cfg(guild.id)
    total_seconds, running = total_monitored_seconds(guild.id)

    lines = [f"**Monitoring:** {'🟢 active' if sess else '⚪ idle'}"]
    if sess:
        vc = sess.get("vc")
        ch = getattr(vc, "channel", None)
        lines.append(f"**Channel:** {ch.name if ch else 'unknown'}")
        lines.append(f"**Max utterance:** {sess.get('chunk_seconds', '?')}s")
        lines.append(f"**Sounds:** {'on' if sess.get('play_sounds') else 'off'}")

    counter_id = cfg.get("counter_channel_id")
    lines.append(f"**Counter channel:** {f'<#{counter_id}>' if counter_id else '❌ not set (run `/counter`)'}")
    live_id = cfg.get("live_channel_id")
    lines.append(f"**Live feed:** {f'<#{live_id}>' if live_id else 'off (`/live`)'}")
    term_counts = " · ".join(
        f"G{i + 1}: {len(g['words']) + len(g['phrases'])}" for i, g in enumerate(group_terms)
    )
    lines.append(f"**Terms loaded:** {term_counts}")
    lines.append(f"**Time tracked:** {format_duration(total_seconds)}{' (running)' if running else ''}")
    lines.append(f"**ASR backend:** {ASR_BACKEND}")

    sound_bits = []
    for grp in (1, 2, 3):
        sid = get_group_sound_id(guild.id, grp)
        sound_bits.append(f"G{grp}: {f'`{sid}`' if sid else '—'}")
    lines.append("**Sounds configured:** " + " · ".join(sound_bits))
    return "\n".join(lines)


def build_stats_embed(guild) -> discord.Embed:
    cfg = guild_cfg(guild.id)
    total_seconds, running = total_monitored_seconds(guild.id)
    incidents = cfg.get("incidents", 0)
    major_incidents = cfg.get("major_incidents", 0)
    embed = discord.Embed(
        title="Voice Moderation Stats",
        description="Live view of voice monitoring totals for this guild.",
        colour=discord.Color.green() if running else discord.Color.orange(),
    )
    embed.add_field(
        name="All-Time Totals",
        value=(
            f"Incidents: **{state['total_incidents']}**\n"
            f"Major incidents: **{state.get('total_major_incidents', 0)}**"
        ),
        inline=False,
    )
    embed.add_field(
        name="Guild Totals",
        value=f"Incidents: **{incidents}**\nMajor incidents: **{major_incidents}**",
        inline=True,
    )
    embed.add_field(
        name="Group Counts",
        value="\n".join(f"Group {i}: **{cfg.get(f'group{i}_incidents', 0)}**" for i in range(1, 4)),
        inline=True,
    )
    embed.add_field(
        name="Monitoring",
        value=(
            f"Time tracked: **{format_duration(total_seconds)}**\n"
            f"Status: **{'Active' if running else 'Idle'}**"
        ),
        inline=True,
    )
    embed.add_field(
        name="Per-Minute Averages",
        value=(
            f"Incidents/min: **{compute_per_minute(incidents, total_seconds):.2f}**\n"
            f"Major/min: **{compute_per_minute(major_incidents, total_seconds):.2f}**"
        ),
        inline=False,
    )
    embed.set_footer(text="Tracking now" if running else "Start with /monitor start")
    return embed


async def core_set_counter(guild, channel) -> str:
    cfg = guild_cfg(guild.id)
    cfg["counter_channel_id"] = channel.id
    cfg["counter_message_id"] = None
    save_state()
    await update_counter_message(guild)
    return f"✅ Incident messages and the dashboard will go to {channel.mention}."


def core_set_modlog(guild, channel) -> str:
    cfg = guild_cfg(guild.id)
    cfg["mod_log_channel_id"] = channel.id
    save_state()
    return f"✅ Mod log channel set to {channel.mention}."


async def core_reset(guild) -> str:
    state["total_incidents"] = 0
    state["total_major_incidents"] = 0
    for grp in (1, 2, 3):
        state[f"total_group{grp}_incidents"] = 0
    for gid, cfg in state["guilds"].items():
        cfg["incidents"] = 0
        cfg["major_incidents"] = 0
        for grp in (1, 2, 3):
            cfg[f"group{grp}_incidents"] = 0
        cfg["call_seconds"] = 0.0
        # Restart (rather than kill) the timer for guilds with an active session.
        cfg["call_started_at"] = time.time() if int(gid) in sessions else None
    save_state()
    await update_counter_message(guild)
    return "✅ Reset all counts and tracked time."


async def core_list_sounds(guild) -> str:
    try:
        available = await guild.fetch_sounds()
    except Exception:
        logger.exception("Failed listing soundboard sounds")
        return "Couldn't list soundboard sounds (missing permission or unsupported)."

    if not available:
        return "This server has no soundboard sounds."

    configured = {}
    for grp in (1, 2, 3):
        sid = get_group_sound_id(guild.id, grp)
        if sid:
            configured.setdefault(str(sid), []).append(f"G{grp}")

    lines = []
    for s in available:
        tags = configured.get(str(s.id))
        lines.append(f"`{s.id}`  {s.name}" + (f"  ← **{', '.join(tags)}**" if tags else ""))

    body = "\n".join(lines)
    if len(body) > 1700:
        body = body[:1700] + "\n…(truncated)"
    return f"**Soundboard sounds in {guild.name}:**\n{body}\n\nAssign one with `/sounds set`."


async def core_set_sound(guild, group: int, sound_id: str) -> str:
    try:
        resolved = await fetch_soundboard_sound(guild, int(sound_id))
    except (TypeError, ValueError):
        return f"`{sound_id}` isn't a valid sound ID. Use `/sounds list` to see them."
    if resolved is None:
        return f"No soundboard sound with ID `{sound_id}` in this server. Try `/sounds list`."
    set_group_sound_id(guild.id, group, sound_id)
    return f"✅ Group {group} (`terms{group}.txt`) will now play **{resolved.name}**."


async def core_test_sound(guild, author, group: int) -> str:
    sound_id_raw = get_group_sound_id(guild.id, group)
    steps = [f"Global toggle: {'on' if ENABLE_SOUNDBOARD_SFX else '**OFF** (ENABLE_SOUNDBOARD_SFX)'}"]
    if not sound_id_raw:
        steps.append(f"Group {group} sound: **not set** — assign one with `/sounds set`.")
        return "\n".join(steps)
    steps.append(f"Group {group} sound ID: `{sound_id_raw}`")

    # Prefer the monitoring session's voice client; otherwise join the caller's channel
    # temporarily, since Discord only accepts the sound while connected.
    sess = sessions.get(guild.id)
    vc = sess.get("vc") if sess else None
    channel = getattr(vc, "channel", None)
    temp_vc = None
    if channel is None:
        if not getattr(author, "voice", None) or not author.voice.channel:
            steps.append("Voice: **not monitoring and you aren't in a VC** — join one and retry.")
            return "\n".join(steps)
        channel = author.voice.channel
        try:
            existing = guild.voice_client
            temp_vc = existing if (existing and existing.is_connected()) else await channel.connect()
            if temp_vc.channel != channel:
                await temp_vc.move_to(channel)
        except Exception as exc:
            steps.append(f"Voice connect: ❌ {type(exc).__name__}: {exc}")
            return "\n".join(steps)
    steps.append(f"Voice channel: **{channel.name}**")

    try:
        sound = await fetch_soundboard_sound(guild, int(sound_id_raw))
        if sound is None:
            steps.append("Resolve sound: ❌ that ID isn't a sound in this server. Try `/sounds list`.")
            return "\n".join(steps)
        steps.append(f"Resolved sound: **{sound.name}**")
        try:
            await channel.send_soundboard_sound(sound)
            steps.append("Playback: ✅ sent")
        except Exception as exc:
            logger.exception("testsound playback failed")
            steps.append(f"Playback: ❌ {type(exc).__name__}: {exc}")
        return "\n".join(steps)
    finally:
        if temp_vc is not None:
            # Let the clip actually play before dropping the connection.
            await asyncio.sleep(3)
            try:
                await temp_vc.disconnect(force=True)
            except Exception:
                pass


def core_list_terms(group: int) -> str:
    terms = read_group_lines(group)
    if not terms:
        return f"Group {group} (`terms{group}.txt`) is empty. Add one with `/terms add`."
    words = [t for t in terms if " " not in t]
    phrases_ = [t for t in terms if " " in t]
    parts = [f"**Group {group}** (`terms{group}.txt`) — {len(terms)} entries"]
    if words:
        parts.append("**Words:** " + ", ".join(f"`{w}`" for w in words))
    if phrases_:
        parts.append("**Phrases:** " + ", ".join(f"`{p}`" for p in phrases_))
    body = "\n".join(parts)
    return body[:1900] + ("\n…(truncated)" if len(body) > 1900 else "")


def core_set_live(guild, channel) -> str:
    cfg = guild_cfg(guild.id)
    if channel is None:
        cfg["live_channel_id"] = None
        save_state()
        return "🛑 Live transcription feed off."
    cfg["live_channel_id"] = channel.id
    save_state()
    return (
        f"📡 Live transcription feed on in {channel.mention}.\n"
        "Every transcript is posted — ⚪ means nothing matched, 🔴 shows which terms hit. "
        "If a word is heard wrong, add the mishearing with `/terms add`."
    )


async def term_autocomplete(ctx: discord.AutocompleteContext):
    """Autofill existing terms of the chosen group, so removal needs no typing."""
    try:
        group = int(ctx.options.get("group") or 1)
    except (TypeError, ValueError):
        group = 1
    if group not in (1, 2, 3):
        group = 1
    query = (ctx.value or "").lower()
    return [t for t in read_group_lines(group) if query in t.lower()][:25]


async def sound_autocomplete(ctx: discord.AutocompleteContext):
    """Autofill the sound picker with this server's actual soundboard sounds."""
    guild = ctx.interaction.guild
    if guild is None:
        return []
    try:
        available = await guild.fetch_sounds()
    except Exception:
        return []
    query = (ctx.value or "").lower()
    return [
        discord.OptionChoice(name=s.name[:100], value=str(s.id))
        for s in available
        if query in s.name.lower()
    ][:25]


# -------- Slash commands --------
monitor_group = bot.create_group("monitor", "Start/stop voice monitoring")
sounds_group = bot.create_group("sounds", "Configure the soundboard clips")


@monitor_group.command(name="start", description="Join a voice channel and start counting flagged words")
@discord.default_permissions(moderate_members=True)
async def slash_monitor_start(
    ctx: discord.ApplicationContext,
    channel: discord.Option(
        discord.VoiceChannel,
        "Voice channel to monitor (defaults to the one you're in)",
        required=False,
    ) = None,
    sounds: discord.Option(bool, "Play a soundboard clip on each detection", required=False) = True,
    chunk_seconds: discord.Option(
        float, "Max seconds of one utterance before it's transcribed anyway (3-30, default 15)",
        required=False, min_value=3, max_value=30,
    ) = 15.0,
):
    await ctx.defer()
    msg = await core_start_monitor(ctx.guild, ctx.author, channel, chunk_seconds, sounds)
    await ctx.respond(msg)


@monitor_group.command(name="stop", description="Stop monitoring and leave voice")
@discord.default_permissions(moderate_members=True)
async def slash_monitor_stop(ctx: discord.ApplicationContext):
    await ctx.respond(core_stop_monitor(ctx.guild))


@monitor_group.command(name="status", description="Show what the bot is currently doing")
async def slash_monitor_status(ctx: discord.ApplicationContext):
    await ctx.respond(core_status(ctx.guild), ephemeral=True)


@sounds_group.command(name="list", description="List this server's soundboard sounds and their IDs")
async def slash_sounds_list(ctx: discord.ApplicationContext):
    await ctx.defer(ephemeral=True)
    await ctx.respond(await core_list_sounds(ctx.guild), ephemeral=True)


@sounds_group.command(name="set", description="Choose which soundboard clip a term group plays")
@discord.default_permissions(moderate_members=True)
async def slash_sounds_set(
    ctx: discord.ApplicationContext,
    group: discord.Option(int, "Which term group", choices=GROUP_CHOICES),
    sound: discord.Option(str, "Sound to play", autocomplete=sound_autocomplete),
):
    await ctx.defer()
    await ctx.respond(await core_set_sound(ctx.guild, group, sound))


@sounds_group.command(name="test", description="Play a group's sound now to verify it works")
@discord.default_permissions(moderate_members=True)
async def slash_sounds_test(
    ctx: discord.ApplicationContext,
    group: discord.Option(int, "Which term group", choices=GROUP_CHOICES) = 1,
):
    await ctx.defer()
    await ctx.respond(await core_test_sound(ctx.guild, ctx.author, group))


terms_group = bot.create_group("terms", "View and edit the flagged word lists")


@terms_group.command(name="list", description="Show the words and phrases in a group")
async def slash_terms_list(
    ctx: discord.ApplicationContext,
    group: discord.Option(int, "Which term group", choices=GROUP_CHOICES) = 1,
):
    await ctx.respond(core_list_terms(group), ephemeral=True)


@terms_group.command(name="add", description="Add a word or phrase to a group (takes effect immediately)")
@discord.default_permissions(moderate_members=True)
async def slash_terms_add(
    ctx: discord.ApplicationContext,
    group: discord.Option(int, "Which term group", choices=GROUP_CHOICES),
    term: discord.Option(str, "Word or phrase to flag (e.g. a mishearing you saw in the live feed)"),
):
    changed, msg = add_group_term(group, term)
    await ctx.respond(msg, ephemeral=not changed)


@terms_group.command(name="remove", description="Remove a word or phrase from a group")
@discord.default_permissions(moderate_members=True)
async def slash_terms_remove(
    ctx: discord.ApplicationContext,
    group: discord.Option(int, "Which term group", choices=GROUP_CHOICES),
    term: discord.Option(str, "Term to remove", autocomplete=term_autocomplete),
):
    changed, msg = remove_group_term(group, term)
    await ctx.respond(msg, ephemeral=not changed)


@terms_group.command(name="reload", description="Re-read the term files from disk")
@discord.default_permissions(moderate_members=True)
async def slash_terms_reload(ctx: discord.ApplicationContext):
    reload_group_terms()
    counts = " · ".join(
        f"G{i + 1}: {len(g['words']) + len(g['phrases'])}" for i, g in enumerate(group_terms)
    )
    await ctx.respond(f"🔄 Reloaded term files — {counts}")


@bot.slash_command(name="live", description="Stream every transcription to a channel to spot mishearings")
@discord.default_permissions(moderate_members=True)
async def slash_live(
    ctx: discord.ApplicationContext,
    enabled: discord.Option(bool, "Turn the live feed on or off") = True,
    channel: discord.Option(
        discord.TextChannel, "Where to post transcripts (defaults to here)", required=False
    ) = None,
):
    await ctx.respond(core_set_live(ctx.guild, (channel or ctx.channel) if enabled else None))


@bot.slash_command(name="counter", description="Set the channel where detections are posted")
@discord.default_permissions(moderate_members=True)
async def slash_counter(
    ctx: discord.ApplicationContext,
    channel: discord.Option(
        discord.TextChannel, "Channel for incident messages (defaults to here)", required=False
    ) = None,
):
    await ctx.defer()
    await ctx.respond(await core_set_counter(ctx.guild, channel or ctx.channel))


@bot.slash_command(name="stats", description="Show incident counts and monitoring time")
async def slash_stats(ctx: discord.ApplicationContext):
    await ctx.respond(embed=build_stats_embed(ctx.guild))


@bot.slash_command(name="setmodlog", description="Set the mod log channel")
@discord.default_permissions(moderate_members=True)
async def slash_setmodlog(
    ctx: discord.ApplicationContext,
    channel: discord.Option(discord.TextChannel, "Mod log channel", required=False) = None,
):
    await ctx.respond(core_set_modlog(ctx.guild, channel or ctx.channel))


@bot.slash_command(name="resetcounts", description="Reset all incident counts and tracked time")
@discord.default_permissions(administrator=True)
async def slash_resetcounts(ctx: discord.ApplicationContext):
    await ctx.defer()
    await ctx.respond(await core_reset(ctx.guild))


# -------- Prefix commands (kept for back-compat) --------
@bot.command(name="setmodlog")
@commands.guild_only()
async def setmodlog(ctx: commands.Context):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    await ctx.reply(core_set_modlog(ctx.guild, ctx.channel))


@bot.command(name="counter")
@commands.guild_only()
async def counter(ctx: commands.Context):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    await ctx.reply(await core_set_counter(ctx.guild, ctx.channel))


@bot.command(name="count")
@commands.guild_only()
async def count(ctx: commands.Context):
    await ctx.reply(embed=build_stats_embed(ctx.guild))


@bot.command(name="status")
@commands.guild_only()
async def status(ctx: commands.Context):
    await ctx.reply(core_status(ctx.guild))


@bot.command(name="monitor")
@commands.guild_only()
async def monitor(ctx: commands.Context, *, args: str = ""):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    try:
        tokens = shlex.split(args)
    except ValueError:
        return await ctx.reply("Malformed options (check your quotes). Example: !monitor start channel=General sound")

    action = tokens[0].lower().strip() if tokens else "start"
    options = tuple(tokens[1:]) if tokens else ()

    if action == "stop":
        return await ctx.reply(core_stop_monitor(ctx.guild))
    if action == "status":
        return await ctx.reply(core_status(ctx.guild))

    # Extract optional channel override tokens like channel=<name> or a channel mention.
    channel_target = None
    filtered_opts: list[str] = []
    channel_mention = re.compile(r"^<#(\d+)>$")
    for opt in options:
        if opt.lower().startswith(("channel=", "vc=")):
            channel_target = opt.split("=", 1)[1].strip()
            continue
        m = channel_mention.match(opt)
        if m:
            channel_target = m.group(1)
            continue
        filtered_opts.append(opt)

    try:
        chunk_seconds, sound_enabled = parse_monitor_options(tuple(filtered_opts))
    except ValueError as bad_opt:
        return await ctx.reply(
            f"Unrecognized monitor option: `{bad_opt.args[0]}`. Use numbers for chunk seconds "
            "and words like `sound`/`nosound`."
        )

    target_channel = None
    if channel_target:
        # Allow numeric id, channel mention, or case-insensitive name match.
        if channel_target.isdigit():
            target_channel = ctx.guild.get_channel(int(channel_target))
        if target_channel is None:
            for vc in ctx.guild.voice_channels:
                if vc.name.lower() == channel_target.lower():
                    target_channel = vc
                    break
        if target_channel is None:
            return await ctx.reply(f"Couldn't find voice channel `{channel_target}`.")

    await ctx.reply(
        await core_start_monitor(ctx.guild, ctx.author, target_channel, chunk_seconds, sound_enabled)
    )


@bot.command(name="sounds")
@commands.guild_only()
async def sounds_cmd(ctx: commands.Context):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    await ctx.reply(await core_list_sounds(ctx.guild))


@bot.command(name="testsound")
@commands.guild_only()
async def testsound(ctx: commands.Context, group: int = 1):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    if group not in (1, 2, 3):
        return await ctx.reply("Usage: `!testsound 1|2|3`")
    await ctx.reply(await core_test_sound(ctx.guild, ctx.author, group))


@bot.command(name="setsound")
@commands.guild_only()
async def setsound(ctx: commands.Context, group: int, sound_id: str):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    if group not in (1, 2, 3):
        return await ctx.reply("Usage: `!setsound <1|2|3> <sound_id>`")
    await ctx.reply(await core_set_sound(ctx.guild, group, sound_id))


@bot.command(name="terms")
@commands.guild_only()
async def terms_cmd(ctx: commands.Context, action: str = "list", group: int = 1, *, term: str = ""):
    action = action.lower().strip()
    if group not in (1, 2, 3):
        return await ctx.reply("Group must be 1, 2 or 3.")
    if action == "list":
        return await ctx.reply(core_list_terms(group))
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    if action == "reload":
        reload_group_terms()
        return await ctx.reply("🔄 Reloaded term files.")
    if action == "add":
        if not term:
            return await ctx.reply("Usage: `!terms add <1|2|3> <word or phrase>`")
        return await ctx.reply(add_group_term(group, term)[1])
    if action == "remove":
        if not term:
            return await ctx.reply("Usage: `!terms remove <1|2|3> <word or phrase>`")
        return await ctx.reply(remove_group_term(group, term)[1])
    await ctx.reply("Usage: `!terms list|add|remove|reload <1|2|3> [term]`")


@bot.command(name="live")
@commands.guild_only()
async def live_cmd(ctx: commands.Context, switch: str = "on"):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    on = switch.lower().strip() not in SOUND_OFF_WORDS
    await ctx.reply(core_set_live(ctx.guild, ctx.channel if on else None))


@bot.command(name="resetcounts")
@commands.guild_only()
async def resetcounts(ctx: commands.Context):
    if not is_mod(ctx):
        return await ctx.reply("Mods only.")
    await ctx.reply(await core_reset(ctx.guild))


@bot.event
async def on_ready():
    global single_words, phrases, major_terms, WORD_REGEX, group_terms, GROUP_REGEX
    single_words, phrases, major_terms = load_terms_file(TERMS_PATH, treat_last_two_as_major=True)
    group_terms = load_group_terms()

    has_main_terms = bool(single_words or phrases)
    has_any_group_terms = any(g["words"] or g["phrases"] for g in group_terms)

    if not has_main_terms:
        if has_any_group_terms:
            logger.info("No base terms in %s; operating with group term files only.", TERMS_PATH)
        else:
            logger.warning("No terms configured; %s and group term files are empty or missing.", TERMS_PATH)

    for idx, g in enumerate(group_terms, start=1):
        if not g["words"] and not g["phrases"]:
            logger.info("Group%d terms file %s is empty or missing; group%d counts will stay at zero.", idx, g["path"], idx)
    GROUP_REGEX = [build_regex(g["words"]) for g in group_terms]
    WORD_REGEX = build_regex(single_words)
    logger.info(
        "Logged in as %s | terms: %d words, %d phrases | debug=%s",
        bot.user,
        len(single_words),
        len(phrases),
        DEBUG,
    )
    await sync_slash_commands()


async def sync_slash_commands():
    """Register slash commands per-guild: they appear instantly, unlike global ones
    which Discord can take up to an hour to propagate."""
    guild_ids = [g.id for g in bot.guilds]
    if not guild_ids:
        logger.warning("Bot is not in any guild; no slash commands registered.")
        return
    try:
        await bot.sync_commands(guild_ids=guild_ids)
        logger.info("Slash commands synced to %d guild(s): %s", len(guild_ids), guild_ids)
    except Exception:
        logger.exception("Failed syncing slash commands")


@bot.event
async def on_guild_join(guild: discord.Guild):
    try:
        await bot.sync_commands(guild_ids=[guild.id])
        logger.info("Slash commands synced to new guild %s", guild.id)
    except Exception:
        logger.exception("Failed syncing slash commands to new guild %s", guild.id)


bot.run(TOKEN)
