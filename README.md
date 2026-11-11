# Voice moderation bot

A Discord bot that sits in a voice channel, transcribes each speaker as they talk and counts how
often terms from lists the server's moderators keep are said. It posts each detection to a counter
channel and keeps running totals per server.

- Built on py-cord 2.8, which records each speaker on their own track.
- Speech is cut into utterances by watching for pauses, and a long stretch of talk is re-scanned
  while it goes on, so a term said early in a long run is caught within about a second.
- Transcription runs on the GPU, with NVIDIA's Parakeet (through `onnx-asr`) or faster-whisper.
  `ASR_COMPARE` runs both and logs where they disagree.
- Terms come in three groups. Each group can have its own message style and a soundboard clip.

## The word lists are not in this repository

The terms the bot listens for are kept by each server's moderators and stay on the machine that runs
the bot. Put one term or phrase per line in `terms.txt` (the base list; its last two lines count as
"major" terms) and in `terms1.txt`, `terms2.txt` and `terms3.txt` for the groups. They can also be
changed live with `/terms add` and `/terms remove`. `bias_terms.txt` holds spellings that help the
recogniser hear the terms.

## Running it

1. Create a bot in the Discord developer portal with the Message Content intent turned on, and put its
   token in `.env` as `DISCORD_TOKEN=...`.
2. Make a virtual environment and install `py-cord[voice]`, `faster-whisper`, `onnx-asr[gpu]`,
   `numpy`, `soundfile`, `librosa`, `python-dotenv` and, for online VAD only, `torch`.
3. Add the term files, then run `python bot.py` (or `start_bot.bat`, which writes a log under `logs/`).

Optional settings in `.env`: `ASR_BACKEND` (`parakeet` or `whisper`), `WHISPER_MODEL`, `PARAKEET_MODEL`,
`PARAKEET_DEVICE`, `PARAKEET_THREADS`, `ASR_COMPARE`, `USE_ONLINE_VAD`, `DETECT_INTERVAL_SECONDS`,
`ENABLE_SOUNDBOARD_SFX`, `DUMP_AUDIO_DIR` and `DEBUG`.

## Commands

| Command | What it does |
| --- | --- |
| `/monitor start`, `stop`, `status` | Join a voice channel and start counting, leave, or report what it is doing |
| `/terms list`, `add`, `remove`, `reload` | Read and change a group's terms |
| `/sounds list`, `set`, `test` | Pick the soundboard clip a group plays |
| `/counter` | Choose the channel detections are posted to |
| `/live` | Stream every transcription to a channel, to spot mishearings |
| `/stats`, `/resetcounts`, `/setmodlog` | Counts and monitored time, a reset, and the mod log channel |

## Credits

`whisper_online.py` is adapted from [whisper_streaming](https://github.com/ufal/whisper_streaming)
by ÚFAL, under the MIT License (see `LICENSE-whisper_streaming`).
