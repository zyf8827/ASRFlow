# Test audio fixtures

This directory holds small, publicly attributable speech samples used by demos,
benchmarks, health checks, and e2e scripts.

## `sample_16k.wav` (committed)

| Field | Value |
| --- | --- |
| Format | WAV, 16-bit PCM, little-endian |
| Sample rate | 16,000 Hz, mono |
| Duration | ~11.0 s |
| Size | ~344 KiB |
| Language | English (public-domain presidential speech excerpt) |
| Spoken content | JFK inaugural address excerpt (“ask not what your country can do for you…”) |
| Upstream file | [`tests/jfk.flac`](https://github.com/openai/whisper/blob/main/tests/jfk.flac) in the [OpenAI Whisper](https://github.com/openai/whisper) repository |
| Download URL (verified) | `https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac` |
| Alternate CDN URL | `https://cdn.jsdelivr.net/gh/openai/whisper@main/tests/jfk.flac` |
| Upstream SHA-256 | `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715` |
| Committed WAV SHA-256 | `fa9c009558aa3f214e4e8d442e16a3153c138065364871d24b6e54d55c70cd68` |
| License | Recording is a public-domain U.S. government work; Whisper repository is MIT. Converted locally to 16 kHz mono PCM for ASR plumbing tests. |

> Note: This fixture is **English**. Chinese ASR quality evaluation should use a
> public Mandarin corpus (e.g. AISHELL-1) via `AUDIO=/path/to.wav` — do not
> commit large datasets here.

## Regenerate / re-fetch

```bash
bash scripts/fetch_fixtures.sh
```

The script downloads the upstream FLAC from the verified URL above, checks the
SHA-256, converts with `ffmpeg` to `fixtures/sample_16k.wav`, and verifies the
WAV checksum.

## Optional public sources (not committed)

| Source | License | Notes |
| --- | --- | --- |
| [LibriSpeech](https://www.openslr.org/12/) test-clean (OpenSLR) | CC BY 4.0 | Full tarball is large (~350 MB); extract a single utterance if needed |
| [Google Speech Commands](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip) mini set | See dataset README (Speech Commands) | Short keyword clips, already 16 kHz mono |
| [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) | Creative Commons (see repo) | Very short digit clips at 8 kHz |

Do **not** replace these fixtures with proprietary, company-recorded, KYC,
finance, law-enforcement, or otherwise non-public audio.
