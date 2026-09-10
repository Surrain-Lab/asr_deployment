# ASR_Benchmark

**The Surrain Lab** &middot; Department of Early Childhood, Multilingual, and Special
Education, University of Nevada, Las Vegas
*Apoyando el desarrollo bilingüe en el hogar y la escuela*

A web front end and job runner for the lab's bilingual (English&ndash;Spanish)
child-ASR pipeline. Point it at folders, press a button, walk away.

Built to run on a GPU server and driven from a browser over an SSH tunnel, it
compares ASR architectures against human CHAT transcripts of child-centred
daylong recordings and reports WER overall, per speaker, and by language.

It **wraps** `Audio_Transcription_Pipeline3.0` rather than replacing it. Every
number it produces comes from the same scripts that produced your existing
results, so output stays comparable with what you have already analysed.

---

## What this needs on the machine

This repository is a front end and job runner. It **wraps** the pipeline rather
than containing it, so a clone alone is not enough. The server also needs:

| Requirement | Notes |
|---|---|
| `Audio_Transcription_Pipeline3.0` | The scripts being wrapped. 27 specific files are required. |
| VTC1 checkout | [voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier), with `apply.sh` |
| VTC2 checkout | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC), with `best.ckpt` fetched via git-lfs |
| conda env for WhisperX | WhisperX, jiwer, langdetect, openpyxl. Called `whisperx` by default. |
| conda env for VTC1 | Python 3.8 + old pyannote-audio. Called `pyannote` by default. Must be separate; the two are incompatible. |
| System tools | `ffmpeg`, `ffprobe`, `sox`, `uv`, `git-lfs` |
| GPU | Optional but strongly recommended. CPU works and is very slow. |

`bin/doctor.sh` checks every one of these and names whatever is missing, so you
find out in seconds rather than several steps into a job.

## Install on a new server

```bash
git clone <this repo> ASR_Benchmark
cd ASR_Benchmark

bash setup.sh                  # creates the 'asrbench' env, copies config.example.yaml
                               # -> config.yaml, then stops so you can edit it

bin/doctor.sh --suggest        # prints paths detected on this machine
$EDITOR config.yaml            # fill them in

bin/doctor.sh                  # must be all-ok before going further
bin/start.sh
```

`config.yaml` is gitignored, so every server keeps its own paths. Nothing else
is machine-specific.

Then, from your laptop:

```bash
ssh -N -L 8000:localhost:8000 <user>@<server>
```

and open <http://localhost:8000>.

The UI binds to `127.0.0.1`, so it is reachable only through that tunnel and
nothing is exposed to the network. There is no authentication, which is exactly
why it must not be bound to `0.0.0.0` without adding some.

| Command | What it does |
|---|---|
| `bin/doctor.sh` | Check this machine can run everything (`--suggest` to detect paths) |
| `bin/start.sh` | Start scheduler + web UI (safe to re-run) |
| `bin/status.sh` | What is up, what is queued, what is running |
| `bin/stop.sh` | Stop the services. **Running jobs keep going.** |

### No sudo required

Everything runs as a normal user. The services are detached with `setsid` rather
than run under systemd, because the machine this was built on has no sudo and
user lingering disabled. If your server does allow systemd user units, a unit
calling `bin/start.sh` works fine and survives reboots, which `setsid` does not.

---

## The five modules

| Module | You give it | You get |
|---|---|---|
| **Human Transcript Cleanup** | annotated CHAT folder | `_clean` / `_tagged` text per clip, per speaker (KCHI, OCH, MAL, FEM, ADULT, CHILDREN), combined per recording, EN/ES word counts |
| **Audio Transcription** | audio folder + a variant | tagged transcripts and speaker-separated text |
| **WER Evaluation** | a reference folder + a hypothesis folder | WER, CER, MER, WIL, WIP, accuracy per clip and per recording |
| **Language & Vocabulary** | a transcript folder | language-corrected tags, EN/ES counts, vocabulary tables, WER by language |
| **Full Benchmark** | audio + annotations | everything above, every selected variant, plus comparison workbooks |

### Pipeline variants

| Variant | Method |
|---|---|
| `whisperx_only` | 5-minute chunks, no speaker labels |
| `whisperx_vtc1` | chunk-transcribe, then attach VTC1 labels by timestamp |
| `whisperx_vtc2` | chunk-transcribe, then attach VTC2 labels **(new)** |
| `vtc1_whisperx` | transcribe each VTC1 segment separately |
| `vtc2_whisperx` | transcribe each VTC2 segment separately |

`whisperx_vtc2` is new in this system. It works because VTC1 and VTC2 write
`rttm_txt` in an identical format and the tagger takes a generic
`--speaker-dir`; only its docstring ever said "VTC1".

---

## Folder layout

Input, one folder per recording, clips inside:

```
<audio folder>/
  <recording-id>/
    <recording-id>_001.wav
    <recording-id>_002.wav
```

Output, one folder per clip:

```
<run>/human_eval/five_mins_benchmark/
  <recording-id>/
    <recording-id>_001/
      <recording-id>_001_clean.txt
      <recording-id>_001_KCHI_clean.txt
      <recording-id>_001_KCHI_tagged.txt
      <recording-id>_001_FEM_clean.txt
      ... MAL, OCH, ADULT, CHILDREN, plus _tagged forms
```

### Pick the right folder level

The corpus is not uniformly nested, and Pipeline 3.0 assumes recordings are
*immediate* subdirectories of whatever it is handed. Handing it the wrong level
produces empty or mis-nested output **with no error**.

So every folder field shows a live preview: how many recordings and clips were
found, which layout it is, and whether the recordings sit below the level you
picked. Two warnings are worth taking seriously:

- **"recordings sit N levels below"** &mdash; handled automatically, just confirming.
- **"recordings come from N different parent folders"** &mdash; you have selected a
  level that merges collections kept apart on disk. This happens when a corpus
  arrives in batches and the annotation root holds one wrapper folder per batch.
  Point at a single batch, unless you really do mean to combine them &mdash; the
  human module flattens whatever you give it, so combining is a valid choice
  when the batches are two halves of one corpus.

---

## How jobs run

Jobs are queued and executed by a scheduler, detached from the web UI. Closing
the browser, dropping the SSH tunnel, and even stopping the services all leave
running jobs alone.

- **Two GPU jobs at a time**, one per L40, pinned with `CUDA_VISIBLE_DEVICES`.
- **Four CPU jobs at a time**, in a separate pool, so a quick WER run never
  queues behind a six-hour transcription.
- Every run writes to `/mnt/fast-scratch/USER/ASR_Benchmark_runs/<id>_<module>_<time>/`
  along with `job.log` and a `job.json` recording exactly what was run.

Results are browsable and downloadable from the job page; the zip skips
normalised audio and chunk intermediates, which are large and regenerable.

---

## Notes on correctness

Things found while building this that are worth knowing:

**Source audio is never modified.** Pipeline 3.0 converted WAVs to 16 kHz mono
in place. Here the normalised copy goes into the run directory instead, and
files already in the right format are symlinked rather than copied. Pointing at
a shared corpus cannot alter it.

**VTC1's output directory is global.** `apply.sh` always writes to
`<model>/output_voice_type_classifier/<recording name>/`, keyed by folder name
alone. Two jobs would overwrite each other, and a silently failing run would
leave a previous run's output to be harvested as though it were fresh. So VTC1
steps hold an exclusive lock, `all.rttm` must be newer than the step that
produced it or the job fails, and the output is copied into the run directory
immediately. Nothing of yours is deleted.

**The Excel and figure scripts hardcode `output_14clips`.** They take no
arguments, so pointed at a new run they would read the old corpus and emit
workbooks that look right but describe different data. `modules/build_reports.py`
rewrites only their `PIPELINES` and `OUTPUT_FILE` assignments, located with
`ast`, so every line of real logic is preserved. The originals are untouched.

**The three `evaluation.py` copies are identical.** So are the three
`combine_calculation` and `count_words` copies &mdash; they differ only in trailing
comments. The WER module therefore works on any two folders.

**The three `langdetect_tags` scripts are *not* identical.** VTC1's handles a
`[SPEECH]` class the others lack; WhisperX's reads only `<clip>_tagged.txt`.
Each pipeline keeps calling its own copy, and the Language module asks which
pipeline produced your transcripts because picking wrong silently changes the
counts.

**Symlinks.** Most project folders under `/home/USER` are symlinks onto the
`USER-research` NFS share. Path checks resolve symlinks so a link cannot be
used to escape the allowed roots, which means the real target must also be
listed in `browse_roots` &mdash; it is.

**Per-language WER needs reading carefully.** The human reference tags lines
`(es)`, `(en)`, `(unknown)` and `(both)`, but langdetect on the hypothesis only
ever emits `en` and `es`. So `unknown` and `both` score 100% WER by construction
rather than by error. The summary reports those rows under "not comparable",
with the share of the corpus they account for, instead of mixing them into the
result.

The bigger caveat: per-language WER conflates two different errors. A word
transcribed correctly but assigned the wrong language counts as a deletion in one
bucket and an insertion in the other, so a pipeline that systematically
over-predicts one language can show a per-language WER above 100% while
transcribing perfectly well. The summary therefore prints the English share of
EN+ES words for both reference and hypothesis next to the WER table, so language
identification and transcription accuracy can be told apart.

**cuDNN on GPU steps.** WhisperX transcription aborts mid-run with
`Unable to load any of {libcudnn_cnn.so.9...}` because ctranslate2 dlopens cuDNN
by name at inference time and cannot find it. This reproduces outside this
system, so it is an environment issue rather than one the wrapper introduced.
Pipeline 3.0 met the same wall and fixed it by prepending the whole conda `lib/`
directory, which shadowed the system `libstdc++` with an older one and broke
every ffmpeg and torchcodec subprocess &mdash; a note in `run_pipeline.py` records
that it was disabled for exactly that reason. The runner instead adds only
`site-packages/nvidia/*/lib`, which contains no `libstdc++`, and only for GPU
steps. Verified: transcription succeeds and `ffprobe` still works.

---

## Configuration

Everything lives in `config.yaml`: script and model locations, the conda envs
(`whisperx` for most work, `pyannote` for VTC1), where runs are written, which
roots the file browser may show, and how many jobs run at once.

## Layout

```
asrbench/
  config.py     config loading and path safety
  db.py         SQLite job store
  discover.py   works out how a chosen folder is nested
  doctor.py     preflight: is this machine able to run anything?
  plan.py       module + parameters -> ordered steps   <- the interesting one
  runner.py     executes one job, writes its log
  worker.py     scheduler: assigns slots, spawns runners
  web/          FastAPI app and templates
modules/        helper scripts, tree normalisers, parameterised report builder
```

`modules/` holds the pieces that exist because the wrapped scripts assume things
the real data does not honour:

| Script | Why it exists |
|---|---|
| `prepare_audio.py` | Flattens nesting and normalises to 16 kHz mono, into the run directory rather than over the source |
| `prepare_transcripts.py` | Same for annotations, and merges a corpus split across wrapper folders |
| `align_trees.py` | Renames hypothesis recording folders to match the reference, so `evaluation.py` does not silently score 100% WER |
| `run_vtc1.py` | Runs `apply.sh` with a staleness guard on its shared output directory |
| `harvest_vtc1.py` | Copies that output into the run directory |
| `build_reports.py` | Rewrites the hardcoded paths in the Excel scripts via `ast` |
| `summarize_wer_by_language.py` | Corpus-level EN/ES rollup, with the not-comparable rows separated out |

`plan.py` is the only file that knows how Pipeline 3.0's scripts fit together.
To add a step or a variant, that is where to look.
