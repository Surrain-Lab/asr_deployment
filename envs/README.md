# Conda environments

The pipeline needs two environments, and they are mutually incompatible &mdash; VTC1
depends on a Python 3.8 era pyannote-audio that cannot coexist with the modern
one WhisperX uses. That is why the runner invokes each step through
`conda run -n <env>` instead of expecting one environment to serve everything.

| Environment | Python | Used for |
|---|---|---|
| `whisperx` | 3.10 | WhisperX transcription, evaluation, langdetect, Excel |
| `pyannote` | 3.8 | VTC1's `apply.sh` only |

VTC2 needs neither: it is invoked through `uv`, which manages its own venv.

## Recreating them

```bash
conda env create -f envs/whisperx.yml
conda env create -f envs/pyannote.yml
```

These are exact exports (`conda env export --no-builds`) from a working
installation, so they pin versions rather than resolving fresh. That is
deliberate: `pyannote-audio==0+unknown` in the VTC1 environment is an editable
install straight from the voice-type-classifier checkout, and a fresh resolve
will not reproduce it.

If `conda env create` fails on the pyannote environment &mdash; likely, as those
pins age &mdash; build it from the model repository instead, which is the documented
route:

```bash
cd <vtc1_model>          # the voice-type-classifier checkout
conda env create -f vtc.yml
conda activate pyannote
pyannote-audio --version # must print a version
```

## Verifying

`bin/doctor.sh` checks both environments exist and reports which is missing.
It does not check package contents, so a successful create is not proof the
environment works &mdash; run a small job and confirm it produces output. Every
step that has been observed to exit 0 while writing nothing is guarded, so a
broken environment surfaces as a failed step rather than an empty result.
