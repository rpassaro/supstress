# supstress

PsychoPy working-memory paradigm (supstress2).

## Local-only directories (not in repo)

These are excluded via `.gitignore` — copy manually across machines:

- `stimuli/` — static image sets (faces, places, fruits). ~850MB, never changes.
- `data/` — participant output (.csv/.log/.psydat).

## Setup on a new machine

1. Clone this repo.
2. Copy `stimuli/` and (if needed) `data/` into the repo root from the other machine (AirDrop / external drive / rsync).
3. Open `supstress2_maintask.psyexp` in PsychoPy.
