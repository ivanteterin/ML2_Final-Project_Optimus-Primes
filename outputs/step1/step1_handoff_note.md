# Step 1 Handoff Note

## Data scope
- VIDEO_RGB/backhand
- VIDEO_RGB/forehand_flat
- VIDEO_RGB/kick_service
- VIDEO_RGB/smash

## Core guarantees
- Metadata generated from repo tree and parsed filenames.
- Skill labels checked against subject-id rule.
- Subject-disjoint split with leakage assertions.
- Split stats exported by skill and action.

## Next-step training requirement
- Do not reshuffle/re-split by video rows.
- Use split column from metadata_rgb_multi4_with_split.csv.
- Report both overall skill metrics and per-action skill metrics.
