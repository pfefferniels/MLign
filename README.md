# MLign — symbolic score→performance alignment

Goal: MEI score + MIDI performance → (close-to) perfect note-level alignment.
Approach: transformer-based note matching trained on synthetic espressivo-rendered performances, beating parangonar.

- research/ — literature & code-study reports
- src/ — the aligner
- scripts/ — data generation, training
- data/ — corpora (gitignored)
- eval/ — benchmark harness vs parangonar/AlignmentTool

