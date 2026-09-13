# Reference lock

The initial port was analyzed against these exact revisions:

| Reference | Revision | Role |
|---|---|---|
| multimodal-art-projection/YuE | `88da114a67df892af0329472073b96a5ef700b93` | Official YuE2 pipeline and cover contract |
| m-a-p/SheetSage2 | `eab522a8168e8b8b8c4856bf8609cd86198f01fe` | Audio-to-symbolic model, grammar, and exporters |
| m-a-p/MERT-v2-FullSong | `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42` | Audio encoder pinned by SheetSage2 |
| 0xShug0/audio.cpp dev | `fbe3eedbf6c504e45189e2cdcf1b257740a28863e` | Independent GGML YuE2 and partial SheetSage2 reference |
| betweentwomidnights/ggml | `fff93d2714e934822100586ce241267e8cc821af` | Shared execution/training backend |

The upstream repos are references, not source dependencies. `audio.cpp` is
especially useful for tensor naming and BART/YuE2 parity ideas, but its YuE2
session is generation-only and its SheetSage2 work currently exposes a decoder
parity runtime rather than the complete audio-to-ABC path.

## Weight provenance

The official configuration pins MERT-v2-FullSong revision
`d8ba1c745e733b3908ce6ad16ebeb17ac7600a42` and SHA-256
`e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d`.
The converter verifies this by default. Converted GGUF files remain derived
from CC BY-NC 4.0 model weights and are not covered by this repository's MIT
code license.

