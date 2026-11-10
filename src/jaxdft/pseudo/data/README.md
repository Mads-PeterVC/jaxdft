# Pseudopotential data

Files here are vendored verbatim and read only by the parsers in
`jaxdft/pseudo/` (CLAUDE.md: nothing outside a parser reads file formats).
Their SHA-256 hashes are pinned by `tests/unit/test_gth.py`.

## `hgh_lda/h-q1.hgh`

HGH (Hartwigsen–Goedecker–Hutter, PRB 58, 3641 (1998)) LDA pseudopotential
for hydrogen, Z_ion = 1, local part only (no projectors), in the CP2K-style
GTH text format.

- Source: JuliaMolSim PseudoLibrary, `pseudos/hgh_lda_hgh/h-q1.hgh`, branch
  `main`, downloaded 2026-11-06 from
  `https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/main/pseudos/hgh_lda_hgh/h-q1.hgh`.
  This is the file DFTK uses for HGH LDA hydrogen.
- SHA-256: `2f4069ba8ea992435a2c614bed3b2a1ed18b808fc1ffc5d7a9dcaa519f1f5969`.
- Values: r_loc = 0.2 Bohr, C1 = −4.18023680 Ha, C2 = 0.72507482 Ha. These are
  identical to the `GTH-PADE-q1` entry for H in CP2K's `data/GTH_POTENTIALS`,
  and to six decimals to ABINIT's HGH file (2a theory post and its review).
- The repository has no licence file. The file contains only the published
  parameters of [HGH 1998], which are scientific data.
