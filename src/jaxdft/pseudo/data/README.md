# Pseudopotential data

Files here are vendored verbatim and read only by the parsers in
`jaxdft/pseudo/` (CLAUDE.md: nothing outside a parser reads file formats).
Their SHA-256 hashes are pinned by `tests/unit/test_gth.py` (hydrogen) and
`tests/unit/test_hgh_nonlocal.py` (silicon).

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

## `hgh_lda/si-q4.hgh`

HGH LDA pseudopotential for silicon, Z_ion = 4, local part plus separable
projectors in the s (two) and p (one) channels, in the same CP2K-style format.
No spin-orbit coefficients.

- Source: JuliaMolSim PseudoLibrary, `pseudos/hgh_lda_hgh/si-q4.hgh`, branch
  `main`, downloaded 2026-11-11 from
  `https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/main/pseudos/hgh_lda_hgh/si-q4.hgh`.
  This is the file DFTK uses for HGH LDA silicon.
- SHA-256: `9d7cca22d6a7873e001af2c31998cdeb570a7468ae5dacab37f8ae5db2220d4b`,
  the hash recorded in `references.bib` (`pseudolibrary_si`) when the 2b theory
  post was written, and confirmed independently by its review.
- Values: r_loc = 0.44 Bohr, C1 = −7.33610297 Ha; r_s = 0.42273813 Bohr,
  h_s = ((5.90692831, −1.26189397), (−1.26189397, 3.25819622)) Ha;
  r_p = 0.48427842 Bohr, h_p = 2.72701346 Ha (2b theory post).
