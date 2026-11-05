# Cohen & Bergstresser (1966), Table I, silicon

`si_table1.json` holds the Si row of Table I of

> M. L. Cohen and T. K. Bergstresser, *Band Structures and Pseudopotential Form
> Factors for Fourteen Semiconductors of the Diamond and Zinc-blende Structures*,
> Phys. Rev. **141**, 789 (1966). doi:10.1103/PhysRev.141.789

**How it was produced.** It was transcribed by hand from a scan of the paper.
No script was involved. The paper itself is not in the repository (copyright);
keep a local copy in `sources/`, which is gitignored.

**Reading the table.** Energies are in eV to 0.1 eV, measured from the
valence-band maximum Γ₂₅′. For each substance the experimental value is printed
directly above the calculated value. The Si row gives experiment for
Γ₂₅′–Γ₁₅, Γ₂₅′–X₁(Δ₁), L₃′–L₁, L₃′–L₃ and X₄–X₁. It gives calculated values
for all seven entries. The Γ₂₅′–Γ₂′ entry, the tightest in the acceptance
set, was checked twice against the scan; it reads 3.8.

**Use.** The *calculated* values are the Milestone 1b acceptance data. The
experimental values are for interpretation only. Tolerances are derived in
`docs/theory/01b_cohen_bergstresser.qmd`: 0.15 eV for `from_vbm` entries and
0.25 eV for `between_levels` entries. They are defined in
`tests/support/tolerances.py`.
