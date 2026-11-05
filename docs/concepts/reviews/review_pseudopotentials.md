# Review — `docs/concepts/pseudopotentials.qmd`

I re-ran the post's physics independently in NumPy/SciPy (model atom, Hankel transforms,
Kerker construction, logarithmic derivatives, HGH form factor, Lindhard screening) and
checked the claims that the prose makes on top of the code. **The physics is correct
throughout** — every derivation, identity and quoted number reproduces. The findings are
one quantitative framing issue, one missing conceptual step, and a set of smaller
precision and clarity items. This is a strong post; most of what follows is about the
seams between sections rather than the sections themselves.

I reviewed the chart *code*, not the rendered output — I cannot execute Quarto or Altair
here, so slider behaviour, layout and legend rendering are unchecked.

## 0. Verified

| Claim | Result |
|---|---|
| `V(r)` is `-Z/r` at the nucleus and `-1/r` far away | correct |
| Tridiagonal discretisation of `-u''/2` | correct |
| Hankel pair and Parseval `∫u² dr = (2/pi)∫ũ²q²dq` | correct convention |
| Model level ordering 1s < 2s < 2p < 3s < 3p, valence near -0.6 and -0.3 Ha | correct (-79.43, -9.12, -8.46, -0.5975, -0.3063 Ha) |
| Real Si core binding energies 1839 / 149.7 / 99.6 eV | match the X-ray tables |
| Kerker form `u_ps = r^(l+1) e^p`, no linear term keeps `V` finite | correct |
| Inverted radial equation `V_ps = eps + (l+1)p'/r + (p'' + p'^2)/2` | correct (re-derived) |
| `p2` from the radial equation at `r_c` | correct |
| HSC identity `dD/deps = -(2/u(R)^2) * int_0^R u^2 dr` | correct (re-derived; numerics agree to 1.5e-7) |
| Norm-conserving matches the AE slope, non-conserving does not | confirmed: -3.3593 / -3.3593 / -4.1406, i.e. 23% off |
| Outward integration and the `i = n-1` index in `log_derivative` | correct |
| Repulsive-at-origin window, and `V_ps(0)` values | confirmed: repulsive for r_c in [1.1, 2.1]; +5.16 Ha at 1.6; -45 Ha at 0.9 |
| 4s error minimal below r_c = 1.6 and rising both ways | confirmed: 1.6 meV at r_c = 1.2, 5.2 meV at 0.9, 72 meV at 2.6 |
| `r_c` must sit outside the outermost 3s node (0.58 Bohr) | correct, and for the right reason |
| HGH Si local form factor, `Z_ion = 4`, `r_loc = 0.44`, `C1 = -7.336` | correct — matches the analytic FT of the HGH local form |
| `cb.local(q)` returns `v_hat` in Ha·Bohr³ | correct, so the comparison table's units are consistent |
| CB positive on shells 8 and 11, screened HGH negative on all three | confirmed (-14.18, +2.70, +5.40 vs -24.89, -15.74, -13.23) |
| CB silicon converges below 1 meV at 6 Ha | consistent with the 1b validation |

## 1. The "missing norm" criterion understates the cutoff by a factor of 2 to 7

This is the one finding that changes a number a reader will carry away. The whole cost
axis of the post — both interactive charts and all the quoted `E_cut` and `N_PW` figures
— uses *norm* not captured below the cutoff. But what sets a production cutoff is energy
convergence, and the kinetic-energy density in q-space carries an extra factor `q^2/2`,
so it converges later. Measured on the pseudo-3s, same orbital, same 1e-4 threshold:

| r_c (Bohr) | `E_cut` for 1e-4 missing **norm** | for 1e-4 missing **kinetic energy** | ratio |
|---|---|---|---|
| 0.9 | 47.4 Ha | 326.2 Ha | 6.9x |
| 1.2 | 13.9 | 38.2 | 2.7x |
| 1.6 | 11.5 | 26.9 | 2.3x |
| 2.1 | 8.0 | 10.9 | 1.4x |
| 2.4 | 5.8 | 9.1 | 1.6x |

The post tells the reader that at `r_c` = 1.6 Bohr the pseudo-3s "needs about 11 Ha".
A reader who knows that real norm-conserving silicon runs at 20-30 Ha will think the
post is wrong, and a reader who does not will come away with a number that is too small
by a factor of two or three. The 27 Ha from the kinetic criterion is the one that
matches practice.

Cheapest fix: add the kinetic-energy curve to the "missing norm vs cutoff" panels as a
second line from the same transform (`(2/pi) ũ² q² * q²/2` instead of `(2/pi) ũ² q²`,
three extra lines of code). That turns the issue into a teaching point — *which* quantity
you demand convergence in sets the cutoff, and the norm is the most forgiving choice.
Failing that, one sentence saying the norm criterion is a lower bound and energy
convergence costs 2-3x more.

The same correction applies to the headline all-electron numbers, which are quoted as
norm-based.

## 2. The pseudopotential the post builds is screened, and the post never says so

This is the biggest conceptual gap. The Kerker construction takes `V_ps = V_ae` outside
`r_c`, and the model `V_ae` tends to `-1/r` — a *neutral* atom. So the object built in
"Building a norm-conserving pseudopotential" is a **screened** pseudopotential: it
already contains the Hartree and exchange-correlation potential of the valence
electrons. It is not what goes into a solid-state code.

A real construction ends with a step the post never performs or names at this point:
unscreening, `V_ion = V_ps - V_H[n_val] - V_xc[n_val]`, which turns the `-1/r` tail into
`-Z_ion/r = -4/r`.

The omission bites two sections later. The form-factor chart plots "ionic tail, -4/r"
and the HGH curve approaching `-4*pi*Z_ion/q^2`, while the potential the reader just
built would approach `-4*pi*1/q^2` — a factor of four, unremarked. The "Bare and
screened potentials" bullet states the distinction in the abstract but never says that
the post's own `V_ps` is on the screened side of it.

One paragraph at the end of the construction section closes this. It is also a natural
place to say *why* unscreening is legitimate: the valence density of the pseudo-atom
equals the true valence density outside `r_c` — which is norm conservation again, in its
other role. That connects the two halves of the post and sets up the ultrasoft/PAW
bullet, where the whole point of the augmentation charge is to repair exactly this
property.

## 3. Norm conservation has a second job the post does not mention

The transferability section presents norm conservation purely through scattering: the
HSC identity and the energy derivative of `D_l`. That is correct and well told. But norm
conservation is equally about the **density**: because the pseudo-orbital carries the
same charge inside `r_c`, the valence density outside `r_c` is right, so the Hartree and
XC potentials the valence electrons generate in a solid are right. One sentence, and it
pays off three times — it justifies unscreening (finding 2), it explains why dropping
norm conservation forces ultrasoft to carry augmentation charges, and it tells the reader
why a condition stated as an integral over the *core* region controls physics *outside*
it.

## 4. Smaller scientific corrections

**4a. "a double radial integral".** The semilocal matrix element is
`<k+G|dV_l|k+G'> = (4*pi/Omega)(2l+1) P_l(cos theta) int r^2 dV_l(r) j_l(|k+G|r) j_l(|k+G'|r) dr`
— a **single** radial integral, but one for every pair of plane waves. The cost
conclusion is right; the mechanism as stated is not. Say "one radial integral for every
pair of plane waves, so an N x N table" — which also makes the contrast with
Kleinman-Bylander (one vector, not a table) sharper.

**4b. Ghost states.** Kleinman-Bylander is presented as a pure win. The standard caveat
belongs in a post that promises not to hide details: the separable form is *not*
equivalent to the semilocal one away from the reference state, and for a bad choice of
local channel it can produce spurious deeply bound "ghost" states that the semilocal
potential does not have. One sentence, and it is directly relevant to jaxdft, which will
have to choose a local channel.

**4c. PAW.** "Ultrasoft and PAW: drop norm conservation and restore the missing charge
with augmentation functions" is right for ultrasoft and misleading for PAW. PAW is a
linear transformation between smooth and all-electron wavefunctions that keeps the
all-electron information and is in principle exact; it is not a softer approximation that
gave something up. The shared consequence (a generalised eigenvalue problem, augmentation
bookkeeping) is correctly stated — just split the characterisation.

**4d. The failed Lindhard comparison has more than one cause.** The conclusion — that
comparing local parts across families is meaningless, because an empirical local
potential absorbs the nonlocal repulsion — is correct and worth making. But the section
reads as if nonlocality were the only problem. At the three shells, `q/2k_F` = 0.55, 0.90
and 1.06, so the comparison sits right on the `2k_F` structure of the Lindhard function;
and silicon is a gapped semiconductor whose `eps(q->0)` saturates at about 11.7 rather
than diverging as the free-electron form does. Linear response applied to an ionic
potential this strong is crude on its own terms. One clause keeps the reader from
concluding that the nonlocal part is the single missing ingredient.

**4e. The model atom's core is deeper than real silicon's.** The post puts the model's
levels next to real Si binding energies and says silicon "shows the same separation of
scales". The separation is indeed the same, but the numbers are not close: the model
gives 2161 / 248 / 230 eV for 1s / 2s / 2p against 1839 / 149.7 / 99.6 eV, so the 2p is
2.3x too deep. That is expected from a fixed rather than self-consistent screening, and
saying so in half a sentence stops a reader from trying to reconcile them.

## 5. Precision and code

**5a. A stray control byte breaks a formula.** At byte offset 34674 the file contains a
literal `0x07` (BEL) where `\a` of `\approx` should be: `It is smallest near $r_c
<BEL>pprox$`. It will render as garbage. Probably a Python escape that went through an
unescaped string on the way in — worth checking whether anything else in the repo was
written the same way.

**5b. The 1s cutoff is quoted to four significant digits but is grid-dependent.** On the
post's q grid I get 4,793 Ha (4.3 million plane waves); extending the grid gives 4,922 Ha
at `q_max` = 320, 4,954 at 640 and 5,074 at 1280, still climbing. The reason is that the
missing-norm measure has a floor of about 5e-5 from the radial grid and the `r <= 20`
truncation in `hankel`, only a factor of two below the 1e-4 threshold being tested. The
physics is unaffected — it is thousands of Hartree either way — but quote it as "about
5,000 Ha" and "a few million plane waves", or raise the threshold to 1e-3 for the 1s,
where the measurement is solid. The 3s number (746 Ha) is well converged.

**5c. `ecut_for` fails silently.** `np.argmax(1 - fraction_curve < missing)` returns 0
when the condition is never met, so the function returns `q[0]**2/2` — about 5e-9 Ha —
rather than raising. The 1s and 3s calls are protected by the assert that follows, but
the `r_c` sweep calls it 18 times with no guard, and a future edit that tightens the
threshold or shortens the q grid would produce a silently wrong curve. Two lines:
`ok = 1 - fraction_curve < missing; if not ok.any(): raise ValueError(...)`.

**5d. `V_ps` is continuous at `r_c` but not differentiable.** The construction matches
`u`, `u'` and `u''`, which makes `V` continuous — the post says this correctly. The chart
shows a visible kink at `r_c` that the text does not account for, and a reader may read it
as a numerical artefact. One clause ("the potential is continuous at `r_c` but its slope
is not; Troullier-Martins adds conditions that smooth it") both explains the plot and
motivates the next paragraph, which already mentions TM.

**5e. Condition ordering in the construction list.** "Conditions 3 and 4 give four
equations for the four coefficients" is true, but the fourth coefficient only exists
because condition 2 fixed the *form* of `p` (quartic, no linear term). As written the
reader counts three matching conditions plus one norm condition against four unknowns and
has to reconstruct where the quartic came from. Stating the parameter count in
condition 2 removes the hop.

## 6. Pedagogical assessment

The arc works: cost of the cusp, why the core can go, how to replace it, what the
replacement costs you, why it has to be an operator rather than a function. The three
devices that carry the most weight are the right ones — the reconstruction slider
(seeing the orbital fail to reassemble), the `r_c` slider (one knob, three consequences
at once), and the log-derivative chart (the abstract transferability claim made visible
in one picture). The decision to compute everything from a model atom rather than assert
it is what makes the post trustworthy, and the inline asserts are a good habit:
they make the prose falsifiable.

Three suggestions, in order of value:

1. The softness/transferability chart is the climax of the post and is currently two
   separate y-axes against `r_c`. Plotting transferability error directly against cutoff
   — a trade-off curve with `r_c` as the parameter along it — would show the Pareto
   front in one line, which is the thing practitioners actually navigate. The data is
   already computed.
2. There is no statement anywhere of what a reader should expect in practice: that
   silicon norm-conserving pseudopotentials run at roughly 20-30 Ha, carbon and oxygen at
   60-80, and 3d metals higher still. One table of real numbers would anchor every
   relative statement in the post, and it is also the natural place to land finding 1.
3. The post explains why the core can be frozen but never says what is lost. Core-level
   spectroscopy, hyperfine and EFG properties, and anything where core states respond
   (high pressure, semicore states in early transition metals) are outside what a
   pseudopotential can do. Two sentences, and it tells the reader where the concept's
   boundary is — which a concept post should.

## 7. Items for the author to decide

- Whether to add the kinetic-energy curve (finding 1) or just a caveat sentence. The
  chart version is better but costs a legend entry in an already busy panel.
- Whether the unscreening paragraph (finding 2) belongs at the end of the construction
  section or as its own short section before "Why pseudopotentials are nonlocal". I would
  put it in the construction section, because that is where the reader forms their mental
  model of what the object is.
