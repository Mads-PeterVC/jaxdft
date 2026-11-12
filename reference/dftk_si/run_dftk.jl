# DFTK reference data for Milestone 2b (theory post 02b, 'Against references',
# 'Test size' and 'Tolerances'): diamond silicon, LDA (Slater + PW92), HGH with
# nonlocal projectors, Gamma-centred k-grids, explicit FFT grid, no symmetries,
# zero temperature.
#
# Run from the repository root:
#     ~/.juliaup/bin/julia --project=reference/dftk_si reference/dftk_si/run_dftk.jl
# Writes reference/dftk_si/dftk_si.json.

using DFTK
using JSON
using Libxc
using LinearAlgebra
using SHA

const HERE = @__DIR__
const PSP_PATH = normpath(joinpath(HERE, "..", "..", "src", "jaxdft", "pseudo", "data",
                                  "hgh_lda", "si-q4.hgh"))
const A_SI = 10.263              # Bohr, the acceptance size (tests/support/silicon.py)
const ECUT = 12.0                # Ha
const FFT_SIZE = (24, 24, 24)    # jaxdft's rule_fft_shape at this size
const N_BANDS = 5                # 4 occupied + the lowest conduction band
const SCF_TOL = 1e-11            # L2 norm of the density change

# fcc primitive vectors a/2 (0,1,1), a/2 (1,0,1), a/2 (1,1,0) (jaxdft.fcc_lattice);
# DFTK takes them as columns.
const LATTICE = A_SI / 2 * [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]

# jaxdft.diamond: origin at the bond centre.
const IDEAL = [[0.125, 0.125, 0.125], [-0.125, -0.125, -0.125]]
# tests/support/silicon.py::DISPLACEMENT, atom 0 moved (fractional).
const DISPLACED = [IDEAL[1] + [0.012, -0.005, 0.008], IDEAL[2]]

function run(name, positions, kgrid)
    Si = ElementPsp(:Si, PspHgh(PSP_PATH))  # load_psp only dispatches on .gth
    model = model_DFT(LATTICE, [Si, Si], positions; functionals=LDA(), symmetries=false)
    @assert model.temperature == 0
    grid = MonkhorstPack(kgrid; kshift=[0, 0, 0])  # Gamma-centred
    basis = PlaneWaveBasis(model; Ecut=ECUT, kgrid=grid, fft_size=FFT_SIZE)
    @assert basis.fft_size == FFT_SIZE
    scfres = self_consistent_field(basis; tol=SCF_TOL,
                                   nbandsalg=FixedBands(; n_bands_converge=N_BANDS),
                                   callback=identity)
    @assert scfres.converged
    forces = compute_forces_cart(scfres)
    energies = Dict(string(k) => v for (k, v) in pairs(scfres.energies))
    Dict(
        "name" => name,
        "lattice_bohr" => [LATTICE[:, i] for i in 1:3],  # rows = lattice vectors
        "positions_fractional" => [collect(p) for p in positions],
        "ecut_hartree" => ECUT,
        "fft_size" => collect(basis.fft_size),
        "kgrid" => collect(kgrid),
        "n_symmetries" => length(basis.symmetries),
        "kpoints_fractional" => [collect(k.coordinate) for k in basis.kpoints],
        "kweights" => collect(basis.kweights),
        "n_plane_waves" => [length(G_vectors(basis, k)) for k in basis.kpoints],
        "energies_hartree" => energies,
        "total_energy_hartree" => scfres.energies.total,
        "eigenvalues_hartree" => [e[1:N_BANDS] for e in scfres.eigenvalues],
        "occupations" => [o[1:N_BANDS] for o in scfres.occupation],
        "fermi_level_hartree" => scfres.εF,
        "forces_hartree_per_bohr" => [collect(f) for f in forces],
        "scf_iterations" => scfres.n_iter,
        "scf_tol" => SCF_TOL,
    )
end

# The ideal geometry on the test grid; the displaced one on 3^3, where the gap over
# the grid stays open (on 2^3 it closes to 0.12 mHa; engineering log 2026-11-11).
results = [run("ideal", IDEAL, (2, 2, 2)), run("displaced", DISPLACED, (3, 3, 3))]
out = Dict(
    "description" => "DFTK reference for jaxdft Milestone 2b: Si, LDA, HGH, k-grids",
    "generator" => "reference/dftk_si/run_dftk.jl",
    "versions" => Dict(
        "julia" => string(VERSION),
        "DFTK" => string(pkgversion(DFTK)),
        "libxc" => string(Libxc.libxc_version),
    ),
    "functional" => ["lda_x", "lda_c_pw"],
    "pseudopotential" => Dict(
        "source" => "PseudoLibrary hgh_lda_hgh/si-q4.hgh (vendored in jaxdft)",
        "sha256" => bytes2hex(open(sha256, PSP_PATH)),
    ),
    "units" => "Hartree atomic units",
    "systems" => results,
)
open(joinpath(HERE, "dftk_si.json"), "w") do io
    JSON.json(io, out; pretty=true)
end
println("wrote ", joinpath(HERE, "dftk_si.json"))
