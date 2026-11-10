# DFTK reference data for Milestone 2a (theory post 02a, 'Against references' and
# decision 5): H2 in a cubic box, LDA (Slater + PW92), HGH local pseudopotential,
# Gamma only, explicit FFT grid, no symmetries.
#
# Run from the repository root:
#     ~/.juliaup/bin/julia --project=reference/dftk_h2 reference/dftk_h2/run_dftk.jl
# Writes reference/dftk_h2/dftk_h2.json.

using DFTK
using JSON
using Libxc
using LinearAlgebra
using SHA

const HERE = @__DIR__
const PSP_PATH = normpath(joinpath(HERE, "..", "..", "src", "jaxdft", "pseudo", "data",
                                  "hgh_lda", "h-q1.hgh"))
const L = 10.0       # Bohr, the acceptance size
const ECUT = 10.0    # Ha
const FFT_SIZE = (30, 30, 30)  # jaxdft's rule_fft_shape at this size
const SCF_TOL = 1e-11          # L2 norm of the density change

"Symmetric H2 along z, bond centre at the cell centre (jaxdft.dimer_in_box)."
function symmetric(d=1.45)
    c = [L / 2, L / 2, L / 2]
    u = [0.0, 0.0, 1.0]
    [c - d / 2 * u, c + d / 2 * u]
end

"Bond tilted off every axis, centre off every grid point (tests/support/h2.py)."
function distorted(d=1.2)
    u = [0.2, 0.1, 1.0] / norm([0.2, 0.1, 1.0])
    c = fill(L / 2, 3) + [0.13, -0.07, 0.21]
    [c - d / 2 * u, c + d / 2 * u]
end

function run(name, cartesian)
    lattice = L * Matrix(1.0I, 3, 3)  # columns are lattice vectors (DFTK convention)
    positions = [lattice \ r for r in cartesian]
    H = ElementPsp(:H, PspHgh(PSP_PATH))  # load_psp only dispatches on .gth
    model = model_DFT(lattice, [H, H], positions; functionals=LDA(), symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=ECUT, kgrid=(1, 1, 1), fft_size=FFT_SIZE)
    scfres = self_consistent_field(basis; tol=SCF_TOL,
                                   nbandsalg=FixedBands(; n_bands_converge=2),
                                   callback=identity)
    @assert scfres.converged
    forces = compute_forces_cart(scfres)
    energies = Dict(string(k) => v for (k, v) in pairs(scfres.energies))
    Dict(
        "name" => name,
        "lattice_bohr" => [lattice[:, i] for i in 1:3],  # rows = lattice vectors
        "positions_fractional" => [collect(p) for p in positions],
        "ecut_hartree" => ECUT,
        "fft_size" => collect(basis.fft_size),
        "n_plane_waves" => length(G_vectors(basis, basis.kpoints[1])),
        "energies_hartree" => energies,
        "total_energy_hartree" => scfres.energies.total,
        "eigenvalues_hartree" => scfres.eigenvalues[1][1:2],
        "occupations" => scfres.occupation[1][1:2],
        "forces_hartree_per_bohr" => [collect(f) for f in forces[1:2]],
        "scf_iterations" => scfres.n_iter,
        "scf_tol" => SCF_TOL,
    )
end

results = [run("symmetric", symmetric()), run("distorted", distorted())]
out = Dict(
    "description" => "DFTK reference for jaxdft Milestone 2a: H2, LDA, HGH, Gamma",
    "generator" => "reference/dftk_h2/run_dftk.jl",
    "versions" => Dict(
        "julia" => string(VERSION),
        "DFTK" => string(pkgversion(DFTK)),
        "libxc" => string(Libxc.libxc_version),
    ),
    "functional" => ["lda_x", "lda_c_pw"],
    "pseudopotential" => Dict(
        "source" => "PseudoLibrary hgh_lda_hgh/h-q1.hgh (vendored in jaxdft)",
        "sha256" => bytes2hex(open(sha256, PSP_PATH)),
    ),
    "units" => "Hartree atomic units",
    "systems" => results,
)
open(joinpath(HERE, "dftk_h2.json"), "w") do io
    JSON.json(io, out; pretty=true)
end
println("wrote ", joinpath(HERE, "dftk_h2.json"))
