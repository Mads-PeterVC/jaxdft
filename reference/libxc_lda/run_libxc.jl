# libxc reference for jaxdft's LDA (theory post 02a, 'Exchange and correlation' and
# decision 11): lda_x and lda_c_pw, spin-unpolarised, energy per particle (zk) and
# potential (vrho) on a logarithmic density grid.
#
# Run from the repository root (uses the pinned project of the DFTK reference):
#     ~/.juliaup/bin/julia --project=reference/dftk_h2 reference/libxc_lda/run_libxc.jl
# Writes reference/libxc_lda/libxc_lda.json.

using JSON
using Libxc

# 1e-10 to 1e3 Bohr^-3: r_s from about 1300 down to 0.06, covering the H2 tail and
# far denser than any valence density.
rho = 10.0 .^ range(-10, 3; length=131)

functionals = Dict()
# lda_c_pw_mod is stored as a negative control: the comparison must tell it apart.
for name in (:lda_x, :lda_c_pw, :lda_c_pw_mod)
    f = Functional(name; n_spin=1)
    r = evaluate(f; rho=reshape(rho, 1, :), derivatives=0:1)
    functionals[string(name)] = Dict(
        "zk" => vec(r.zk),        # energy per particle (Ha)
        "vrho" => vec(r.vrho),    # d(rho * zk)/d rho (Ha)
        "density_threshold" => f.density_threshold,
    )
end

out = Dict(
    "description" => "libxc LDA reference for jaxdft Milestone 2a",
    "generator" => "reference/libxc_lda/run_libxc.jl",
    "versions" => Dict("julia" => string(VERSION), "libxc" => string(Libxc.libxc_version),
                       "Libxc.jl" => string(pkgversion(Libxc))),
    "units" => "Hartree atomic units; rho in Bohr^-3",
    "rho" => rho,
    "functionals" => functionals,
)
open(joinpath(@__DIR__, "libxc_lda.json"), "w") do io
    JSON.json(io, out; pretty=true)
end
println("wrote ", joinpath(@__DIR__, "libxc_lda.json"))
