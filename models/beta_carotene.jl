michaelismenten(x, kcat, km) = (kcat*x)/(km+x)
michaelismenten_dual(x, y, kcat, km1, km2) = (kcat*((x*y)/(km1*km2)))/(1+(x/km1)+(y/km2))

function bc_params(param_name)
    params = Dict("kcat_crtE" => 0.2456 * 60 * 60, #[1/s] DL,
    "kcat_crtB" => 0.066 * 60 * 60, #[1/s -> 1/hr] DL
    "kcat_crtI" => 4.2255 * 60 * 60, #[1/s] DL
    "kcat_crtY" => 42.9099 * 60 * 60, #[1/s] DL
    "km_crtE_fpp" => 0.0321, #[mM] Brenda
    "km_crtE_ipp" => 0.0234, #[mM] Brenda
    "km_crtB" => 0.01682, #[mM] Brenda
    "km_crtI" => 9.179, #[mM] Brenda
    "km_crtY" => 0.035, #[mM] Brenda
    )
    return params[param_name]
end

function beta_carotene(du, u, p, t)
    lam, v_in_fpp, v_in_ipp, W = p
    fpp, ipp, ggp, phy, lyc, bcar, crtE, crtB, crtI, crtY = u
    k_crtE, k_crtB, k_crtI, k_crtY = W

    v_crtE = crtE * michaelismenten_dual(fpp, ipp, bc_params("kcat_crtE"), bc_params("km_crtE_fpp"), bc_params("km_crtE_ipp"))
    v_crtB = crtB * michaelismenten(ggp, bc_params("kcat_crtB"), bc_params("km_crtB"))
    v_crtI = crtI * michaelismenten(phy, bc_params("kcat_crtI"), bc_params("km_crtI"))
    v_crtY = crtY * michaelismenten(lyc, bc_params("kcat_crtY"), bc_params("km_crtY"))

    du[1] = v_in_fpp - v_crtE - lam*fpp
    du[2] = v_in_ipp - v_crtE - lam*ipp
    du[3] = v_crtE - v_crtB - lam*ggp
    du[4] = v_crtB - v_crtI - lam*phy
    du[5] = v_crtI - v_crtY - lam*lyc
    du[6] = v_crtY - lam*bcar
    du[7] = k_crtE - lam*crtE
    du[8] = k_crtB - lam*crtB
    du[9] = k_crtI - lam*crtI
    du[10] = k_crtY - lam*crtY
end

function native_metabolism(du, u, p, t)
    lam, v_in_fpp, v_in_ipp = p
    fpp, ipp= u

    du[1] = v_in_fpp - lam*fpp
    du[2] = v_in_ipp - lam*ipp
end