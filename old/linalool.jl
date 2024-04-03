michaelismenten(x, kcat, km) = (kcat*x)/(km+x)

function lnl_params(param_name)
    params = Dict("kcat_thmgr_hmgcoa" => 0.0035*53.01 * 60, #[1/min -> 1/hr]
    "km_thmgr_hmgcoa" => 50 / 1000, #[uM -> mM]
    "kcat_mevk_mev" => 0.0035*53.01 * 60, #FILL IN
    "km_mevk_mev" => 50 / 1000, #FILL IN
    "kcat_idi_dmapp" => 456.75 * 60, #[1/min -> 1/hr]
    "km_idi_dmapp" => 43 / 1000, #[uM -> mM]
    "kcat_idi_ipp" => 20*33.352 * 60, #[1/min -> 1/hr]
    "km_idi_ipp" => 43 / 1000, #[uM -> mM]
    "kcat_erg20_gpp" => 0.021*60/10 * 60, #[1/min -> 1/hr]
    "km_erg20_gpp" => 27.56 / 1000, #[uM -> mM]
    "kcat_erg20_dmapp" => 0.012*60/10 * 60, #[1/min -> 1/hr]
    "km_erg20_dmapp" => 0.49 / 1000, #[uM -> mM]
    "kcat_lis_gpp" => 14.4 * 60, #[1/min -> 1/hr]
    "km_lis_gpp" => 25 / 1000, #[uM -> mM]
    "alpha" => (0.0028 * 60)/1000, #[uM/min-RPU -> mM/hr-RPU]
    "lam" => 0.00577 * 60 #[1/min -> 1/hr]
    )
    return params[param_name]
end

function linalool(du, u, p, t)
    lam, v_in_hmgcoa, v_ipp, v_fpp, W = p
    hmgcoa, mev, ipp, dmapp, gpp, fpp, linalool, thmgr, idi, erg20, lis = u
    k_a, k_b, k_c, k_d = W #[RPU]

    alpha = lnl_params("alpha")
    mevk = 0 #cellular concentration of mevalonate kinase, determined to be in excess, FILL IN

    v_thmgr = thmgr * michaelismenten(hmgcoa, lnl_params("kcat_thmgr_hmgcoa"), lnl_params("km_thmgr_hmgcoa"))
    v_mevk = mevk * michaelismenten(mev, lnl_params("kcat_mevk_mev"), lnl_params("km_mevk_mev"))
    v_idi_ipp = idi * michaelismenten(ipp, lnl_params("kcat_idi_ipp"), lnl_params("km_idi_ipp"))
    v_idi_dmapp = idi * michaelismenten(dmapp, lnl_params("kcat_idi_dmapp"), lnl_params("km_idi_dmapp"))
    v_erg20_dmapp = erg20 * michaelismenten(dmapp, lnl_params("kcat_erg20_dmapp"), lnl_params("km_erg20_dmapp"))
    v_erg20_gpp = erg20 * michaelismenten(gpp, lnl_params("kcat_erg20_gpp"), lnl_params("km_erg20_gpp"))
    v_lis_gpp = lis * michaelismenten(gpp, lnl_params("kcat_lis_gpp"), lnl_params("km_lis_gpp"))

    du[1] = v_in_hmgcoa - lam*hmgcoa #hmgcoa #changed from lit model where assumed constant from native metabolism, now from FBA
    du[2] =  v_thmgr - v_mevk - lam*mev #mev
    du[3] = v_mevk + v_idi_dmapp - v_idi_ipp - v_erg20_dmapp - v_erg20_gpp - lam*ipp + v_ipp #ipp
    du[4] = v_idi_ipp - v_idi_dmapp - v_erg20_dmapp - lam*dmapp #dmapp
    du[5] = v_erg20_dmapp - v_erg20_gpp - v_lis_gpp - lam*gpp #gpp
    du[6] = v_erg20_gpp - lam*fpp + v_fpp #fpp
    du[7] = v_lis_gpp - lam*linalool #linalool
    du[8] = alpha*k_a - lam*thmgr #thmgr
    du[9] = alpha*k_b - lam*idi #idi
    du[10] = alpha*k_c - lam*erg20 #erg20
    du[11] = alpha*k_d - lam*lis #lis
end