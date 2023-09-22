repression(x, k, theta, n) = k/(1+(x/theta)^n)
activation(x, k, theta, n) = (k*(x^n))/(theta^n + x^n)
reversible_michaelismenten(x, y, vm, keq, kmx, kmy) = (vm*(x - (y/keq)))/(x + kmx*(1+(y/kmy)))
hillequation(x, vm, n, km) = vm*x^n / (km^n + x^n)
michaelismenten(x, vm, km) = (vm*x)/(km+x)
michaelismenten_substrateactivation(x, vm, km, a, ka) = ((vm * (1+ (a*x)/(ka + x)))*x)/(km + x)

function ga_params(param_name)
    params = Dict("vm_ino1" => 0.2616 * 60 * 60, #[mM/s],
    "km_ino1_g6p" => 1.18, #[mM],
    "vm_t_mi"=> 0.045  * 60 * 60, #[mM/s],
    "km_t_mi"=> 15, #[mM],
    "vm_miox" => 0.2201 * 60 * 60, #[mM/s],
    "km_miox_mi" => 24.7, #[mM],
    "a_miox" => 5.4222, #[no units],
    "ka_miox_mi" => 20, #[mM],
    "vm_glc" => 0.1 * 60 * 60,#EC 2.7.1.63 [mM/s],
    "km_glc" => 0.082, #[mM]
    "vm_pgi" => 1.1094 * 60 * 60,#[mM/s],
    "keq_pgi" => 0.3, #[no units],
    "km_pgi_g6p" => 0.28, #[mM],
    "km_pgi_f6p" => 0.147, #[mM],
    "vm_pfk" => 2.615 * 60 * 60,#[mM/s],
    "km_pfk_f6p" => 0.16, #[mM],
    "n_pfk" => 3, #[no units],
    "vm_zwf" => 0.0853 * 60 * 60,
    "km_zwf_g6p" => 0.1,
    "v_pts" => 0.1656 * 60 * 60,
    "kcat_udh" => 161 * 60 * 60, #[1/s], from Brenda
    "km_udh" => 0.2 #[mM], from Brenda
    )
    return params[param_name]
end

function glucaric_acid(du, u, p, t)
    g6p, f6p, mi, ino1, miox = u

    A, W, v_in, lam = p
    
    k_ino1, theta_ino1, k_miox, theta_miox = W
    
    k_ino1 = k_ino1 * 60 * 60
    k_miox = k_miox * 60 * 60

    v_pgi = reversible_michaelismenten(g6p, f6p, ga_params("vm_pgi"), ga_params("keq_pgi"), ga_params("km_pgi_g6p"), ga_params("km_pgi_f6p"))
    v_zwf = michaelismenten(g6p, ga_params("vm_zwf"), ga_params("km_zwf_g6p"))
    v_pfk = hillequation(f6p, ga_params("vm_pfk"), ga_params("n_pfk"), ga_params("km_pfk_f6p"))
    v_ino1 = ino1 * michaelismenten(g6p, ga_params("vm_ino1"), ga_params("km_ino1_g6p"))
    v_tm = michaelismenten(mi, ga_params("vm_t_mi"), ga_params("km_t_mi"))
    v_miox = miox * michaelismenten_substrateactivation(mi, ga_params("vm_miox"), ga_params("km_miox_mi"), ga_params("a_miox"), ga_params("ka_miox_mi"))

    u_ino1_mi = sum(A[1] .* [activation(mi, k_ino1, theta_ino1, 2), repression(mi, k_ino1, theta_ino1, 2), k_ino1])
    u_miox_mi = sum(A[2] .* [activation(mi, k_miox, theta_miox, 2), repression(mi, k_miox, theta_miox, 2), k_miox])

    du[1] = v_in - v_zwf - v_pgi - v_ino1 - lam*g6p
    du[2] = v_pgi + 0.5*v_zwf - v_pfk - lam*f6p
    du[3] = v_ino1 - v_tm - v_miox - lam*mi
    du[4] = u_ino1_mi  - lam*ino1
    du[5] = u_miox_mi - lam*miox
end