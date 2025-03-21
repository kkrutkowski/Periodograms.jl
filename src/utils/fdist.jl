#----------------------------------------------------------------------
# 1. Digamma (ψ) function
function psi(x::T) where T <: AbstractFloat
    # Coefficients (note: indices in Julia start at 1)
    p1 = T[0.00895385022981970, 4.77762828042627, 142.441585084029,
           1186.45200713425, 3633.51846806499, 4138.10161269013, 1305.60269827897]
    q1 = T[44.8452573429826, 520.752771467162, 2210.00799247830,
           3641.27349079381, 1908.31076596300, 0.00000691091682714533]
    p2 = T[-2.12940445131011, -7.01677227766759, -4.48616543918019, -0.648157123766197]
    q2 = T[32.2703493791143, 89.2920700481861, 54.6117738103215, 7.77788548522962]

    dx0   = T(1.461632144968362341262659542325721325)
    xmax1 = T(4503599627370496.0)
    xsmall = T(1e-9)
    aug = zero(T)

    # For x < 0.5 use reflection formula
    if x < T(0.5)
        if abs(x) <= xsmall
            return x == zero(T) ? zero(T) : -one(T)/x
        else
            w = -x
            sgn = T(pi)/T(4)
            if w <= zero(T)
                w = -w
                sgn = -sgn
            end
            if w >= xmax1
                return zero(T)
            end
            w -= floor(w)
            nq = Int(floor(w * T(4)))
            w = T(4)*(w - T(0.25)*nq)
            if isodd(nq)
                w = one(T) - w
            end
            z = (T(pi)/T(4)) * w
            if isodd(div(nq,2))
                sgn = -sgn
            end
            if isodd(div(nq+1,2))
                aug = sgn * (tan(z)*T(4))
            else
                if z == zero(T)
                    return zero(T)
                end
                aug = sgn * (T(4)/tan(z))
            end
        end
        x = one(T) - x
    end

    if x <= T(3)
        den = x
        upper = p1[1]*x
        for i in 1:5
            den   = (den + q1[i]) * x
            upper = (upper + p1[i+1]) * x
        end
        den = (upper + p1[7])/(den + q1[6])
        return den*(x - dx0) + aug
    else
        if x < xmax1
            w = one(T)/(x*x)
            den = w
            upper = p2[1]*w
            for i in 1:3
                den   = (den + q2[i]) * w
                upper = (upper + p2[i+1]) * w
            end
            aug += upper/(den + q2[4]) - T(0.5)/x
        end
        return aug + log(x)
    end
end

#----------------------------------------------------------------------
# 2. ln(1+a) function (alnrel)
function alnrel(a::T) where T <: AbstractFloat
    if abs(a) > T(0.375)
        return log(1 + a)
    else
        t = a/(a + T(2))
        t2 = t*t
        # Coefficients (from C, re-indexed)
        p = T[-1.29418923021993, 0.405303492862024, -0.0178874546012214]
        q = T[-1.62752256355323, 0.747811014037616, -0.0845104217945565]
        w = (((p[3]*t2 + p[2])*t2 + p[1])*t2 + one(T)) /
            (((q[3]*t2 + q[2])*t2 + q[1])*t2 + one(T))
        return T(2)*t*w
    end
end

#----------------------------------------------------------------------
# 3. ln(Γ(1+a)) for -0.2 ≤ a ≤ 1.25
function gamln1(a::T) where T <: AbstractFloat
    # Coefficients (C arrays re-indexed to Julia’s 1-indexing)
    p = T[0.577215664901533, 0.844203922187225, -0.168860593646662,
          -0.780427615533591, -0.402055799310489, -0.0673562214325671,
          -0.00271935708322958]
    q = T[2.88743195473681, 3.12755088914843, 15.6875193295039,
          3.61951990101499, 0.0325038868253937, 0.00667465618796164]
    r = T[0.422784335098467, 0.848044614534529, 0.565221050691933,
          0.156513060486551, 0.0170502484022650, 0.000497958207639485]
    s = T[1.24313399877507, 0.548042109832463, 0.10155218743983,
          0.0713309612391, 0.00116165475989616]

    if a < T(0.6)
        top = ((((p[7]*a + p[6])*a + p[5])*a + p[4])*a + p[3])*a + p[2]
        top = top*a + p[1]
        bot = ((((q[6]*a + q[5])*a + q[4])*a + q[3])*a + q[2])*a + q[1]
        bot = bot*a + one(T)
        return -a*(top/bot)
    else
        x = a - one(T)
        top = ((((r[6]*x + r[5])*x + r[4])*x + r[3])*x + r[2])*x + r[1]
        bot = ((((s[5]*x + s[4])*x + s[3])*x + s[2])*x + s[1])*x + one(T)
        return x*(top/bot)
    end
end

#----------------------------------------------------------------------
# 4. ln(Γ(a)) for a > 0
function gamln(a::T) where T <: AbstractFloat
    d = T(0.418938533204673)
    c = T[0.0833333333333333, -0.00277777777760991, 0.00079365066682539,
          -0.00059520293135187, 0.000837308034031215, -0.00165322962780713]
    if a <= T(0.8)
        return gamln1(a) - log(a)
    elseif a <= T(2.25)
        return gamln1(a - one(T))
    elseif a < T(10)
        n = Int(floor(a - T(1.25)))
        t = a
        w = one(T)
        for _ in 1:n
            t -= one(T)
            w *= t
        end
        return gamln1(t - one(T)) + log(w)
    else
        t = (one(T)/a)^2
        w = ((((c[6]*t + c[5])*t + c[4])*t + c[3])*t + c[2])*t + c[1]
        w /= a
        return (d + w) + (a - T(0.5))*(log(a) - one(T))
    end
end

#----------------------------------------------------------------------
# 5. Compute ln(Γ(b)/Γ(a+b)) when b ≥ 8
function algdiv(a::T, b::T) where T <: AbstractFloat
    # Use the coefficients (shifted from C)
    carr = T[0.083333333333333, -0.00277777777760991, 0.00079365066682539,
             -0.00059520293135187, 0.000837308034031215, -0.00165322962780713]
    if a > b
        h = b/a
        c_val = one(T)/(one(T)+h)
        x = h/(one(T)+h)
        d_val = a + (b - T(0.5))
    else
        h = a/b
        c_val = h/(one(T)+h)
        x = one(T)/(one(T)+h)
        d_val = b + (a - T(0.5))
    end
    t_val = (one(T)/b)^2
    s3  = one(T) + (x + x^2)
    s5  = one(T) + (x + x^2*s3)
    s7  = one(T) + (x + x^2*s5)
    s9  = one(T) + (x + x^2*s7)
    s11 = one(T) + (x + x^2*s9)
    # Compute w as in C:
    temp = (carr[6]*s11)
    temp = temp*t_val + carr[5]*s9
    temp = temp*t_val + carr[4]*s7
    temp = temp*t_val + carr[3]*s5
    temp = temp*t_val + carr[2]*s3
    w = (temp*t_val + carr[1]) * c_val / b
    u = d_val * alnrel(a/b)
    v = a * (log(b) - one(T))
    return u > v ? ((w - v) - u) : ((w - u) - v)
end

#----------------------------------------------------------------------
# 6. Compute the correction: δ(a₀) + δ(b₀) − δ(a₀+b₀)
function bcorr(a0::T, b0::T) where T <: AbstractFloat
    carr = T[0.083333333333333, -0.00277777777760991, 0.00079365066682539,
             -0.00059520293135187, 0.000837308034031215, -0.00165322962780713]
    a = min(a0, b0)
    b = max(a0, b0)
    h = a/b
    c_val = h/(one(T)+h)
    x = one(T)/(one(T)+h)
    x2 = x*x
    s3  = one(T) + (x + x2)
    s5  = one(T) + (x + x2*s3)
    s7  = one(T) + (x + x2*s5)
    s9  = one(T) + (x + x2*s7)
    s11 = one(T) + (x + x2*s9)
    t_val = (one(T)/b)^2
    temp = (carr[6]*s11)
    temp = temp*t_val + carr[5]*s9
    temp = temp*t_val + carr[4]*s7
    temp = temp*t_val + carr[3]*s5
    temp = temp*t_val + carr[2]*s3
    w = (temp*t_val + carr[1]) * c_val / b
    t_a = (one(T)/a)^2
    temp_corr = (carr[6]*t_a)
    temp_corr = temp_corr*t_a + carr[5]
    temp_corr = temp_corr*t_a + carr[4]
    temp_corr = temp_corr*t_a + carr[3]
    temp_corr = temp_corr*t_a + carr[2]
    temp_corr = temp_corr*t_a + carr[1]
    corr = temp_corr / a
    return corr + w
end

#----------------------------------------------------------------------
# 7. Compute ln(Γ(a+b)) for 1 ≤ a ≤ 2 and 1 ≤ b ≤ 2
function gsumln(a::T, b::T) where T <: AbstractFloat
    x = a + b - T(2)
    if x <= T(0.25)
        return gamln1(one(T) + x)
    elseif x <= T(1.25)
        return gamln1(x) + alnrel(x)
    else
        return gamln1(x - one(T)) + log(x*(one(T)+x))
    end
end

#----------------------------------------------------------------------
# 8. ln(Beta(a, b)) = ln(Γ(a)Γ(b)/Γ(a+b))
function betaln(a0::T, b0::T) where T <: AbstractFloat
    e = T(0.918938533204673)
    a = min(a0, b0)
    b = max(a0, b0)
    if a >= T(8)
        w = bcorr(a, b)
        h = a/b
        c_val = h/(one(T)+h)
        u = -(a - T(0.5))*log(c_val)
        v = b * alnrel(h)
        return u > v ? ((-T(0.5)*log(b) + e + w) - v - u) : ((-T(0.5)*log(b) + e + w) - u - v)
    end
    if a < one(T)
        return b > T(8) ? gamln(a) + algdiv(a,b)  : gamln(a) + (gamln(b) - gamln(a+b))
    end
    # Make local copies so that later modifications do not affect inputs.
    aa = a
    bb = b
    w_val = zero(T)
    if aa <= T(2)
        if bb <= T(2)
            return gamln(aa) + gamln(bb) - gsumln(aa, bb)
        end
        if bb >= T(8)
            return gamln(aa) + algdiv(aa, bb)
        end
        w_val = zero(T)
    elseif aa > T(2)
        if bb <= T(1000)
            n = Int(floor(aa - one(T)))
            prod = one(T)
            for _ in 1:n
                aa -= one(T)
                h = aa / bb
                prod *= h/(one(T)+h)
            end
            w_val = log(prod)
            if bb >= T(8)
                return w_val + gamln(aa) + algdiv(aa, bb)
            end
        else
            n = Int(floor(aa - one(T)))
            prod = one(T)
            for _ in 1:n
                aa -= one(T)
                prod *= aa/(one(T) + (aa/bb))
            end
            return (log(prod) - n*log(bb)) + (gamln(aa) + algdiv(aa, bb))
        end
    end
    n = Int(floor(bb - one(T)))
    prod = one(T)
    for _ in 1:n
        bb -= one(T)
        prod *= bb/(aa+bb)
    end
    return w_val + log(prod) + (gamln(aa) + gamln(bb) - gsumln(aa, bb))
end

#----------------------------------------------------------------------
# 9. Complex logarithm and log(sin(z))
function sf_complex_log(zr::T, zi::T) where T <: AbstractFloat
    z = Complex(zr, zi)
    logz = log(z)
    return (real(logz), imag(logz))
end

function sf_complex_logsin(zr::T, zi::T) where T <: AbstractFloat
    z = Complex(zr, zi)
    return (real(log(sin(z))), imag(log(sin(z))))
end

#----------------------------------------------------------------------
# 10. Angle restriction (symmetric)
function gsl_sf_angle_restrict_symm(theta::T) where T <: AbstractFloat
    P1 = T(4) * 0.78539812564849853515625
    P2 = T(4) * 3.7748947079307981766760e-08
    P3 = T(4) * 2.6951514290790594840552e-15
    TwoPi = T(2)*(P1 + P2 + P3)
    y = sign(theta)*T(2)*floor(abs(theta)/TwoPi)
    r = ((theta - y*P1) - y*P2) - y*P3
    if r > π
        r -= 2*(P1+P2+P3)
    elseif r < -π
        r += 2*(P1+P2+P3)
    end
    if abs(theta) > T(0.0625)/eps(T)
        return NaN
    end
    return r
end

#----------------------------------------------------------------------
# 11. Complex ln(Γ(z)) via the Lanczos method (γ=7, kmax=8)
function lngamma_lanczos_complex(zr::T, zi::T) where T <: AbstractFloat
    lanczos_7_c = [T(0.99999999999980993227684700473478),
                   T(676.520368121885098567009190444019),
                   T(-1259.13921672240287047156078755283),
                   T(771.3234287776530788486528258894),
                   T(-176.61502916214059906584551354),
                   T(12.507343278686904814458936853),
                   T(-0.13857109526572011689554707),
                   T(9.984369578019570859563e-6),
                   T(1.50563273514931155834e-7)]
    # Lanczos is usually written for Γ(z+1)
    zr_adj = zr - one(T)
    # Compute Ag = lanczos_7_c[1] + Σ[k=1:8] lanczos_7_c[k+1]/( (zr_adj+k) - i*zi )
    Ag = Complex(lanczos_7_c[1], zero(T))
    for k in 1:8
        R = zr_adj + T(k)
        denom = R^2 + zi^2
        Ag += Complex(lanczos_7_c[k+1]*R/denom, -lanczos_7_c[k+1]*zi/denom)
    end
    log1 = log(Complex(zr_adj + T(7.5), zi))
    logAg = log(Ag)
    yr = (zr_adj + T(0.5))*real(log1) - zi*imag(log1) - (zr_adj + T(7.5)) +
         log(sqrt(2*pi)) + real(logAg)
    yi = zi*real(log1) + (zr_adj + T(0.5))*imag(log1) - zi + imag(logAg)
    return (yr, yi)
end

#----------------------------------------------------------------------
# 12. Hypergeometric 2F1 by Gauss series
function hyperg_2F1_series(a::T, b::T, c::T, x::T) where {T <: AbstractFloat}
    if abs(c) < eps(T)
        return zero(T)  # safeguard: c too close to 0
    end
    sum_pos = one(T)
    sum_neg = zero(T)
    del_pos = one(T)
    del_neg = zero(T)
    del = one(T)
    k = zero(T)
    i = 0
    while true
        i += 1
        if i > 30000
            return sum_pos - sum_neg
        end
        del_prev = del
        del *= (a + k) * (b + k) * x / ((c + k) * (k + one(T)))
        if del > zero(T)
            del_pos = del
            sum_pos += del
        elseif del == zero(T)
            # Exact termination (e.g. a or b negative integer)
            del_pos = zero(T)
            del_neg = zero(T)
            break
        else
            del_neg = -del
            sum_neg -= del
        end
        current_sum = sum_pos - sum_neg
        if abs(del_prev / current_sum) < eps(T) && abs(del / current_sum) < eps(T)
            break
        end
        k += one(T)
        if abs((del_pos + del_neg) / current_sum) <= eps(T)
            break
        end
    end
    return sum_pos - sum_neg
end

#----------------------------------------------------------------------
# 13. Hypergeometric 2F1 using Luke’s recurrence
function hyperg_2F1_luke(a::T, b::T, c::T, xin::T) where {T <: AbstractFloat}
    RECUR_BIG = T(1e100)
    nmax = 20000
    n = 3               # starting index (an Int)
    x = -xin          # note the sign change
    x3 = x^3
    t0 = a * b / c
    t1 = (a + one(T)) * (b + one(T)) / (2 * c)
    t2 = (a + T(2)) * (b + T(2)) / (2 * (c + one(T)))
    F = one(T)

    # Initial values for the recurrence:
    Bnm3 = one(T)                                   # B₀
    Bnm2 = one(T) + t1 * x                          # B₁
    Bnm1 = one(T) + t2 * x * (one(T) + (t1/ T(3)) * x)  # B₂

    Anm3 = one(T)                                   # A₀
    Anm2 = Bnm2 - t0 * x                            # A₁
    Anm1 = Bnm1 - t0 * (one(T) + t2 * x) * x + t0 * t1 * (c/(c + one(T))) * x^2  # A₂

    while true
        # Convert n to T for arithmetic
        nT = T(n)
        npam1 = nT + a - one(T)
        npbm1 = nT + b - one(T)
        npcm1 = nT + c - one(T)
        npam2 = nT + a - T(2)
        npbm2 = nT + b - T(2)
        npcm2 = nT + c - T(2)
        tnm1 = 2 * nT - one(T)
        tnm3 = 2 * nT - T(3)
        tnm5 = 2 * nT - T(5)
        n2 = nT^2
        F1 = (T(3)*n2 + (a + b - T(6))*nT + T(2) - a*b - T(2)*(a + b)) /
             (T(2)*tnm3*npcm1)
        F2 = -(T(3)*n2 - (a + b + T(6))*nT + T(2) - a*b) * npam1 * npbm1 /
             (T(4)*tnm1*tnm3*npcm2*npcm1)
        F3 = (npam2 * npam1 * npbm2 * npbm1 * (nT - a - T(2)) * (nT - b - T(2))) /
             (T(8)*tnm3^2*tnm5*(nT + c - T(3))*npcm2*npcm1)
        E = - npam1 * npbm1 * (nT - c - one(T)) / (T(2)*tnm3*npcm2*npcm1)

        An = (one(T) + F1 * x) * Anm1 + (E + F2 * x) * x * Anm2 + F3 * x3 * Anm3
        Bn = (one(T) + F1 * x) * Bnm1 + (E + F2 * x) * x * Bnm2 + F3 * x3 * Bnm3
        r = An / Bn

        prec = abs((F - r) / F)
        F = r

        if prec < eps(T) || n > nmax
            break
        end

        # Scale to avoid overflow/underflow
        if abs(An) > RECUR_BIG || abs(Bn) > RECUR_BIG
            An  /= RECUR_BIG; Bn  /= RECUR_BIG
            Anm1 /= RECUR_BIG; Bnm1 /= RECUR_BIG
            Anm2 /= RECUR_BIG; Bnm2 /= RECUR_BIG
            Anm3 /= RECUR_BIG; Bnm3 /= RECUR_BIG
        elseif abs(An) < one(T)/RECUR_BIG || abs(Bn) < one(T)/RECUR_BIG
            An  *= RECUR_BIG; Bn  *= RECUR_BIG
            Anm1 *= RECUR_BIG; Bnm1 *= RECUR_BIG
            Anm2 *= RECUR_BIG; Bnm2 *= RECUR_BIG
            Anm3 *= RECUR_BIG; Bnm3 *= RECUR_BIG
        end

        n += 1
        Bnm3, Bnm2, Bnm1 = Bnm2, Bnm1, Bn
        Anm3, Anm2, Anm1 = Anm2, Anm1, An
    end

    return F
end

#----------------------------------------------------------------------
# 14. Hypergeometric 2F1 via reflection formulas
function hyperg_2F1_reflect(a::T, b::T, c::T, x::T) where {T <: AbstractFloat}
    locEPS = T(1000) * eps(T)
    d = c - a - b
    intd = floor(d + T(0.5))
    d_integer = abs(d - intd) < locEPS

    if d_integer
        ln_omx = log(one(T) - x)
        ad = abs(d)
        # Initialize F1 and F2
        F1 = zero(T)
        F2 = zero(T)
        if d >= zero(T)
            d1 = d
            d2 = zero(T)
        else
            d1 = zero(T)
            d2 = d
        end
        lng_ad2 = gamln(a + d2)
        lng_bd2 = gamln(b + d2)
        lng_c   = gamln(c)

        # Evaluate F1
        if ad < eps(T)
            F1 = zero(T)
        else
            lng_ad  = gamln(ad)
            lng_ad1 = gamln(a + d1)
            lng_bd1 = gamln(b + d1)
            sum1 = one(T)
            term = one(T)
            ln_pre1_val = lng_ad + lng_c + d2 * ln_omx - lng_ad1 - lng_bd1
            # For integer ad assume ad is an integer value:
            for i in 1:(Int(ad) - 1)
                j = i - 1
                term *= ((a + d2 + T(j)) * (b + d2 + T(j)) /
                         ((one(T) + d2 + T(j)) * T(i))) * (one(T) - x)
                sum1 += term
            end
            F1 = exp(ln_pre1_val) * sum1
        end

        # Evaluate F2 using a psi recurrence
        maxiter = 2000
        psi_1    = -ℯ              # using Julia’s built-in constant ℯ for Euler's number
        psi_1pd  = psi(one(T) + ad)
        psi_apd1 = psi(a + d1)
        psi_bpd1 = psi(b + d1)
        psi_val = psi_1 + psi_1pd - psi_apd1 - psi_bpd1 - ln_omx
        fact = one(T)
        sum2_val = psi_val
        ln_pre2_val = lng_c + d1 * ln_omx - lng_ad2 - lng_bd2
        for j in 1:maxiter
            term1 = one(T)/T(j) + one(T)/(ad + T(j))
            term2 = one(T)/(a + d1 + T(j) - one(T)) + one(T)/(b + d1 + T(j) - one(T))
            psi_val += term1 - term2
            fact *= (a + d1 + T(j) - one(T)) * (b + d1 + T(j) - one(T)) /
                    ((ad + T(j)) * T(j)) * (one(T) - x)
            delta = fact * psi_val
            sum2_val += delta
            if abs(delta) < eps(T) * abs(sum2_val)
                break
            end
        end
        F2 = exp(ln_pre2_val) * sum2_val

        sgn_2 = isodd(Int(intd)) ? -one(T) : one(T)
        return F1 + sgn_2 * F2

    else
        # d is not an integer
        ln_g1ca = gamln(c - a)
        ln_g1cb = gamln(c - b)
        ln_g2a  = gamln(a)
        ln_g2b  = gamln(b)
        ln_gc   = gamln(c)
        ln_gd   = gamln(d)
        ln_gmd  = gamln(-d)
        # Assume positive signs
        sgn1 = one(T)
        sgn2 = one(T)
        ln_pre1_val = ln_gc + ln_gd - ln_g1ca - ln_g1cb
        ln_pre2_val = ln_gc + ln_gmd - ln_g2a - ln_g2b + d * log(one(T) - x)
        pre1 = exp(ln_pre1_val) * sgn1
        pre2 = exp(ln_pre2_val) * sgn2

        F1 = hyperg_2F1_series(a, b, one(T) - d, one(T) - x)
        F2 = hyperg_2F1_series(c - a, c - b, one(T) + d, one(T) - x)
        return pre1 * F1 + pre2 * F2
    end
end

#----------------------------------------------------------------------
# 15. Helper: (1 - x)^p computed via a “safe” logarithm
function pow_omx(x::T, p::T) where {T <: AbstractFloat}
    if abs(x) < sqrt(sqrt(eps(T)))
        ln_omx = -x * (one(T) + x*(one(T)/2 + x*(one(T)/3 + x/4 + x^2/5)))
    else
        ln_omx = log(one(T) - x)
    end
    return exp(p * ln_omx)
end

#----------------------------------------------------------------------
# 16. Main interface to hypergeometric 2F1
function sf_hyperg_2F1(a::T, b::T, c::T, x::T) where {T <: AbstractFloat}
    locEPS = T(1000) * eps(T)
    d = c - a - b
    rinta = floor(a + T(0.5))
    rintb = floor(b + T(0.5))
    rintc = floor(c + T(0.5))
    a_neg_integer = (a < zero(T) && abs(a - rinta) < locEPS)
    b_neg_integer = (b < zero(T) && abs(b - rintb) < locEPS)
    c_neg_integer = (c < zero(T) && abs(c - rintc) < locEPS)

    # Handle x == 1 case when (c - a - b) > 0, c ≠ 0, and c not a negative integer.
    if abs(x - one(T)) < locEPS && (c - a - b) > zero(T) && c != zero(T) && !c_neg_integer
        lngamc    = gamln(c)
        lngamcab  = gamln(c - a - b)
        lngamca   = gamln(c - a)
        lngamcb   = gamln(c - b)
        return exp(lngamc + lngamcab - lngamca - lngamcb)
    end

    if x < -one(T) || x >= one(T)
        return NaN
    end

    if c_neg_integer
        # If c is a negative integer, then either a or b must be a negative integer
        # of smaller magnitude than c to cancel the series.
        if !((a_neg_integer && a > c + T(0.1)) || (b_neg_integer && b > c + T(0.1)))
            return NaN
        end
    end

    if abs(c - b) < locEPS || abs(c - a) < locEPS
        return pow_omx(x, d)  # (1 - x)^(c - a - b)
    end

    if a ≥ zero(T) && b ≥ zero(T) && c ≥ zero(T) && x ≥ zero(T) && x < T(0.995)
        # Series with all positive terms and x not too close to 1.
        return hyperg_2F1_series(a, b, c, x)
    end

    if abs(a) < T(10) && abs(b) < T(10)
        if a_neg_integer
            return hyperg_2F1_series(rinta, b, c, x)
        end
        if b_neg_integer
            return hyperg_2F1_series(a, rintb, c, x)
        end
        if x < T(-0.25)
            return hyperg_2F1_luke(a, b, c, x)
        elseif x < T(0.5)
            return hyperg_2F1_series(a, b, c, x)
        else
            if abs(c) > T(10)
                return hyperg_2F1_series(a, b, c, x)
            else
                return hyperg_2F1_reflect(a, b, c, x)
            end
        end
    else
        # At least one of a or b is large.
        # Let bp be the one of larger magnitude.
        if abs(a) > abs(b)
            bp = a; ap = b
        else
            bp = b; ap = a
        end
        if x < zero(T)
            return hyperg_2F1_luke(a, b, c, x)
        end
        # The C code uses a comma operator so that (fabs(ap), 1.0) evaluates to 1.0.
        if floatmax(T) * abs(bp) * abs(x) < 2 * abs(c)
            return hyperg_2F1_series(a, b, c, x)
        end
        return NaN
    end
end

#----------------------------------------------------------------------
# 17. Compute log(fdtrc): a helper for the F distribution tail probability
function logfdtrc(x::T, ia::Int, ib::Int) where {T <: AbstractFloat}
    if ia < 1 || ib < 1 || x < zero(T)
        error("fdtrc domain error: ia < 1, ib < 1, or x < 0")
    end
    a = T(ia)
    b = T(ib)
    w = b / (b + a * x)
    if w <= zero(T)
        return -Inf
    elseif w >= one(T)
        return zero(T)
    end
    # Compute the hypergeometric function ₂F₁(0.5b + 0.5a, 1, 0.5b + 1, w)
    hyp2f1 = sf_hyperg_2F1(0.5*b + 0.5*a, one(T), 0.5*b + one(T), w)
    ln_beta = betaln(0.5*b, 0.5*a)
    result = log(hyp2f1) + 0.5*b*log(w) + 0.5*a*log(one(T)-w) - log(0.5*b) - ln_beta
    return result
end

#----------------------------------------------------------------------
# 18. Compute z from an F–distribution tail probability
# for the Z-test for covariance
function get_z_cov(R::T, N::Int) where {T <: AbstractFloat}
    result = -logfdtrc(R, 2*N - 3, 2*N - 3)
    if result < zero(T)
        result = zero(T)
    end
    return result
end
