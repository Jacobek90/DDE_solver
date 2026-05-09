using LinearAlgebra
using Plots
using Printf
using LaTeXStrings

struct History
    t::Float64
    y::Float64
    K::Vector{Float64}
    h::Float64
end

struct CRK4Method{T <: AbstractFloat}
    a::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    b_theta::Vector{Function}
end

function find_breaking(tau::Function, ξ0::Real) # Funkce hledajici body zlomu
    Ξ = [ξ0]
    δ = 1e-8
    max_iter = 100

    for i in 1:4
        ξ = Ξ[i]
        D0 = (tau(ξ + δ) - tau(ξ - δ)) / (2 * δ)
        tn = ξ + tau(ξ) / (1 - D0)

        iter = 0
        while abs(tn - ξ - tau(tn)) >= 1e-12 && iter < max_iter
            D = (tau(tn + δ) - tau(tn - δ)) / (2 * δ)
            tn = tn - (tn - ξ - tau(tn)) / (1 - D)
            iter += 1
        end

        if iter == max_iter
            break
        end

        push!(Ξ, tn)
    end
    return Ξ
end

function search(history::Vector{History}, t_query::Float64) # binarni vyhledavani casu z historie
    left = 1
    right = length(history)
    idx = 0

    while left <= right
        mid = (left + right) ÷ 2
        if history[mid].t <= t_query
            idx = mid
            left = mid + 1
        else
            right = mid - 1
        end
    end

    return idx
end

function fy(f, t, y, ydel) # parcialni derivace
    δ = 1e-8
    (f(t, y + δ, ydel) - f(t, y - δ, ydel)) / (2 * δ)
end

function fyd(f, t, y, ydel)
    δ = 1e-8
    (f(t, y, ydel + δ) - f(t, y, ydel - δ)) / (2 * δ)
end

function get_delayed(phi::Function, t_query::Float64, t0::Real, history::Vector{History}, method::CRK4Method, overlap::AbstractVector{Bool}, j::Int) # funkce vracejici hodnotu vnitrni faze Y
    if t_query <= t0
        return phi(t_query), overlap
    end

    step = history[end]
    current_t = step.t + step.h
    if t_query > current_t
        overlap[j] = true
        return 0.0, overlap
    end

    idx = search(history, t_query)
    step = history[idx]
    theta = (t_query - step.t) / step.h
    interpolated_val = step.y + step.h * sum(method.b_theta[i](theta) * step.K[i] for i in 1:5) # interpolace na intervalu v minulosti 5 stupnovym rozsirenim
    return interpolated_val, overlap
end

function newton_overlap(k::Vector{Float64}, f::Function, tau::Function, overlap::AbstractVector{Bool}, h::Float64, method::CRK4Method, history::Vector{History}, current_t::Float64, current_y::Float64) # Newtnova metoda pro implicitni soustavu
    n_overlap = sum(overlap)
    if n_overlap == 0
        return k
    end
    first_j = findfirst(overlap)
    if first_j > 1
        k_pred = k[first_j - 1]
        for j in first_j:5
            k[j] = k_pred
        end
    end
    n = 5 - first_j + 1
    J = zeros(n, n)
    R = zeros(n)
    tol = 1e-12
    max_iter = 100
    iter = 0
    delta = fill(Inf, n)
    while maximum(abs.(delta)) >= tol && iter < max_iter
        iter += 1
        for p in 1:n
            j = first_j + p - 1
            t_stage = current_t + method.c[j] * h
            delayed_t = t_stage - tau(t_stage)
            theta = (delayed_t - current_t) / h

            ydel = current_y + h * sum(method.b_theta[m](theta) * k[m] for m in 1:5) # zde probiha priprava vnitrnich fazi
            stage_y = current_y + h * sum(method.a[j, l] * k[l] for l in 1:j-1)

            R[p] = k[j] - f(t_stage, stage_y, ydel)

            fy_val = fy(f, t_stage, stage_y, ydel)
            fyd_val = fyd(f, t_stage, stage_y, ydel)
            for q in 1:n
                s = first_j + q - 1
                if j == s
                    J[p, q] = 1.0 - h * method.a[j, s] * fy_val - h * method.b_theta[s](theta) * fyd_val
                else
                    J[p, q] = -h * method.a[j, s] * fy_val - h * method.b_theta[s](theta) * fyd_val
                end
            end
        end

        if !all(isfinite, J) || !all(isfinite, R)
            @warn "Inf/NaN"
            break
        end
        delta = -J \ R # Iterace Newtonovy metody
        for p in 1:n
            j = first_j + p - 1
            k[j] += delta[p]
        end
    end
    if iter >= max_iter
        @warn "Newton iteration did not converge"
    end
    return k
end

function solve_dde(f::Function, tau::Function, t0::Real, tn::Real, y0::Real, h::Real, phi::Function) # samotny resic
    a = [0.0 0.0 0.0 0.0 0.0;
         0.5 0.0 0.0 0.0 0.0;
         0.0 0.5 0.0 0.0 0.0;
         0.0 0.0 1.0 0.0 0.0;
         1/6 1/3 1/3 1/6 0.0]

    b = [1/6, 1/3, 1/3, 1/6, 0.0]
    c = [0.0, 0.5, 0.5, 1.0, 1.0]

    b_theta = [θ -> θ - 1.5*θ^2 + (2/3)*θ^3,
               θ -> θ^2 - (2/3)*θ^3,
               θ -> θ^2 - (2/3)*θ^3,
               θ -> 0.5*θ^2 - (1/3)*θ^3,
               θ -> -θ^2 + θ^3]

    method = CRK4Method{Float64}(a, b, c, b_theta)
    history = Vector{History}()
    push!(history, History(t0, y0, zeros(5), h))

    t_break = find_breaking(tau, t0)
    t_break = sort(t_break)

    valid_breaks = filter(t -> t <= tn, t_break)
    n_intervals = length(valid_breaks) - 1
    y = [y0]
    tt = [t0]
    num = 1

    for i in 1:n_intervals
        next_break = t_break[i + 1]
        while tt[num] + h < next_break
            overlap = falses(5)
            y_del = zeros(5)
            t_a = zeros(5)
            k = zeros(5)
            current_t = tt[num]
            current_y = y[num]
            t_a[1] = current_t
            y_del[1], overlap = get_delayed(phi, t_a[1] - tau(t_a[1]), t0, history, method, overlap, 1)
            k[1] = f(t_a[1], current_y, y_del[1])
            for j in 2:5
                t_a[j] = current_t + method.c[j] * h
                y_del[j], overlap = get_delayed(phi, t_a[j] - tau(t_a[j]), t0, history, method, overlap, j)
                if !overlap[j]
                    stage_y = current_y + h * sum(method.a[j, l] * k[l] for l in 1:j-1)
                    k[j] = f(t_a[j], stage_y, y_del[j])
                else
                    for jj = j:5
                        overlap[jj] = true
                    end
                    k = newton_overlap(k, f, tau, overlap, h, method, history, current_t, current_y)
                    break
                end
            end
            y_new = current_y + h * sum(method.b[i] * k[i] for i in 1:4)
            push!(y, y_new)
            push!(tt, current_t + h)
            push!(history, History(current_t, current_y, k, h))
            num += 1
        end
        h2 = next_break - tt[num]
        if h2 > 0
            overlap = falses(5)
            y_del = zeros(5)
            t_a = zeros(5)
            k = zeros(5)
            current_t = tt[num]
            current_y = y[num]
            t_a[1] = current_t
            y_del[1], overlap = get_delayed(phi, t_a[1] - tau(t_a[1]), t0, history, method, overlap, 1)
            k[1] = f(t_a[1], current_y, y_del[1])
            for j in 2:5
                t_a[j] = current_t + method.c[j] * h2
                y_del[j], overlap = get_delayed(phi, t_a[j] - tau(t_a[j]), t0, history, method, overlap, j)
                if !overlap[j]
                    stage_y = current_y + h2 * sum(method.a[j, l] * k[l] for l in 1:j-1)
                    k[j] = f(t_a[j], stage_y, y_del[j])
                else
                    for jj = j:5
                        overlap[jj] = true
                    end
                    k = newton_overlap(k, f, tau, overlap, h2, method, history, current_t, current_y)
                    break
                end
            end
            y_new = current_y + h2 * sum(method.b[i] * k[i] for i in 1:4)
            push!(y, y_new)
            push!(tt, current_t + h2)
            push!(history, History(current_t, current_y, k, h2))
            num += 1
        end
    end

    while tt[num] + h < tn # Zde cyklus pokracuje po poslednim bodu zlomu 
        overlap = falses(5)
        y_del = zeros(5)
        t_a = zeros(5)
        k = zeros(5)
        current_t = tt[num]
        current_y = y[num]
        t_a[1] = current_t
        y_del[1], overlap = get_delayed(phi, t_a[1] - tau(t_a[1]), t0, history, method, overlap, 1)
        k[1] = f(t_a[1], current_y, y_del[1])
        for j in 2:5
            t_a[j] = current_t + method.c[j] * h
            y_del[j], overlap = get_delayed(phi, t_a[j] - tau(t_a[j]), t0, history, method, overlap, j)
            if !overlap[j]
                stage_y = current_y + h * sum(method.a[j, l] * k[l] for l in 1:j-1)
                k[j] = f(t_a[j], stage_y, y_del[j])
            else
                for jj = j:5
                    overlap[jj] = true
                end
                k = newton_overlap(k, f, tau, overlap, h, method, history, current_t, current_y)
                break
            end
        end
        y_new = current_y + h * sum(method.b[i] * k[i] for i in 1:4)
        push!(y, y_new)
        push!(tt, current_t + h)
        push!(history, History(current_t, current_y, k, h))
        num += 1
    end
    h2 = tn - tt[num]
    if h2 > 0
        overlap = falses(5)
        y_del = zeros(5)
        t_a = zeros(5)
        k = zeros(5)
        current_t = tt[num]
        current_y = y[num]
        t_a[1] = current_t
        y_del[1], overlap = get_delayed(phi, t_a[1] - tau(t_a[1]), t0, history, method, overlap, 1)
        k[1] = f(t_a[1], current_y, y_del[1])
        for j in 2:5
            t_a[j] = current_t + method.c[j] * h2
            y_del[j], overlap = get_delayed(phi, t_a[j] - tau(t_a[j]), t0, history, method, overlap, j)
            if !overlap[j]
                stage_y = current_y + h2 * sum(method.a[j, l] * k[l] for l in 1:j-1)
                k[j] = f(t_a[j], stage_y, y_del[j])
            else
                for jj = j:5
                    overlap[jj] = true
                end
                k = newton_overlap(k, f, tau, overlap, h2, method, history, current_t, current_y)
                break
            end
        end
        y_new = current_y + h2 * sum(method.b[i] * k[i] for i in 1:4)
        push!(y, y_new)
        push!(tt, current_t + h2)
        push!(history, History(current_t, current_y, k, h2))
        num += 1
    end

    return y, tt
end


## Testovaci ulohy
#=
let
    println("\n=== ANALÝZA KONVERGENCE: Hutchinsonova rovnice (MMS) ===")
    
    r = 0.5
    K = 10.0
    tau_Hutch(t) = 2.0 + 0.5 * cos(t)
    
    exact_sol(t) = 5.0 + 2.0 * sin(t)
    exact_sol_deriv(t) = 2.0 * cos(t)
    exact_del(t) = exact_sol(t - tau_Hutch(t))
    
    R(t) = exact_sol_deriv(t) - (r * exact_sol(t) * (1.0 - exact_del(t) / K))
    f_Hutch(t, y, ydel) = r * y * (1.0 - ydel / K) + R(t)
    phi_Hutch(t) = exact_sol(t)
    
    t0_Hutch = 0.0
    tn_end_Hutch = 50.0
    y0_Hutch = exact_sol(0.0)
    
    h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    errors = Float64[]
    
    println("-"^65)
    @printf("| %-10s | %-22s | %-23s |\n", "Krok h", "Max. absolutní chyba", "Empirický řád (EOC)")
    println("-"^65)
    
    for i in 1:length(h_vals)
        h = h_vals[i]
        
        y_num, t_num = solve_dde(f_Hutch, tau_Hutch, t0_Hutch, tn_end_Hutch, y0_Hutch, h, phi_Hutch)
        
        y_exact = exact_sol.(t_num)
        max_err = maximum(abs.(y_num .- y_exact))
        push!(errors, max_err)
        
        if i == 1
            @printf("| %-10.5f | %-22.5e | %-23s |\n", h, max_err, "-")
        else
            h_prev = h_vals[i-1]
            err_prev = errors[i-1]
            
            eoc = log(err_prev / max_err) / log(h_prev / h)
            
            @printf("| %-10.5f | %-22.5e | %-23.4f |\n", h, max_err, eoc)
        end
    end
end
=#

#=
using Plots
using Printf
let
    println("\n=== VYKRESLENÍ: Hutchinsonova rovnice (MMS s časově proměnným zpožděním) ===")
    
    r = 0.5
    K = 10.0

    tau_Hutch(t) = 2.0 + 0.5 * cos(t)
    
    exact_sol(t) = 5.0 + 2.0 * sin(t)
    exact_sol_deriv(t) = 2.0 * cos(t)
    
    exact_del(t) = exact_sol(t - tau_Hutch(t))
    

    R(t) = exact_sol_deriv(t) - (r * exact_sol(t) * (1.0 - exact_del(t) / K))
    
    f_Hutch(t, y, ydel) = r * y * (1.0 - ydel / K) + R(t)
    phi_Hutch(t) = exact_sol(t)
    

    t0_Hutch = 0.0
    tn_end_Hutch = 100.0
    y0_Hutch = exact_sol(0.0)
    
    h_plot = 0.1 
    
    y_num, t_num = solve_dde(f_Hutch, tau_Hutch, t0_Hutch, tn_end_Hutch, y0_Hutch, h_plot, phi_Hutch)
    y_exact = exact_sol.(t_num)
    
    
    t_dense = range(t0_Hutch, tn_end_Hutch, length=1000)
    y_dense = exact_sol.(t_dense)
    
    p1 = plot(t_dense, y_dense, 
            label="Analytické řešení", 
            linewidth=2, 
            color=:blue,
            title="Testovací úloha Hutchinsonovy rovnice",
            xlabel=L"Čas $t$",                             
            ylabel=L"Populace $y(t)$",
            legend=:bottomright)

    scatter!(p1, t_num, y_num, 
            label=latexstring("Aproximace (\$h = $(h_plot)\$)"),
            markersize=2, 
            color=:red, 
            markerstrokewidth=0, 
            alpha=0.7)
    
    error_vals = abs.(y_num .- y_exact)
    p2 = plot(t_num, error_vals, 
          label="Absolutní chyba", 
          color=:darkorange, 
          linewidth=2,
          title="Vývoj chyby integrace v čase", 
          xlabel=L"Čas $t$",
          ylabel=L"|y_{\mathrm{num}}(t) - y_{\mathrm{exact}}(t)|",
          legend=:topright)
    
    
    p_final = plot(p1, p2, layout=(2, 1), size=(800, 600), margin=5Plots.mm)
    display(p_final)
    
    @printf("Vykreslení dokončeno. Maximální zjištěná absolutní chyba: %e\n", maximum(error_vals))
end
=#

#=
let
    # Parametry logistického modelu
    r = 0.5    # Přirozená rychlost růstu
    K = 10.0   # Nosná kapacita prostředí

    # Definice časově proměnného zpoždění tau(t)
    tau_Hutch(t) = 2.0 + 0.5 * sin(t)

    # Výsledná pravá strana DDE předávaná numerickému řešiči
    f_Hutch(t, y, ydel) = r * y * (1.0 - ydel / K)

    phi_Hutch(t) = 1.0

    # Časová a počáteční podmínka
    t0_Hutch = 0.0
    tn_end_Hutch = 100.0
    y0_Hutch = phi_Hutch(0.0)

    h_plot = 0.1 


    y_num, t_num = solve_dde(f_Hutch, tau_Hutch, t0_Hutch, tn_end_Hutch, y0_Hutch, h_plot, phi_Hutch)


    p1 = plot(t_num, y_num, label="Aproximace (h = $h_plot)", linewidth=2, color=:red, 
            title="Řešení Hutchinsonovy rovnice s proměnným zpožděním", 
            xlabel="Čas (t)", ylabel="Populace y(t)", legend=:bottomright)


    hline!(p1, [K], label="Nosná kapacita K = $K", linestyle=:dash, color=:black, alpha=0.6)

    display(p1)
end
=#

#=
using Printf
using Plots
using LaTeXStrings


let
    println("\n=== ANALÝZA KONVERGENCE: Mackey-Glassova rovnice (MMS) ===")
    
    # Standardní parametry pro chaotický režim
    beta = 0.2
    gamma = 0.1
    n_MG = 10.0
    tau_const = 17.0
    tau_MG(t) = tau_const
    
    # Hladká, striktně kladná testovací funkce
    exact_sol(t) = 1.2 + 0.2 * sin(t)
    exact_sol_deriv(t) = 0.2 * cos(t)
    exact_del(t) = exact_sol(t - tau_const)
    
    # Konstrukce rezidua a upravené pravé strany
    R(t) = exact_sol_deriv(t) - (beta * exact_del(t) / (1.0 + exact_del(t)^n_MG) - gamma * exact_sol(t))
    f_test(t, y, ydel) = beta * ydel / (1.0 + ydel^n_MG) - gamma * y + R(t)
    phi_test(t) = exact_sol(t)
    
    t0 = 0.0
    tn_end = 50.0
    y0 = exact_sol(0.0)
    
    # --- Analýza konvergence ---
    h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125]
    errors = Float64[]
    
    println("-"^65)
    @printf("| %-10s | %-22s | %-23s |\n", "Krok h", "Max. absolutní chyba", "Empirický řád (EOC)")
    println("-"^65)
    
    for i in 1:length(h_vals)
        h = h_vals[i]
        y_num, t_num = solve_dde(f_test, tau_MG, t0, tn_end, y0, h, phi_test)
        
        y_exact = exact_sol.(t_num)
        max_err = maximum(abs.(y_num .- y_exact))
        push!(errors, max_err)
        
        if i == 1
            @printf("| %-10.5f | %-22.5e | %-23s |\n", h, max_err, "-")
        else
            eoc = log(errors[i-1] / max_err) / log(h_vals[i-1] / h)
            @printf("| %-10.5f | %-22.5e | %-23.4f |\n", h, max_err, eoc)
        end
    end
    
    # --- Vykreslení testovací úlohy (pro nejmenší krok) ---
    h_plot = h_vals[end]
    y_num_plot, t_num_plot = solve_dde(f_test, tau_MG, t0, tn_end, y0, h_plot, phi_test)
    y_exact_plot = exact_sol.(t_num_plot)
    
    t_dense = range(t0, tn_end, length=1000)
    y_dense = exact_sol.(t_dense)
    
    p1 = plot(t_dense, y_dense, label="Analytické řešení", linewidth=2, color=:blue,
              title="Testovací úloha Mackey-Glassovy rovnice",
              xlabel=L"Čas $t$", ylabel=L"Populace $y(t)$", legend=:bottomright)
    scatter!(p1, t_num_plot[1:20:end], y_num_plot[1:20:end], 
             label=latexstring("Aproximace (\$h = $(h_plot)\$)"), 
             markersize=3, color=:red, markerstrokewidth=0)
             
    p2 = plot(t_num_plot, abs.(y_num_plot .- y_exact_plot), label="Absolutní chyba", 
              color=:darkorange, linewidth=2, title="Vývoj chyby integrace v čase",
              xlabel=L"Čas $t$", ylabel=L"|y_{\mathrm{num}}(t) - y_{\mathrm{exact}}(t)|", legend=:topright)
              
    p_final = plot(p1, p2, layout=(2, 1), size=(800, 600), margin=5Plots.mm)
    display(p_final)
end
=#

#=
let
    println("\n=== VYKRESLENÍ: Mackey-Glassova rovnice ===")
    
    beta = 0.2
    gamma = 0.1
    n_MG = 10.0
    tau_const = 17.0
    tau_MG(t) = tau_const
    
    f_MG(t, y, ydel) = beta * ydel / (1.0 + ydel^n_MG) - gamma * y
    
    phi_MG(t) = 1.2
    
    t0 = 0.0
    tn_end = 2000.0
    y0 = phi_MG(0.0)
    
    h = 0.1 
    
    y_num, t_num = solve_dde(f_MG, tau_MG, t0, tn_end, y0, h, phi_MG)
    
    idx_start = findfirst(>=(500.0), t_num)
    
    delay_steps = round(Int, tau_const / h)
    
    y_t = y_num[idx_start:end]
    y_t_minus_tau = y_num[(idx_start - delay_steps):(end - delay_steps)]
    

    idx_plot_start = findfirst(>=(1700.0), t_num)
    p1 = plot(t_num[idx_plot_start:end], y_num[idx_plot_start:end], 
              label="Stav populace", linewidth=1.5, color=:darkcyan,
              title="Řešení Mackey-Glassovy rovnice",
              xlabel=L"Čas $t$", ylabel=L"Populace $y(t)$", legend=:topright)
              

    p2 = plot(y_t, y_t_minus_tau, label="", color=:purple, linewidth=0.5, alpha=0.8,
              title="Fázový portrét",
              xlabel=L"y(t)", ylabel=L"y(t-\tau)", aspect_ratio=:equal)
              
    p_final = plot(p1, p2, layout=(1, 2), size=(1000, 450), margin=5Plots.mm)
    display(p_final)
end
=#

#=
using Plots
using Printf
using LaTeXStrings

let
    println("\n=== ANALÝZA KONVERGENCE: Lineární rovnice (Metoda kroků) ===")
    
    tau_val = 1.0
    tau_lin(t) = tau_val
    
    function exact_sol_steps(t)
        if t <= 0.0
            return 1.0
        elseif t <= 1.0
            return t + 1.0
        elseif t <= 2.0
            return 0.5 * (t - 1.0)^2 + (t - 1.0) + 2.0
        elseif t <= 3.0
            return (1.0/6.0) * (t - 2.0)^3 + 0.5 * (t - 2.0)^2 + 2.0 * (t - 2.0) + 3.5
        elseif t <= 4.0
            y3 = 37.0 / 6.0 
            return (1.0/24.0) * (t - 3.0)^4 + (1.0/6.0) * (t - 3.0)^3 + (t - 3.0)^2 + 3.5 * (t - 3.0) + y3
        elseif t <= 5.0
            y3 = 37.0 / 6.0
            y4 = 87.0 / 8.0
            return (1.0/120.0) * (t - 4.0)^5 + (1.0/24.0) * (t - 4.0)^4 + (1.0/3.0) * (t - 4.0)^3 + 1.75 * (t - 4.0)^2 + y3 * (t - 4.0) + y4
        else
            error("Analytické řešení je implementováno pouze pro t ∈ [-1, 5]")
        end
    end

 
    f_lin(t, y, ydel) = ydel
    phi_lin(t) = 1.0
    
    t0 = 0.0
    tn_end = 5.0
    y0 = phi_lin(0.0)
    

    h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    errors = Float64[]
    
    println("-"^65)
    @printf("| %-10s | %-22s | %-23s |\n", "Krok h", "Max. absolutní chyba", "Empirický řád (EOC)")
    println("-"^65)
    
    for i in 1:length(h_vals)
        h = h_vals[i]
        
        y_num, t_num = solve_dde(f_lin, tau_lin, t0, tn_end, y0, h, phi_lin)
        
        y_exact = exact_sol_steps.(t_num)
        max_err = maximum(abs.(y_num .- y_exact))
        push!(errors, max_err)
        
        if i == 1
            @printf("| %-10.5f | %-22.5e | %-23s |\n", h, max_err, "-")
        else
            h_prev = h_vals[i-1]
            err_prev = errors[i-1]
            
            eoc = log(err_prev / max_err) / log(h_prev / h)
            @printf("| %-10.5f | %-22.5e | %-23.4f |\n", h, max_err, eoc)
        end
    end

    h_plot = 0.05
    y_num_plot, t_num_plot = solve_dde(f_lin, tau_lin, t0, tn_end, y0, h_plot, phi_lin)
    y_exact_plot = exact_sol_steps.(t_num_plot)

    t_dense = range(t0, tn_end, length=1000)
    y_dense = exact_sol_steps.(t_dense)

    p1 = plot(t_dense, y_dense, 
              label="Analytické řešení", linewidth=2, color=:blue,
              title="Řešení lineární testovací rovnice",
              xlabel=L"Čas $t$", ylabel=L"Hodnota $y(t)$", legend=:topleft)

    scatter!(p1, t_num_plot, y_num_plot, 
             label=latexstring("Aproximace (\$h = $(h_plot)\$)"),
             markersize=3, color=:red, markerstrokewidth=0, alpha=0.7)
             
    # Vyznačení bodů nespojitosti derivací
    vline!(p1, [1.0, 2.0, 3.0], label="Zlomové body (k * tau)", linestyle=:dash, color=:gray, alpha=0.6)

    error_vals_plot = abs.(y_num_plot .- y_exact_plot)
    p2 = plot(t_num_plot, error_vals_plot, 
              label="Absolutní chyba", color=:darkorange, linewidth=2,
              title="Vývoj chyby", 
              xlabel=L"Čas $t$", ylabel=L"|y_{\mathrm{num}}(t) - y_{\mathrm{exact}}(t)|", legend=:topleft)
              
    vline!(p2, [1.0, 2.0, 3.0], label="", linestyle=:dash, color=:gray, alpha=0.6)

    p_final = plot(p1, p2, layout=(2, 1), size=(800, 650), margin=5Plots.mm)
    display(p_final)
end
=#