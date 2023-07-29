using Dates, MarketData, TimeSeries, Plots, StateSpaceModels, Distributions, 
      StatsPlots, GLM, HypothesisTests, DataFrames, LinearAlgebra, StatsFuns,
      FLoops, Optim
      

start = DateTime(2018, 1, 1)
btc_price = yahoo(
    "BTC-USD", 
    YahooOpt(period1=start, period2=DateTime(2023, 5, 1), interval="1wk")
)
# eth = yahoo("ETH-USD", opt)

spx_price = yahoo(
    "^GSPC",
    YahooOpt(period1=start, period2=DateTime(2023, 5, 2), interval="1wk")
)
# nq = yahoo("^IXIC", opt)

btc_log_return = percentchange(btc_price, :log)
spx_log_return = percentchange(spx_price, :log)
plot(
    plot(btc_price[:Open, :High, :Low, :Close], title="BTC-USD"),
    # plot(eth[:Open, :High, :Low, :Close], title="ETH-USD"),
    plot(spx_price[:Open, :High, :Low, :Close], title="S&P 500"),
    # plot(nq[:Open, :High, :Low, :Close], title="BNB-USD")
    layout=(2, 1), legend=false
)

plot(
    plot(btc_log_return[:Open, :High, :Low, :Close], title="BTC-USD"),
    # plot(eth[:Open, :High, :Low, :Close], title="ETH-USD"),
    plot(spx_log_return[:Open, :High, :Low, :Close], title="S&P 500"),
    # plot(nq[:Open, :High, :Low, :Close], title="BNB-USD")
    layout=(2, 1), legend=false
)
################################################################################
# ols and adf test
################################################################################
btc = btc_log_return[:AdjClose]
spx = spx_log_return[:AdjClose]

ADFTest(values(btc), :none, 9)
ADFTest(values(spx), :constant, 5)

function lag_(data, lag...; constant=false)
    max_lag = maximum(lag)
    lag_data = [
        values(TimeSeries.lag(data, i, padding=true))
        for i in lag
    ]
    x = reduce(hcat, lag_data)[begin+max_lag:end, :]
    if constant
        x = hcat(ones(size(x, 1)), x)
    end
    y = values(data)[begin+max_lag:end]
    id = timestamp(data)[begin+max_lag:end]

    return x, y, id
end


# spx
spx_ols_c1 = lm(lag_(spx, 1; constant=true)[begin:end-1]...)

spx_ols_4 = lm(lag_(spx, range(1, 4)...)[begin:end-1]...)
spx_ols_c4 = lm(lag_(spx, range(1, 4)...; constant=true)[begin:end-1]...)

spx_ols_5 = lm(lag_(spx, range(1, 5)...)[begin:end-1]...)
spx_ols_c5 = lm(lag_(spx, range(1, 5)...; constant=true)[begin:end-1]...)


# btc
btc_ols_c1 = lm(lag_(btc, 1; constant=true)[begin:end-1]...)

btc_ols_5 = lm(lag_(btc, range(1,5)...)[begin:end-1]...)
btc_ols_c5 = lm(lag_(btc, range(1,5)...; constant=true)[begin:end-1]...)

btc_old_9 = lm(lag_(btc, range(1,9)...)[begin:end-1]...)
btc_old_c9 = lm(lag_(btc, range(1,9)...; constant=true)[begin:end-1]...)


################################################################################
# quasi-maximum likelihood
################################################################################

function quasi_loglikelihood(p, _dist_param, ϕ, y, X, _f)
    dist_param = exp(_dist_param)
    p = normcdf.(p)

    k = 2 * size(X, 2)
    ϕ₀, ϕ₁ = ϕ[begin:begin+Int(k/2-1)], ϕ[begin+Int(k/2):end]

    A = vcat(
        LinearAlgebra.I(2) - [p[1] 1-p[1]; 1-p[2] p[2]],  # p = [p₀₀, p₁₁]
        reshape(ones(2), 1, 2)
    )
   init_pred_proba = ( inv(A'*A) * A' )[:, 3]

    T = size(y, 1)
    pred_proba = Matrix{Float64}(undef, 2, T+1)
    pred_proba[:, begin] = init_pred_proba
    filter_proba = Matrix{Float64}(undef, 2, T)

    f(x) = _f == normpdf ? _f(0, sqrt(dist_param), x) : _f(dist_param, x)
    llh = Vector{Float64}(undef, T)
    for t = eachindex(y)
        ϵ₀, ϵ₁ = y[t] - X[t, :]'*ϕ₀, y[t] - X[t, :]'*ϕ₁
        lhₜ₀, lhₜ₁ = f(ϵ₀), f(ϵ₁)
        lhₜ = lhₜ₀ * pred_proba[1, t] + lhₜ₁ * pred_proba[2, t]

        filter_proba[1, t] = (pred_proba[1, t] * lhₜ₀) / lhₜ
        filter_proba[2, t] = (pred_proba[2, t] * lhₜ₁) / lhₜ

        pred_proba[1, t+1] = p[1]     * filter_proba[1, t] + (1-p[2]) * filter_proba[2, t]
        pred_proba[2, t+1] = (1-p[1]) * filter_proba[1, t] + p[2]     * filter_proba[2, t]

        llh[t] = log(lhₜ)
    end

    return mean(llh), pred_proba, filter_proba
end


struct result
    opt_minimizer::Vector{Float64}
    coevec::Vector{Float64}
    stderror::Vector{Float64}
    smooth_proba::Matrix{Float64}
    resid::Vector{Float64}
    AIC::Float64
    BIC::Float64
end

function numeric_optimize(X, y, _init=nothing)
    obj_fn = θ -> -quasi_loglikelihood(
        θ[begin:begin+1], θ[begin+2], θ[begin+3:end], 
        y, X, StatsFuns.normpdf
    )[1]
    init = isnothing(_init) ? [0., 0., 2, fill(0.1, 2*size(X, 2))...] : _init

    h = TwiceDifferentiable(obj_fn, ones(length(init)))
    opt = optimize(
        h, 
        init,
        BFGS(), 
        Optim.Options(
            g_tol=1e-8, 
            iterations=2000, 
            store_trace=false, 
            show_trace=false
        )
    )

    minimizer = opt.minimizer
    coevec = deepcopy(minimizer)
    coevec[begin:begin+1] = normcdf.(coevec[begin:begin+1])
    coevec[begin+2] = exp(coevec[begin+2]);

    coevec[begin:begin+2] |> display
    [coevec[begin+3:begin+3+size(X, 2)-1] coevec[begin+3+size(X, 2):end]] |> display
    println()

    # delta method
    stderror = sqrt.(diag(inv(Optim.hessian!(h, opt.minimizer))))
    stderror[begin:begin+1] = stderror[begin:begin+1] .* normpdf.(minimizer[begin:begin+1])
    stderror[begin+2] = stderror[begin+2] * exp(minimizer[begin+2]);

    stderror[begin:begin+2] |> display
    [stderror[begin+3:begin+3+size(X, 2)-1] stderror[begin+3+size(X, 2):end]] |> display


    predict_proba, filter_proba = quasi_loglikelihood(
        quantile.(Normal(), coevec[begin:begin+1]), 
        log(coevec[begin+2]), 
        coevec[begin+3:end], 
        y, X, StatsFuns.normpdf
    )[2:3]
    p = coevec[begin:begin+1]

    T = size(filter_proba, 2)
    smooth_proba = Matrix{Float64}(undef, 2, T); smooth_proba[:, end] = filter_proba[:, end]
    for t = 1:T-1
        curr = T-t
        smooth_proba[1, curr] = filter_proba[1, curr] * (
            (p[1]*smooth_proba[1, curr+1])/predict_proba[1, curr+1] +
            ((1-p[1])*smooth_proba[2, curr+1])/predict_proba[2, curr+1]
        )

        smooth_proba[2, curr] = filter_proba[2, curr] * (
            ((1-p[2])*smooth_proba[1, curr+1])/predict_proba[1, curr+1] +
            (p[2]*smooth_proba[2, curr+1])/predict_proba[2, curr+1]
        )
    end


    k = 2 * size(X, 2)
    ϕ = coevec[begin+3:end]
    ϕ₀, ϕ₁ = ϕ[begin:begin+Int(k/2-1)], ϕ[begin+Int(k/2):end]

    res = similar(y)
    for i = eachindex(res)
        ŷ = [X[i, :] .* ϕ₀ X[i, :] .* ϕ₁] * smooth_proba[:, i]
        res[i] = y[i] - sum(ŷ)
    end
    SSR = sum(res .^2)
    T = length(y)
    AIC = T * log(SSR/T) + 2(k+3)
    BIC = T * log(SSR/T) + (k+3)log(T)

    println()

    [["AIC", "BIC"] [AIC, BIC]] |> display

    return result(minimizer, coevec, stderror, smooth_proba, res, AIC, BIC)
end


Bz(a) = (a .- mean(a)) ./ std(a)


################################################################################
# model selectio and residual
################################################################################

spx_msar_1 = numeric_optimize(lag_(spx, 1; constant=true)[begin:end-1]...);

spx_masar_4 = numeric_optimize(lag_(spx, 1:4...; constant=true)[begin:end-1]...);

spx_msar_5 = numeric_optimize(lag_(spx, 1:5...; constant=true)[begin:end-1]...);


btc_msar_1 = numeric_optimize(lag_(btc, 1; constant=true)[begin:end-1]...);

btc_msar_5 = numeric_optimize(lag_(btc, 1:5...; constant=true)[begin:end-1]...);

btc_msar_9 = numeric_optimize(lag_(btc, 1:9...; constant=true)[begin:end-1]...);


test_resid = z(spx_masar_4.resid)
JarqueBeraTest(test_resid)

histogram(test_resid, normalize=true, labels="standardized residuals")
plot!(Normal(), labels="pdf of Standard Normal")

test_resid = z(btc_msar_5.resid)
JarqueBeraTest(test_resid)

histogram(test_resid, normalize=true, labels="standardized residuals")
plot!(Normal(), labels="pdf of Standard Normal")

################################################################################


################################################################################
# filtering probability
################################################################################
x, y_spx, id_spx = lag_(spx, 1:4...; constant=true)
spx_msar_4 = numeric_optimize(x, y_spx)
filter_proba_spx = spx_msar_4.smooth_proba
pred_state_spx = [
    filter_proba_spx[1, i] >= filter_proba_spx[2, i] ? 0 : 1
    for i = axes(filter_proba_spx, 2)
]

plot(
    plot(TimeArray(id_spx, filter_proba_spx[2, :]), title ="smoothing probability in state 0"),
    plot(TimeArray(id_spx, y_spx), title="S%&P500 weekly return"),
    plot(TimeArray(id_spx, 1 .- pred_state_spx), title="predicted state"),
    plot(TimeArray(id_spx, filter_proba_spx[1, :]), title="smoothing probability in state 1"),
    layout=(4, 1), legend=false
)

x, y_btc, id_btc = lag_(btc, 1:5...; constant=true)
spx_msar_5 = numeric_optimize(x, y_btc)
filter_proba_btc = btc_msar_5.smooth_proba
pred_state_btc = [
    filter_proba_btc[1, i] >= filter_proba_btc[2, i] ? 0 : 1
    for i = axes(filter_proba_btc, 2)
]
plot(
    plot(TimeArray(id_btc, filter_proba_btc[1, :]), title ="smoothing probability in state 0"),
    plot(TimeArray(id_btc, y_btc), title="BTC-USD weekly return "),
    plot(TimeArray(id_btc, pred_state_btc), title="predicted state"),
    plot(TimeArray(id_btc, filter_proba_btc[2, :]), title="smoothing probability in state 1"),
    layout=(4, 1), legend=false
)


plot(
    plot(
        TimeArray(id_spx[begin+1:begin+91], 
        1 .- pred_state_spx[begin+1:begin+91]), 
        labels="spx",
        color=:orange,
        linewidth=2
    ),
    plot(
        TimeArray(id_btc[begin:begin+90], pred_state_btc[begin:begin+90]), 
        labels="btc", linewidth=2,
    ),
    layout=(2, 1)
)



plot(
    plot(
        TimeArray(id_spx[begin+92:begin+182], 
        1 .- pred_state_spx[begin+92:begin+182]), 
        labels="spx",
        color=:orange,
        linewidth=2
    ),
    plot(
        TimeArray(id_btc[begin+91:begin+181], pred_state_btc[begin+91:begin+181]), 
        labels="btc", linewidth=2,
    ),
    layout=(2, 1)
)



plot(
    plot(
        TimeArray(id_spx[begin+183:end], 
        1 .- pred_state_spx[begin+183:end]), 
        labels="spx",
        color=:orange,
        linewidth=2
    ),
    plot(
        TimeArray(id_btc[begin+182:end], pred_state_btc[begin+182:end]), 
        labels="btc", linewidth=2,
    ),
    layout=(2, 1)
)
################################################################################
# ARMA
################################################################################
# AR1
auto_arima(values(btc_log_return[:AdjClose]), max_p=20, max_q=20)

p = 1:10
ar_aic = Vector{Float64}(undef, length(p))
ar_bic = similar(ar_aic)
for i = eachindex(p, ar_aic, ar_bic)
    sarima = SARIMA(values(btc_log_return[:AdjClose]); order = (p[i], 0, 0))
    fit!(sarima)
    res = sarima.results
    ar_aic[i], ar_bic[i] = res.aic, res.bic
end

plot([aic bic], label=["aic" "bic"], title="AR")

y = values(btc_log_return[:AdjClose])
sarima = SARIMA(y; order = (1, 0, 0))
fit!(sarima)
ar1, sigma = sarima.results.coef_table.coef