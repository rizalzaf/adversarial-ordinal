## Adversarial Ordinal Regression with Multiclass Features

include("types.jl")
include("shared.jl")

# find game value
function solve_exact(psi::Vector)
  nc = length(psi)

  mi = -Inf
  mj = -Inf
  id_i = 0
  id_j = 0
  for i = 1:nc
    if psi[i] - i > mi
      mi = psi[i] - i
      id_i = i
    end
    if psi[i] + i >= mj
      mj = psi[i] + i
      id_j = i
    end
  end

  gv = (mi + mj) / 2

  return gv, [id_i, id_j]
end

# calculate gradient
function calc_grad(ids::Vector, xi::Vector, yi::Integer, nc::Integer, idmi::Vector)

  m = length(xi)
  gr = zeros(m * nc)

  if ids[1] == ids[2] && ids[1] == yi
  else
    if ids[1] == ids[2]
      gr[idmi[ids[1]]] = xi
    else
      gr[idmi[ids[1]]] = xi / 2
      gr[idmi[ids[2]]] = xi / 2
    end
    gr[idmi[yi]] -= xi
  end

  return gr
end

# function and gradient
function fg!(theta::Vector, g::Vector, xi::Vector, yi::Integer, lambda::Real, nc::Integer, idmi::Vector)

  psi = psi_list(theta, xi, yi, nc, idmi)
  gv, ids = solve_exact(psi)
  f = gv + (lambda / 2 ) * dot(theta, theta)

  gr = calc_grad(ids, xi, yi, nc, idmi)
  g[:] = gr + lambda * theta

  return f
end

# train ordinal regressin adversarial
function train_or_adv_mc(X::Matrix, y::Vector, lambda::Float64=0.0;
  step::Real=0.1, ftol::Real=1e-6, grtol::Real=1e-6, show_trace::Bool=true, max_iter::Int=1000, verbose::Bool=true)

  n = length(y)
  # add one
  X1 = [ones(n) X]'   # transpose
  m = size(X1, 1)

  # number of class
  nc = maximum(y)
  nf = nc * m   # number of features

  # prepare saved vars
  idmi = map(i -> idi(m, i), collect(1:nc))

  # parameters. init with zero
  theta = rand(nf) - 0.5

  # storing ids
  IDS_storage = zeros(Int64, 2, n)
  GV_storage = zeros(n)

  f_prev = Inf
  iter = 0
  pass_iter = 0
  n_sampled = 0
  is_sampled = zeros(Bool, n)
  diff = zeros(nf)
  gv_sum = 0.0

  while true
    iter = iter + 1

    i = rand(1:n)   # take sample
    xi = X1[:,i]
    yi = y[i]

    if !is_sampled[i]
      n_sampled += 1
    end

    psi = psi_list(theta, xi, yi, nc, idmi)
    gv, ids = solve_exact(psi)

    if !is_sampled[i]
      diff[idmi[ids[1]]] += xi / 2
      diff[idmi[ids[2]]] += xi / 2
      diff[idmi[yi]] -= xi

      gv_sum += gv
    else
      ids_prev = IDS_storage[:,i]
      diff[idmi[ids_prev[1]]] -= xi / 2
      diff[idmi[ids_prev[2]]] -= xi / 2
      diff[idmi[ids[1]]] += xi / 2
      diff[idmi[ids[2]]] += xi / 2

      gv_sum += gv - GV_storage[i]
    end

    theta = (1 - step * lambda) * theta - (step / n_sampled) * diff

    IDS_storage[:,i] = ids
    GV_storage[i] = gv
    is_sampled[i] = true

    if iter % n == 0

      pass_iter += 1

      f = (gv_sum / n_sampled) + (lambda / 2 ) * dot(theta, theta)

      # if verbose println("pass iter : ", batch_iter, ", λ : ", lambda, ", f : ", f, ", abs diff : ", mean(abs(diff)), ", nobs : ", n_sampled, "/", n) end
      if verbose println("pass iter : ", pass_iter, ", λ : ", lambda, ", f : ", f, ", nobs : ", n_sampled, "/", n) end

      # discount step
      step = step * 0.95

      if pass_iter >= max_iter
        if verbose println("maximum iteration reached!!") end
        break
      end

      if mean(abs.(diff)) < grtol
        if verbose println("diff breaks!!") end
        break
      end

      if abs(f_prev - f) < ftol
        if verbose println("function breaks!!") end
        break
      end
      f_prev = f

    end

  end

  return ORAdvMCModel(theta, nc)

end

function predict_or_adv(model::ORAdvMCModel, X_test::Matrix)

  theta = model.theta
  nc = model.n_class
  n = size(X_test, 1)

  X1 = [ones(n) X_test]'   # transpose
  m = size(X1, 1)

  # prepare saved vars
  idmi = map(i -> idi(m, i), collect(1:nc))

  pred = zeros(Int64, n)
  for i = 1:n
    psi = psi_list(theta, X1[:,i], nc, idmi)
    pred[i] = indmax(psi)
  end

  return pred::Vector{Int64}
end

function test_or_adv(model::ORAdvMCModel, X_test::Matrix, y_test::Vector)

  pred = predict_or_adv(model, X_test)
  mae = mean(abs.(pred - y_test))

  return mae::Float64
end
