## Adversarial Ordinal Regression with Thresholded Features

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

function compute_phi(xi::Vector, nf::Integer, nc::Integer)
  m = length(xi)
  phi = zeros(nf, nc)

  for j = 1:nc
    phi[1:m,j] = j * xi
    if j < nc
      phi[m+j:end, j] = 1.
    end
  end

  return phi
end

function compute_psi(xi::Vector, theta::Vector, yi::Integer, nc::Integer)
  m = length(xi)
  psi = zeros(nc)

  w = theta[1:m]
  wxi = dot(w, xi)

  for j = 1:nc
    psi[j] = (j - yi) * wxi
  end

  tmp = 0.
  if yi < nc
    tmp -= sum(theta[m+yi:end])
  end
  psi[nc] += tmp
  for j = nc-1:-1:1
    tmp += theta[m+j]
    psi[j] += tmp
  end
  
  return psi
end

# no yi
function compute_psi(xi::Vector, theta::Vector, nc::Integer)
  m = length(xi)
  psi = zeros(nc)

  w = theta[1:m]
  wxi = dot(w, xi)

  for j = 1:nc
    psi[j] = j * wxi
  end

  tmp = 0.
  for j = nc-1:-1:1
    tmp += theta[m+j]
    psi[j] += tmp
  end
  
  return psi
end

# train ordinal regressin adversarial
function train_or_adv_th(X::Matrix, y::Vector, lambda::Float64=0.0;
  step::Real=0.1, ftol::Real=1e-8, grtol::Real=1e-8, show_trace::Bool=true, max_iter::Int=1000, verbose::Bool=true)

  n = length(y)
  # add one
  X1 = [ones(n) X]'   # transpose
  m = size(X1, 1)

  # number of class
  nc = maximum(y)
  nf = m + nc - 1   # number of features

  # parameters. init with zero
  w = rand(m) - 0.5
  c = rand(nc - 1)
  theta = [w; c]

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

    psi = compute_psi(xi, theta, yi, nc)
    gv, ids = solve_exact(psi)

    if !is_sampled[i]
      diff[1:m] = ((ids[1] + ids[2]) / 2 - yi) * xi

      diff[m+ids[1]:end] += 0.5
      diff[m+ids[2]:end] += 0.5     
      diff[m+yi:end] -= 1. 

      gv_sum += gv
    else
      ids_prev = IDS_storage[:,i]

      diff[1:m] += ((ids[1] + ids[2]) / 2 - (ids_prev[1] + ids_prev[2]) / 2) * xi

      diff[m+ids_prev[1]:end] -= 0.5
      diff[m+ids_prev[2]:end] -= 0.5     
      diff[m+ids[1]:end] += 0.5
      diff[m+ids[2]:end] += 0.5
      
      gv_sum += gv - GV_storage[i]
    end

    theta[1:m] = (1 - step * lambda) * theta[1:m] - (step / n_sampled) * diff[1:m]
    theta[m+1:end] -= (step / n_sampled) * diff[m+1:end]

    IDS_storage[:,i] = ids
    GV_storage[i] = gv
    is_sampled[i] = true

    if iter % n == 0

      pass_iter += 1

      f = (gv_sum / n_sampled) + (lambda / 2 ) * dot(theta[1:m], theta[1:m])

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

  return ORAdvTHModel(theta, nc)

end


function predict_or_adv_th(model::ORAdvTHModel, X_test::Matrix)

  theta = model.theta
  nc = model.n_class
  nf = length(theta)
  n = size(X_test, 1)

  X1 = [ones(n) X_test]'   # transpose
  m = size(X1, 1)

  pred = zeros(Int64, n)
  for i = 1:n
    xi = X1[:,i]
    psi = compute_psi(xi, theta, nc)
    pred[i] = indmax(psi)
  end

  return pred::Vector{Int64}
end

function test_or_adv_th(model::ORAdvTHModel, X_test::Matrix, y_test::Vector)

  pred = predict_or_adv_th(model, X_test)
  mae = mean(abs.(pred - y_test))

  return mae::Float64
end