function approx_grad(func::Function, x::Number, h::Number=1e-4)
  return (func(x+h) - func(x)) / h
end

function approx_grad(func::Function, x::Vector, h::Number=1e-4)
  ag = zeros(x)
  fx = func(x)
  for i = 1:length(ag)
    xp = copy(x)
    xp[i] += h
    ag[i] = (func(xp) - fx) / h
  end
  return ag
end

function standardize(data::Matrix)
  m, n = size(data)
  standardized = zeros(m, n)
  mean_vector = zeros(n)
  std_vector = zeros(n)
  for i = 1:n
    mean_vector[i] = mean(data[:,i])
    std_vector[i] = std(data[:,i])
    if std_vector[i] != 0
      standardized[:,i] = (data[:,i] - mean_vector[i]) ./ std_vector[i]
    elseif mean_vector[i] < 1 && mean_vector[i] >= 0
      standardized[:,i] = data[:,i]
    else
      standardized[:,i] = 1
    end
  end
  return standardized::Matrix{Float64}, mean_vector::Vector{Float64}, std_vector::Vector{Float64}
end

function standardize(data::Matrix, mean_vector::Vector{Float64}, std_vector::Vector{Float64})
  m, n = size(data)
  standardized = zeros(m, n)
  for i = 1:n
    if std_vector[i] != 0
      standardized[:,i] = (data[:,i] - mean_vector[i]) ./ std_vector[i]
    elseif mean_vector[i] < 1 && mean_vector[i] >= 0
      standardized[:,i] = data[:,i]
    else
      standardized[:,i] = 1
    end
  end
  return standardized
end


# function normalize(X::Matrix)
#   # normalize to 0 1
#   r_nrm = 1.0   # range
#   shift = 0.0
#   X_max = maximum(X, 1)
#   X_min = minimum(X, 1)
#   X_nrm = (r_nrm * broadcast(-, X, X_min) ./ broadcast(-, X_max, X_min)) + shift
#
#   return X_nrm
# end

function k_fold(n::Int, k::Int)
  idx = randperm(n)

  # allocate folds
  folds = Vector[]
  n_f = round(Int, floor(n/k))
  add_f = n % k
  j = 1
  for i=1:k
    if i <= add_f
      push!(folds, idx[j:j+n_f])
      j += n_f+1
    else
      push!(folds, idx[j:j+n_f-1])
      j += n_f
    end
  end

  return folds::Vector{Vector}
end
