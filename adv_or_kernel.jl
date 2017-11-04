## Kernelized Adversarial Ordinal Regression

include("types.jl")
include("kernel.jl")

function train_adv_or_kernel(X::Matrix, y::Vector, C::Real=1.0,
  kernel::Symbol=:linear, kernel_params::Vector=[], feature::Symbol=:mc;
  perturb::Real=0.0, tol::Real=1e-6, psdtol::Real=1e-6,
  log::Real=0, n_thread::Int=0, verbose::Bool=true)

  n = length(y)
  # add one
  X1 = [ones(n) X]'   # transpose
  m = size(X1, 1)

  # number of class
  nc = maximum(y)
  
  # dual parameter
  nd = n * nc * 2   # number of dual parameter
  alpha = zeros(nd)

  if verbose
    println("Start >> prepare variables")
    tic()
  end

  # kernel
  kernel_func = linear_kernel
  if kernel == :gaussian
    kernel_func = gaussian_kernel
  elseif kernel == :polynomial
    kernel_func = polynomial_kernel
  end

  # precompute kernels
  K = zeros(n,n)
  for i=1:n, j=i:n
    K[i,j] = kernel_func(X1[:,i], X1[:,j], kernel_params...)
    if (i != j); K[j,i] = K[i,j] end
  end
  
  # params for QP  
  Q = zeros(nd,nd)
  nu = zeros(nd)
  A = spzeros(2n,nd)        # sparse matrix
  b = ones(2n) * (C/2)

  # params layout
  # [i1j1a1, i2j1a1, ..., inj1a1, i1j2a1, ..., injka1, i1j1a1 ..., injka2] 
  ly = Vector{Tuple{Int64,Int64,Int64}}(nd)
  l = 1
  for k=1:2, j=1:nc, i=1:n
    ly[l] = (i,j,k)
    l += 1
  end

  # prepare nu (linear coefficient)
  l = 1
  for k=1:2, j=1:nc
    s = (k == 1) ? 1 : -1
    nu[l:l+n-1] = s * j * ones(n)
    l += n
  end

  if feature == :th
    # Quadratic coefficient
    for i=1:nd, j=i:nd
      ii = ly[i][1]; c_ii = ly[i][2]
      jj = ly[j][1]; c_jj = ly[j][2]

      Q[i,j] = (c_ii - y[ii]) * (c_jj - y[jj]) * K[ii,jj]
      th_ii = zeros(nc-1)            
      if c_ii < y[ii]
        th_ii[c_ii:y[ii]-1] = 1.
      elseif c_ii > y[ii]
        th_ii[y[ii]:c_ii-1] = -1.
      end
      th_jj = zeros(nc-1)   
      if c_jj < y[jj]
        th_jj[c_jj:y[jj]-1] = 1.
      elseif c_jj > y[jj]
        th_jj[y[jj]:c_jj-1] = -1.
      end
      Q[i,j] += dot(th_ii, th_jj)
      
      if (i != j); Q[j,i] = Q[i,j] end
    end
    # println("sym : ", issymmetric(Q))
  else
    # Quadratic coefficient
    for i=1:nd, j=i:nd
      ii = ly[i][1]; c_ii = ly[i][2]
      jj = ly[j][1]; c_jj = ly[j][2]

      # use kernel 
      if (c_ii == c_jj) && (y[ii] == y[jj]) && !(c_ii == y[ii]) && !(c_jj == y[jj])
        Q[i,j] = 2K[ii,jj]
      elseif (c_ii == y[jj]) && (y[ii] == c_jj) && !(c_ii == y[ii]) && !(c_jj == y[jj])
        Q[i,j] = -2K[ii,jj]
      elseif (c_ii == c_jj) && !(c_ii == y[ii]) && !(c_jj == y[jj])
        Q[i,j] = K[ii,jj]
      elseif (y[ii] == y[jj]) && !(c_ii == y[ii]) && !(c_jj == y[jj])
        Q[i,j] = K[ii,jj]
      elseif (c_ii == y[jj]) && !(c_ii == y[ii]) && !(c_jj == y[jj])
        Q[i,j] = -K[ii,jj]
      elseif (y[ii] == c_jj) && !(c_ii == y[ii]) && !(c_jj == y[jj])
        Q[i,j] = -K[ii,jj]
      end

      if (i != j); Q[j,i] = Q[i,j] end
    end
  end
  
  ## add perturbation
  for i=1:nd
    Q[i,i] = Q[i,i] + perturb
  end

  # prepare A
  for i=1:nd
    ii = ly[i][1]
    if ly[i][3] == 1
      A[ii, i] = 1.    
    elseif ly[i][3] == 2
      A[ii+n, i] = 1.
    end    
  end

  if verbose
    toc()
    tic()
  end

  if verbose println(">> Optim :: Gurobi") end

  # gurobi solver
  # gurobi environtment
  env = Gurobi.Env()
  # Method : 0=primal simplex, 1=dual simplex, 2=barrier  ; default for QP: barrier
  # Threads : default = 0 (use all threads)
  setparams!(env, PSDTol=psdtol, LogToConsole=log, Method=2, Threads=n_thread)
  
  ## init model
  model = gurobi_model(env,
              sense = :minimize,
              H = Q,
              f = -nu,
              Aeq = A,
              beq = b,
              lb = zeros(nd)
              )
  # Print the model to check correctness
  # print(model)

  # Solve with Gurobi
  Gurobi.optimize(model)

  if verbose
    toc()
    println("<< End QP")
  end

  # get solution
  alpha = get_solution(model)
  
  return KernelORAdvModel(kernel, kernel_params, feature, alpha, nc, ly)
end

function predict_or_adv_kernel(model::KernelORAdvModel, X_test::Matrix, X_train::Matrix, y_train::Vector)

  alpha = model.alpha
  nc = model.n_class
  ly = model.layout
  feature = model.feature
  n = size(X_test, 1)

  nd = length(alpha)

  X1 = [ones(n) X_test]'   # transpose
  m = size(X1, 1)

  # training data
  n_tr = size(X_train, 1)
  X1_tr = [ones(n_tr) X_train]'   # transpose

  # kernel
  kernel = model.kernel
  kernel_params = model.kernel_params
  # kernel function
  kernel_func = linear_kernel
  if kernel == :gaussian
    kernel_func = gaussian_kernel
  elseif kernel == :polynomial
    kernel_func = polynomial_kernel
  end

  # compute Kernel
  K = [ kernel_func(X1_tr[:,i], X1[:,j], kernel_params...)::Float64 for i=1:n_tr, j=1:n]

  pred = zeros(n)
  for i=1:n
    fs = zeros(nc)

    if feature == :th
      for j = 1:nd
        ii = ly[j][1]
        c_ii = ly[j][2]
        th_ii = zeros(nc-1)            
        if c_ii < y_train[ii]
          th_ii[c_ii : y_train[ii]-1] = 1.
        elseif c_ii > y_train[ii]
          th_ii[y_train[ii] : c_ii-1] = -1.
        end
        for l = 1:nc          
          fs[l] -= alpha[j] *  ( (c_ii - y_train[ii]) * l * K[ii,i] + sum(th_ii[l:end]) )
        end        
      end
    else
      for j = 1:nd
        mult = zeros(nc)
        ii = ly[j][1]
        c_ii = ly[j][2]
        mult[c_ii] += 1.
        mult[y_train[ii]] -= 1.
        
        fs -= alpha[j] * (mult * K[ii,i])
      end
    end
    
    pred[i] = indmax(fs)
  end

  return pred::Vector{Float64}
end

function test_or_adv_kernel(model::KernelORAdvModel, X_test::Matrix, y_test::Vector, X_train::Matrix, y_train::Vector)
  n = size(X_test, 1)

  pred = predict_or_adv_kernel(model, X_test, X_train, y_train)
  mae = mean(abs.(pred - y_test))

  return mae::Float64

end
