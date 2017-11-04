@everywhere using Gurobi

@everywhere include("adv_or_kernel.jl")
@everywhere include("util.jl")

# subnormal number
set_zero_subnormals(true)

## set
perturb = 0.
log = 0
psdtol = 1e-6
verbose = false

feature = :th   # thresholded feature
# feature = :mc     # multiclass feature

### prepare data

dname = "diabetes"
D_all = readcsv("data-example/" * dname * ".csv")
id_train = readcsv("data-example/" * dname * ".train")
id_test = readcsv("data-example/" * dname * ".test")

id_train = round.(Int64, id_train)
id_test = round.(Int64, id_test)

println(dname)

### Cross Validation, using first split
function cross_validate(X_train::Matrix, y_train::Vector, kf::Integer, pars::Vector)
  n_train = length(y_train)
  npar = length(pars)
  
  # k folds
  folds = k_fold(n_train, kf)
  loss_list = SharedArray{Float64}(npar)

  @sync @parallel for i = 1:npar

    C = pars[i][1]
    gamma = pars[i][2]

    println(i, " | Adversarial | C = ", pars[i][1], ", Gamma = ", pars[i][2])

    losses = zeros(n_train)
    # k fold
    for j = 1:kf
      # prepare training and validation
      id_tr = vcat(folds[[1:j-1; j+1:end]]...)
      id_val = folds[j]

      X_tr = X_train[id_tr, :];  y_tr = y_train[id_tr]
      X_val = X_train[id_val, :];  y_val = y_train[id_val]

      print("    ",j, "-th fold : ")
      @time model = train_adv_or_kernel(X_tr, y_tr, C, :gaussian, [gamma], feature, perturb=perturb, log=log, psdtol=psdtol, verbose=verbose)
      
      mae = test_or_adv_kernel(model, X_val, y_val, X_tr, y_tr)
      losses[id_val] = mae      
    end

    loss_list[i] = mean(losses)    
    # println("loss : ", string(mean(losses)))    
  end

  return indmin(loss_list)
end

## First stage
id_tr = vec(id_train[1,:])
id_ts = vec(id_test[1,:])
X_train = D_all[id_tr,1:end-1]
y_train = round.(Int, D_all[id_tr, end])

X_test = D_all[id_ts,1:end-1]
y_test = round.(Int, D_all[id_ts, end])

X_train, mean_vector, std_vector = standardize(X_train)
X_test = standardize(X_test, mean_vector, std_vector)

Cs =  [2.0^i for i=0:3:12]
Gs =  [2.0^(-12+i) for i=0:3:12]

ncs = length(Cs)
Pars = [ Tuple{Float64,Float64}((Cs[i], Gs[j])) for i=1:ncs, j=1:ncs ]
pars = vec(Pars)

# fold
n_train = size(X_train, 1)
n_test = size(X_test, 1)
kf = 5

idx = randperm(n_train)
X_train = X_train[idx,:]
y_train = y_train[idx]

println("First CV")
id_best = cross_validate(X_train, y_train, kf, pars)
C0 = pars[id_best][1]
G0 = pars[id_best][2]

println("best C : ", C0, "  |  best G : ", G0)

Cs =  [C0*2.0^(i-3) for i=1:5]
Gs =  [G0*2.0^(i-3) for i=1:5]

ncs = length(Cs)

Pars = [ Tuple{Float64,Float64}((Cs[i], Gs[j])) for i=1:ncs, j=1:ncs ]
pars = vec(Pars)

## Second stage
idx = randperm(n_train)
X_train = X_train[idx,:]
y_train = y_train[idx]

println("Second CV")
id_best = cross_validate(X_train, y_train, kf, pars)
C_best = Pars[id_best][1]
G_best = Pars[id_best][2]

println("best C : ", C_best, "  |  best G : ", G_best)

### Evaluation

n_split = size(id_train, 1)

v_mae = SharedArray{Float64}(n_split)

println("Evaluation")

@sync @parallel for i = 1:n_split
  # standardize
  id_tr = vec(id_train[i,:])
  id_ts = vec(id_test[i,:])
  X_train = D_all[id_tr,1:end-1]
  y_train = round.(Int, D_all[id_tr, end])

  X_test = D_all[id_ts,1:end-1]
  y_test = round.(Int, D_all[id_ts, end])

  X_train, mean_vector, std_vector = standardize(X_train)
  X_test = standardize(X_test, mean_vector, std_vector)

  #train and test
  model = train_adv_or_kernel(X_train, y_train, C_best, :gaussian, [G_best], feature, perturb=perturb, log=log, psdtol=psdtol, verbose=verbose)
  
  mae = test_or_adv_kernel(model, X_test, y_test, X_train, y_train)
  println(mae)

  v_mae[i] = mae  
end

println(dname)
println("mean mae : ", mean(v_mae))
println("std mae : ", std(v_mae))
println("mae list : ")
for i = 1:n_split
  println(v_mae[i])
end