include("adv_or_th.jl")
include("util.jl")

verbose = false
### prepare data

dname = "diabetes"
D_all = readcsv("data-example/" * dname * ".csv")
id_train = readcsv("data-example/" * dname * ".train")
id_test = readcsv("data-example/" * dname * ".test")

id_train = round.(Int64, id_train)
id_test = round.(Int64, id_test)

println(dname)

### Cross Validation, using first split
## First stage

id_tr = vec(id_train[1,:])
id_ts = vec(id_test[1,:])
X_train = D_all[id_tr,1:end-1]
y_train = round.(Int, D_all[id_tr, end])

X_test = D_all[id_ts,1:end-1]
y_test = round.(Int, D_all[id_ts, end])

X_train, mean_vector, std_vector = standardize(X_train)
X_test = standardize(X_test, mean_vector, std_vector)

lambdas =  [2.0^-i for i=1:2:13]
nls = length(lambdas)

# fold
n_train = size(X_train, 1)
n_test = size(X_test, 1)
kf = 5

# k folds
folds = k_fold(n_train, kf)

loss_list = zeros(nls)

# The first stage of CV
idx = randperm(n_train)
X_train = X_train[idx,:]
y_train = y_train[idx]

for i = 1:nls

  println(i, " | Adversarial | lambda = ", string(lambdas[i]))

  losses = zeros(n_train)
  # k fold
  for j = 1:kf
    # prepare training and validation
    id_tr = vcat(folds[[1:j-1; j+1:end]]...)
    id_val = folds[j]

    X_tr = X_train[id_tr, :];  y_tr = y_train[id_tr]
    X_val = X_train[id_val, :];  y_val = y_train[id_val]

    print("    ",j, "-th fold : ")
    @time model = train_or_adv_th(X_tr, y_tr, lambdas[i], verbose=verbose)

    ls = test_or_adv_th(model, X_val, y_val)
    losses[id_val] = ls

  end

  loss_list[i] = mean(losses)

  println("loss : ", string(mean(losses)))
  println()

end

ind_max= indmin(loss_list)
L0 = lambdas[ind_max]
lambdas =  [L0*2.0^((i-4)/2) for i=1:7]
nls = length(lambdas)

println("stage 1 lambda : ", L0)

## Second stage
idx = randperm(n_train)
X_train = X_train[idx,:]
y_train = y_train[idx]

for i = 1:nls

  println(i, " | Adversarial | lambda = ", string(lambdas[i]))

  losses = zeros(n_train)
  # k fold
  for j = 1:kf
    # prepare training and validation
    id_tr = vcat(folds[[1:j-1; j+1:end]]...)
    id_val = folds[j]

    X_tr = X_train[id_tr, :];  y_tr = y_train[id_tr]
    X_val = X_train[id_val, :];  y_val = y_train[id_val]

    print("    ",j, "-th fold : ")
    @time model = train_or_adv_th(X_tr, y_tr, lambdas[i], verbose=verbose)
    
    ls = test_or_adv_th(model, X_val, y_val)
    losses[id_val] = ls

  end

  loss_list[i] = mean(losses)

  println("loss : ", string(mean(losses)))
  println()

end

ind_max= indmin(loss_list)
lambda_best = lambdas[ind_max]
# lambda_best = 1e-3

println("best lambda : ", lambda_best)
println("Start Evaluation")

### Evaluation

n_split = size(id_train, 1)

v_model = Vector{ClassificationModel}()
v_mae = zeros(n_split)

for i = 1:n_split
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
  model = train_or_adv_th(X_train, y_train, lambda_best, verbose=verbose)
  
  result = test_or_adv_th(model, X_test, y_test)
  
  v_mae[i] = result
  push!(v_model, model)

  println(result)
end

println()

println(dname)
println("adversarial mean mae : ", mean(v_mae))
println("adversarial std mae : ", std(v_mae))
