abstract type ClassificationModel end

struct ORAdvMCModel <: ClassificationModel
  theta::Vector{Float64}
  n_class::Integer
end

struct ORAdvTHModel <: ClassificationModel
  theta::Vector{Float64}
  n_class::Integer
end

struct KernelORAdvModel <: ClassificationModel
  kernel::Symbol
  kernel_params::Vector{Float64}
  feature::Symbol
  alpha::Vector{Float64}
  n_class::Integer
  layout::Vector{Tuple{Int64,Int64,Int64}}
end
