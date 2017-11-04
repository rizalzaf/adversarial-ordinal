## shared functions

## get featrure indeces for class i
function idi(m::Integer, i::Integer)
  return ((i-1)*m+1 : i*m)::UnitRange{Int64}
end

# calculate psi given parameter and data
function psi_list(theta::Vector, xi::Vector, yi::Integer, nc::Integer, idmi::Vector)
  psis = zeros(nc)
  for j=1:nc
    if j != yi
      v1 = dot(theta[idmi[j]], xi)
      v2 = dot(theta[idmi[yi]], -xi)
      psis[j] = v1 + v2
    end
  end

  return psis::Vector{Float64}
end

## no y. for prediction
function psi_list(theta::Vector, xi::Vector, nc::Integer, idmi::Vector)
  psis = zeros(nc)
  for j=1:nc
    val = dot(theta[idmi[j]], xi)
    psis[j] = val
  end

  return psis::Vector{Float64}
end
