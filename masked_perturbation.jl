# Masked perturbation family for sticker-like adversarial attacks.
# Perturbations are allowed only within a specified region (the "sticker").

"""
this code is based off MIPVerify's MaskedLInfNormBoundedPerturbationFamily code
"""

using MIPVerify
using JuMP

"""
sticker_mask(shape; row_range, col_range)

Create a boolean mask for the sticker region.
"""
function sticker_mask(shape; row_range, col_range)
    mask = zeros(Bool, shape)
    r1, r2 = row_range
    c1, c2 = col_range
    mask[1, r1:r2, c1:c2, :] .= true
    return mask
end

"""
Perturbation family that constrains perturbations to a masked region 
Outside the mask, perturbation is forced to zero. Inside the mask, each pixel can
vary by norm
"""
struct MaskedLInfNormBoundedPerturbationFamily <: MIPVerify.RestrictedPerturbationFamily
    norm_bound::Real
    mask::Array{Bool}

    function MaskedLInfNormBoundedPerturbationFamily(norm_bound::Real, mask::Array{Bool})
        @assert(norm_bound > 0, "Norm bound $(norm_bound) should be positive")
        return new(norm_bound, mask)
    end
end

Base.show(io::IO, pp::MaskedLInfNormBoundedPerturbationFamily) =
    print(io, "masked-linf-$(pp.norm_bound)-$(sum(pp.mask))px")

function MIPVerify.get_perturbation_specific_keys(
    nn::MIPVerify.NeuralNet,
    input::Array{<:Real},
    pp::MaskedLInfNormBoundedPerturbationFamily,
    m::JuMP.Model,
)::Dict{Symbol,Any}
    @assert(size(pp.mask) == size(input), "Mask size $(size(pp.mask)) must match input size $(size(input))")
    input_range = CartesianIndices(size(input))

    # v_e is the perturbation added
    v_e = map(input_range) do i
        if pp.mask[i]
            JuMP.@variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound)
        else
            JuMP.@variable(m, lower_bound = 0, upper_bound = 0)  # fixed at 0
        end
    end

   # v_x0 is the input with the perturbation added
    v_x0 = map(input_range) do i
        JuMP.@variable(
            m,
            lower_bound = pp.mask[i] ? max(0, input[i] - pp.norm_bound) : input[i],
            upper_bound = pp.mask[i] ? min(1, input[i] + pp.norm_bound) : input[i],
        )
    end
    JuMP.@constraint(m, v_x0 .== input .+ v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end
