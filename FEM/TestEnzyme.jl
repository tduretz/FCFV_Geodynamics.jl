using Enzyme

# f(x) = [x[1]*x[2], x[2]]

# grad = jacobian(Forward, f, [2.0, 3.0])

# ------------ Linear viscous with anonymous function

R(ε) = 2.0 .* ε


grad = jacobian(Forward, R, [2.0, 3.0, 1.0])

# # ------------ Linear viscous with a standard function

function Stress!(τ,ε,η)
    τ .= 2.0*η.*ε
end

η = 1.0
S = zeros(3)
ε = [1.0, 1.0, 1.0] 

# S_closed = (S,ε) -> Stress!(S,ε,η)


# grad = jacobian(Forward, S_closed, ε)
rad = jacobian(Forward, Stress!, Duplicated(ε))

