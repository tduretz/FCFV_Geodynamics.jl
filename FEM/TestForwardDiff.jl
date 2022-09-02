using ForwardDiff

# ------------ Linear viscous with a standard function

function Stress!(τ,ε,η)
    τ .= 2.0*η.*ε
end

η = 1.0
τ = zeros(3)
ε = [1.0, 1.0, 1.0] 

S_closed = (τ,ε) -> Stress!(τ,ε,η)

grad = ForwardDiff.jacobian(S_closed, τ, ε)

# ------------ VEP

function Stress_VEP!(τip,εip,m,C,P,sinϕ,ηvp)
    τip              .= 2η.*(εip)
    τii               = sqrt((m.*τip)'*τip)   # don't worry, same as: τii = sqrt(0.5*(τip[1]^2 + τip[2]^2) + τip[3]^2)
    τy                = C + P*sinϕ         # need to add cosϕ to call it Drucker-Prager but this one follows Stokes2D_simpleVEP
    F                 = τii - τy  
    if (F>0.0) 
        λ             = F/(η + ηvp)
        τip          .= 2η.*(εip.-0.5*λ.*τip./τii)
        τii           = sqrt((m.*τip)'*τip)
        τy            = C + P*sinϕ + ηvp*λ
        F             = τii - τy
    end
end

η   = 10.0
ηvp = 0.
τip = zeros(3)
εip = [10, 1.0, 1.0] 
##
m = [0.5; 0.5; 1] 
εip_vp = zeros(3)
C = 1.
P = 0.
sinϕ = 0.0

Stress_VEP!(τip,εip,m,C,P,sinϕ,ηvp)

S_closed = (τ,ε) -> Stress_VEP!(τ,ε,m,C,P,sinϕ,ηvp)

Dvep = ForwardDiff.jacobian(S_closed, τ, ε)
