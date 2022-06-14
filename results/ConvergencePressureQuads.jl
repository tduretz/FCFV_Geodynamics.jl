using Plots

# Quads
# new = 0
q_old_pinf = [4.711814469750017,   3.350190842509175,   2.1167122161774152,  1.2136104628255033]
q_old_p1   = [   2.42e-01,            1.31e-01,            6.88e-02,            3.55e-02    ]  

# new = 1
q_new_pinf = [13.40444649162992,    19.401693695108044, 23.465937390070494,  24.58233052514887 ]
q_new_p1   = [  5.10e-01,             4.35e-01,           3.96e-01,            3.93e-01        ]

# Triangles
# new = 0
t_old_pinf = [9.109365242600466, 2.3965657501035658, 1.1986689745308254, 0.6932202993644836]
t_old_p1   = [7.70e-02,          3.47e-02,           1.33e-02,           6.58e-03]  

# new = 1
t_new_pinf = [9.146149772170457,  5.6298889773453595, 2.9452544057136514, 1.4064992174016986]
t_new_p1   = [ 2.18e-01,          1.08e-01,           4.79e-02,           2.15e-02]


n = [ 1.0, 2.0,  4.0,  8.0 ]

p1 = plot( log10.(n), log10.(q_old_p1), color=:black, label="old, L1", title="Quads")
p1 = plot!(log10.(n), log10.(q_new_p1), color=:blue, label="new, L1", ylabel="error", xlabel="1/h")
p1 = plot!(log10.(n), log10.(q_old_pinf), color=:green, label="old, Linf",  linsestyle=:dashdot)
p1 = plot!(log10.(n), log10.(q_new_pinf), color=:red, label="new, Linf", linsestyle=:dot)

p2 = plot( log10.(n), log10.(t_old_p1), color=:black, label="old, L1", title="Triangles")
p2 = plot!(log10.(n), log10.(t_new_p1), color=:blue, label="new, L1", ylabel="error", xlabel="1/h")
p2 = plot!(log10.(n), log10.(t_old_pinf), color=:green, label="old, Linf",  linsestyle=:dashdot)
p2 = plot!(log10.(n), log10.(t_new_pinf), color=:red, label="new, Linf", linsestyle=:dot)

display( plot(p1,p2) )

