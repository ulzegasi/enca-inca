from julia_bootstrap import init_julia
init_julia()

from src.sdde_solar_dynamo_julia import sn, summary_statistics

theta = (15.0, 10.0, 20.0, 0.2, 1.0)

# realistic length
y = sn(theta, Twarmup=200, Tobs=929, seed=123)

print("len(y) =", len(y))
print("first 3 =", list(y[:3]))

ss = summary_statistics(y)

print("len(ss) =", len(ss))
print("first 3 ss =", list(ss[:3]))