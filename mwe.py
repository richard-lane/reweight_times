import pickle
import matplotlib.pyplot as plt
import numpy as np
from hep_ml.reweight import GBReweighter
from sklearn.model_selection import train_test_split

# Read the decay times from the LHCb simulation - I've serialised it here
print("reading pickle")
with open("mc_times.pickle", "rb") as f:
    mc_times = pickle.load(f)

# Generate some random numbers from an exponential distribution with the right decay constant
d_lifetime_ps = 0.49
N = len(mc_times)
print("gen times")
exp_times = np.random.exponential(d_lifetime_ps, N)

mc_train, mc_test, model_train, model_test = train_test_split(mc_times, exp_times)

bdt = GBReweighter()
print("Training bdt")
bdt.fit(original=model_train, target=mc_train)
weights = bdt.predict_weights(model_test)

kw = {"bins": np.linspace(0.0, 9.0, 100), "alpha": 0.3, "density": True}
plt.figure(figsize=(12.0, 9.0))

plt.hist(mc_test, label="Original", **kw)
plt.hist(model_test, label="Target", **kw)
plt.hist(model_test, label="Target Weighted", weights=weights, **kw)
plt.legend()

plt.xlabel("Time /ps")
plt.ylabel("Counts")
plt.savefig("mwe.png")