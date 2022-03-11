import numpy as np
import matplotlib.pyplot as plt

pos_ary = np.load("pos_ary.npy")
f_ary   = np.load("f_ary.npy")
NDCWPA_f  = np.load("NDCWPA_f.npy")
NDCWPA_K  = np.load("NDCWPA_K.npy")
RandAcc_f = np.load("RandAcc_f.npy")
RandAcc_K = np.load("RandAcc_K.npy")

# plt.plot(pos_ary, f_ary, label="gt force")
plt.plot(pos_ary, NDCWPA_f-f_ary, label="CWPA force error")
plt.plot(pos_ary, RandAcc_f-f_ary, label="DWNA force error")
plt.xlabel("position(m)")
plt.ylabel("force(m)")
plt.legend()
plt.savefig("compare_f_err.png")

plt.clf()
plt.plot(pos_ary, NDCWPA_K, label="CWPA stiffness")
plt.plot(pos_ary, RandAcc_K, label="DWNA stiffness")
plt.xlabel("position(m)")
plt.ylabel("stiffness(N/m)")
plt.savefig("compare_K.png")