import numpy as np
import matplotlib.pyplot as plt

acc_array = list()
acc_array.append(int(np.load("acc_test/hidden_5.npy")))
acc_array.append(int(np.load("acc_test/hidden_10.npy")))
acc_array.append(int(np.load("acc_test/hidden_15.npy")))
acc_array.append(int(np.load("acc_test/hidden_20.npy")))
acc_array.append(int(np.load("acc_test/hidden_25.npy")))
acc_array.append(int(np.load("acc_test/hidden_30.npy")))
acc_array.append(int(np.load("acc_test/hidden_35.npy")))
acc_array.append(int(np.load("acc_test/hidden_40.npy")))
acc_array.append(int(np.load("acc_test/hidden_45.npy")))
acc_array.append(int(np.load("acc_test/hidden_50.npy")))
acc_array.append(int(np.load("acc_test/hidden_60.npy")))
acc_array.append(int(np.load("acc_test/hidden_70.npy")))
acc_array.append(int(np.load("acc_test/hidden_80.npy")))
acc_array.append(int(np.load("acc_test/hidden_90.npy")))

x = [5,10,15,20,25,30,35,40,45,50,60,70,80,90]
plt.ylabel("Accuracy")
plt.xlabel("Hidden Layer")
plt.plot(x,acc_array)
plt.show()
print(acc_array)