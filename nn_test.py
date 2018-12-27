import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import expit

momentum = 0.1
l_rate = 0.1
epoch = 1
e_tolerance = 0.1
print("Öğrenme Katsayısı : "+str(l_rate)+"  Momentum : "+str(momentum)+"  Hata Toleransı : "+str(e_tolerance))
# 100 input nöron / 2 hidden layer (50 nöron) / 3 output nöron
input_count = 57
hidden_count = 40
out_count = 1


#Sigmoid aktivasyon fonksiyonu
def act_sig(x):
    act = 1 / (1 + np.exp(-1 * x))
    return act_step(act)

def act_step(x):
    if x < 0.35:
        return 0
    elif x > 0.65:
        return 1
    else:
        return x

# RMSE hata fonksionu
def er_rmse(target,output):
    error = 0
    count = len(target)
    for i in range(count):
        error = error + (float(target[i])-output[i])**2
    return math.sqrt((1/count)*error)
# MSE hata fonksionu
def er_mse(target,output):
    error = 0
    count = len(target)
    for i in range(count):
        error = error + (target[i]-output[i])**2
    return (1/count)*error

#Ağın başlangıç değerlerinin atanması

input_data = np.load("nn_input2.npy")[2700:-1]
target_data = np.load("nn_class2.npy")[2700:-1]
print("Test Veri Sayısı : "+str(len(input_data)))
input_weight = np.load("nn_weight/input_weight.npy")
h1_weight = np.load("nn_weight/h1_weight.npy")
h1_bias = np.load("nn_weight/h1_bias.npy")
output_bias = np.load("nn_weight/output_bias.npy")
target_1_count = 0
target_0_count = 0
for i in range(len(target_data)):
    if int(target_data[i]) == 1:
        target_1_count = target_1_count + 1
    elif int(target_data[i]) == 0:
        target_0_count = target_0_count + 1
print("Spam Olmayan Veri Sayısı : "+ str(target_0_count) +"   Spam Veri Sayısı : "+ str(target_1_count))


#İLERİ BESLEME
def feedforward(input,weight,bias):
    output = list()
    w_array = np.array(weight)
    for i in range(len(bias)):
        out = 0
        for y in range(len(input)):
            out = out + (float(input[y])*w_array[y,i])
        out = act_sig(out+bias[i])
        output.append(out)
    return output

#Test Aşaması

output_error = list()
acc_array = list()
true_count = 0
pre_count = 0

for j in range(epoch):
    e_true_count = 0
    e_pre_count = 0
    epoch_error = list()
    print("Epoch" + str(j))
    for i in range(len(input_data)):
        h1_out = feedforward(input_data[i],input_weight,h1_bias[0])
        n_out = feedforward(h1_out,h1_weight,output_bias[0])
        n_err = er_rmse(target_data[i],n_out)
        epoch_error.append(n_err)
        if(n_err<e_tolerance):
            e_true_count += 1
        e_pre_count += 1
    true_count += e_true_count
    pre_count += e_pre_count
    acc_rate = int((true_count/pre_count)*100)
    ep_rate = int((e_true_count/e_pre_count)*100)
    print("Epoch Rate : "+ str(ep_rate)+"%  Accuracy :  " + str(acc_rate)+"%")
    output_error.append(epoch_error)
    acc_array.append(acc_rate)

plt.plot(acc_array)
plt.ylabel("Accuracy")
plt.show()
