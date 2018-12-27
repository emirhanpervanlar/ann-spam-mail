import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import expit

momentum = 0.1
l_rate = 0.1
epoch = 1
e_tolerance = 0.1

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
rand_data = random.randint(0,4600)
input_data = np.load("nn_input2.npy")[rand_data]
target_data = np.load("nn_class2.npy")[rand_data]

input_weight = np.load("nn_weight/input_weight.npy")
h1_weight = np.load("nn_weight/h1_weight.npy")
h1_bias = np.load("nn_weight/h1_bias.npy")
output_bias = np.load("nn_weight/output_bias.npy")


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

h1_out = feedforward(input_data,input_weight,h1_bias[0])
n_out = feedforward(h1_out,h1_weight,output_bias[0])
n_err = er_rmse(target_data[0],n_out)

target_com = ""
if target_data == "0":
    target_com = "Spam değil"
elif target_data == "1":
    target_com = "Spam"

print("Tahmin Edilen Örnek Numarası : "+str(rand_data) + "  Örnek Durumu : " + target_com)

pre_com = ""
if n_out[0] == 0:
    pre_com = "Spam değil"
elif n_out[0] == 1:
    pre_com = "Spam"
else:
    pre_com = "Hata : " + str(n_err)

print("Tahmin Sonucu : " + pre_com)




