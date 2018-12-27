import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import expit



load_input = np.load("first_nn_input.npy")[1000:2700]
load_class = np.load("first_nn_class.npy")[1000:2700]
print("Eğitim Veri Sayısı : " +str(len(load_input)))
momentum = 0.5
l_rate = 0.8
epoch = 2
e_tolerance = 0.1
print("Öğrenme Katsayısı : "+str(l_rate)+"  Momentum : "+str(momentum)+"  Hata Toleransı : "+str(e_tolerance)+"  Epoch : "+str(epoch))
# 100 input nöron / 2 hidden layer (50 nöron) / 3 output nöron
input_count = 57
hidden_count = 40
out_count = 1


#Sigmoid aktivasyon fonksiyonu
def act_sig(x):
    # x=x/200
    # print("x"+str(-1*x))
    # print("activasyon"+str(1/(1+np.exp(-1*x))))
    return 1/(1+np.exp(-1*x))


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



# EĞİTİM AĞI

#Ağın başlangıç değerlerinin atanması
input_data = load_input

target_data = load_class



#Ağırlıkları dışardan aldık




input_weight = list()
h1_weight = list()   #Hidden Layer 1
#Rasgele -1,1 aralığında ağrlık değerleri ataması
for y in range(input_count):
    input_weight.append([random.uniform(-1, 1) for i in range(hidden_count)])
for y in range(hidden_count):
    h1_weight.append([random.uniform(-1, 1) for i in range(hidden_count)])


#Rasgele -1,1 aralığında bias değerleri ataması
h1_bias = list()

output_bias = list()
h1_bias.append([random.uniform(-1, 1) for a in range(hidden_count)])
output_bias.append([random.uniform(-1, 1) for j in range(out_count)])
h1_bias_delta = np.zeros(hidden_count)
output_bias_delta = np.zeros(out_count)
input_weight_delta = np.zeros((input_count,hidden_count))
h1_weight_delta = np.zeros((hidden_count,hidden_count))

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

#Eğitim Aşaması

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
        # print("Hedef")
        # print(target_data[i])
        # print("Cıktı")
        # print(n_out)
        epoch_error.append(n_err)
        if(n_err<e_tolerance):
            e_true_count += 1
        e_pre_count += 1
        # print("İterasyon_ "+str(i)+"  Hata : "+str(n_err))
        if n_err > e_tolerance:
            out_err_unit = list()
            for a in range(len(n_out)):
                e_o_unit = float(n_out[a]) * (1 - n_out[a]) * (float(target_data[i][a]) - n_out[a])
                out_err_unit.append(e_o_unit)
                output_bias_delta[a] = (momentum * e_o_unit) + (l_rate * output_bias_delta[a])

            h1_err_unit = list()
            for c in range(len(h1_out)):
                h1_e_unit = 0
                for xc in range(len(n_out)):
                    h1_e_unit = h1_e_unit + (out_err_unit[xc] * h1_weight[c][xc])
                    h1_weight_delta[c][xc] = (momentum * out_err_unit[xc] * h1_out[c]) + (
                                l_rate * h1_weight_delta[c][xc])
                h1_err_unit.append(h1_out[c] * (1 - h1_out[c]) * h1_e_unit)
                h1_bias_delta[c] = (momentum * h1_e_unit) + (l_rate * h1_bias_delta[c])

            for d in range(len(input_data[i])):
                for xd in range(len(h1_out)):
                    input_weight_delta[d][xd] = (momentum * h1_err_unit[xd] * float(input_data[i][d])) + (
                            l_rate * input_weight_delta[d][xd])

            # Ağırlık güncelleme
            for a in range(len(h1_out)):
                for b in range(len(n_out)):
                    h1_weight[a][b] = h1_weight[a][b] + h1_weight_delta[a][b]
            for a in range(len(input_data[i])):
                for b in range(len(h1_out)):
                    input_weight[a][b] = input_weight[a][b] + input_weight_delta[a][b]

    true_count += e_true_count
    pre_count += e_pre_count
    acc_rate = int((true_count/pre_count)*100)
    ep_rate = int((e_true_count/e_pre_count)*100)
    print("Epoch True : "+str(e_true_count)+"   Epoch Rate : "+ str(ep_rate)+"%  Accuracy :  " + str(acc_rate)+"%")
    output_error.append(epoch_error)
    acc_array.append(acc_rate)




acc_save_array = np.array(acc_array[-1])
# np.save("acc_test/hidden_"+str(hidden_count),acc_save_array)
plt.plot(acc_array)
plt.ylabel("Accuracy")
plt.show()
# print(output_error)