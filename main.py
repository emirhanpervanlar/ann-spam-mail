from tkinter import *
from tkinter import messagebox
from tkintertable import TableCanvas, TableModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import math
import random


def act_sig(x):
    act = 1/(1+np.exp(-1*x))
    return act_step(act)

def act_step(x):
    if x < float(e_step_t.get()):
        return 0
    elif x > (1-float(e_step_t.get())):
        return 1
    else:
        return x

def er_rmse(target, output):
    error = 0
    count = len(target)
    for i in range(count):
        error = error + (float(target[i]) - output[i]) ** 2
    return np.sqrt((1 / count) * error)


def feedforward(input, weight, bias):
    output = list()
    w_array = np.array(weight)
    for i in range(len(bias)):
        out = 0
        for y in range(len(input)):
            out = out + (float(input[y]) * w_array[y, i])
        out = act_sig(out + bias[i])
        output.append(out)
    return output

def nn_edu():
    epoch = e_count.get()
    momentum = float(m_m.get())
    l_rate = float(l_r.get())
    e_tolerance = float(e_t.get())
    print(hidden_c.get())
    if epoch <= 0:
        messagebox.showinfo("Hata", "Epoch değeri hatalı girildi")
    elif momentum < 0:
        messagebox.showinfo("Hata", "Momentum değeri hatalı girildi.0-1 arası değer giriniz.")
    elif l_rate < 0:
        messagebox.showinfo("Hata", "Öğrenme Katsayısı hatalı girildi.0-1 arası değer giriniz.")
    elif e_tolerance < 0:
        messagebox.showinfo("Hata", "Hata toleransı değeri hatalı girildi.0-1 arası değer giriniz.")
    elif edu_count.get() <= 0:
        messagebox.showinfo("Hata", "Eğitim Örnek Sayısı değeri hatalı girildi.0'dan büyük değer giriniz.")
    elif int(hidden_c.get()) <= 0:
        messagebox.showinfo("Hata", "Ara Katman Nöron Sayısı değeri hatalı girildi.0'dan büyük değer giriniz.")
    elif float(e_step_t.get()) < 0:
        messagebox.showinfo("Hata", "Basamak Değeri hatalı girildi.0'dan büyük değer giriniz.")
    else:
        load_data = np.load("nn_input2.npy")[0:edu_count.get()]
        class_data = np.load("nn_class2.npy")[0:edu_count.get()]
        input_count = len(load_data[0])
        hidden_count = 40
        out_count = 1
        input_data = load_data
        target_data = class_data
        if rand_weight.get() == 0:
            hidden_count = int(hidden_c.get())
            #Ağırlıkları rasgele verdik
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
        else:
            input_weight = np.load("nn_weight/input_weight.npy")
            h1_weight = np.load("nn_weight/h1_weight.npy")
            h1_bias = np.load("nn_weight/h1_bias.npy")
            output_bias = np.load("nn_weight/output_bias.npy")
        target_1_count = 0
        target_0_count = 0
        for i in range(len(class_data)):
            if int(class_data[i]) == 1:
                target_1_count = target_1_count + 1
            elif int(class_data[i]) == 0:
                target_0_count = target_0_count + 1
        print("Eğitim Veri Sayısı :  " +str(len(load_data)))
        print("Spam Olmayan Veri Sayısı : " + str(target_0_count) + "   Spam Veri Sayısı : " + str(target_1_count))
        print("Öğrenme Katsayısı : " + str(l_rate) + "  Momentum : " + str(momentum) + "  Hata Toleransı : " + str(e_tolerance) + "  Epoch : " + str(epoch))

        input_weight_delta = np.zeros((input_count, hidden_count))
        h1_weight_delta = np.zeros((hidden_count, hidden_count))
        h1_bias_delta = np.zeros(hidden_count)
        output_bias_delta = np.zeros(out_count)
        output_error = list()
        acc_array = list()
        true_count = 0
        pre_count = 0
        plt.show(block=False)

        for j in range(epoch):
            # Her epochta örnekleri karıştırıyoruz
            if h_u_data.get() == 1:
                s = np.arange(load_data.shape[0])
                np.random.shuffle(s)
                input_data = load_data[s]
                target_data = class_data[s]
            e_true_count = 0
            e_pre_count = 0
            epoch_error = list()
            print("Epoch" + str(j))
            for i in range(len(input_data)):
                h1_out = feedforward(input_data[i], input_weight, h1_bias[0])
                n_out = feedforward(h1_out, h1_weight, output_bias[0])
                n_err = er_rmse(target_data[i], n_out)
                epoch_error.append(n_err)
                # GERİ BESLEME
                if (n_err < e_tolerance):
                    e_true_count += 1
                e_pre_count += 1
                if n_err > e_tolerance:
                    out_err_unit = list()
                    for a in range(len(n_out)):
                        e_o_unit = float(n_out[a]) * (1 - n_out[a]) * (float(target_data[i][a]) - n_out[a])
                        out_err_unit.append(e_o_unit)
                        output_bias_delta[a] = (momentum * e_o_unit) + (l_rate * output_bias_delta[a])
                        output_bias[0][a] = output_bias[0][a] + output_bias_delta[a]
                    h1_err_unit = list()
                    for c in range(len(h1_out)):
                        h1_e_unit = 0
                        for xc in range(len(n_out)):
                            h1_e_unit = h1_e_unit + (out_err_unit[xc] * h1_weight[c][xc])
                            h1_weight_delta[c][xc] = (momentum * out_err_unit[xc] * h1_out[c]) + (
                                    l_rate * h1_weight_delta[c][xc])
                        h1_err_unit.append(h1_out[c] * (1 - h1_out[c]) * h1_e_unit)
                        h1_bias_delta[c] = (momentum * h1_e_unit) + (l_rate * h1_bias_delta[c])
                        h1_bias[0][c] = h1_bias[0][c] + h1_bias_delta[c]

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

            # Accuracy Hesaplama
            true_count += e_true_count
            pre_count += e_pre_count
            acc_rate = int((true_count / pre_count) * 100)
            ep_rate = int((e_true_count / e_pre_count) * 100)
            print("Epoch True : " + str(e_true_count) + "   Epoch Rate : " + str(ep_rate) + "%  Accuracy :  " + str(
                acc_rate) + "%")
            output_error.append(epoch_error)
            acc_array.append(acc_rate)


            plt.title("Eğitim Sonuçları")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            ax1.plot(acc_array)
            plt.pause(0.001)

            # Ağırlıkların kaydedilmesi
            # np.save("nn_weight/h1_weight", h1_weight)
            # np.save("nn_weight/input_weight", input_weight)
            # np.save("nn_weight/h1_bias", h1_bias)
            # np.save("nn_weight/output_bias", output_bias)

            # Doğruluk oranı kaydı ve grafik olarak gösterme

        # plt.plot(acc_array)
        # plt.title("Eğitim Sonuçları")
        # plt.ylabel("Accuracy")
        # plt.xlabel("Epoch")
        # plt.show()

def nn_test():

    e_tolerance = float(e_t.get())
    input_data = np.load("nn_input2.npy")[edu_count.get():-1]
    target_data = np.load("nn_class2.npy")[edu_count.get():-1]
    if e_tolerance < 0:
        messagebox.showinfo("Hata", "Hata toleransı değeri hatalı girildi.0-1 arası değer giriniz.")
    else:
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
        print("Eğitim Veri Sayısı :  " + str(len(target_data)))
        print("Spam Olmayan Veri Sayısı : " + str(target_0_count) + "   Spam Veri Sayısı : " + str(target_1_count))
        print("Hata Toleransı : " + str( e_tolerance))

        output_error = list()
        acc_array = list()
        true_count = 0
        pre_count = 0
        e_true_count = 0
        e_pre_count = 0
        epoch_error = list()
        for i in range(len(input_data)):
            h1_out = feedforward(input_data[i], input_weight, h1_bias[0])
            n_out = feedforward(h1_out, h1_weight, output_bias[0])
            n_err = er_rmse(target_data[i], n_out)
            epoch_error.append(n_err)
            if (n_err < e_tolerance):
                e_true_count += 1
            e_pre_count += 1
        true_count += e_true_count
        pre_count += e_pre_count
        acc_rate = int((true_count / pre_count) * 100)
        ep_rate = int((e_true_count / e_pre_count) * 100)
        print("Epoch Rate : " + str(ep_rate) + "%  Accuracy :  " + str(acc_rate) + "%")
        messagebox.showinfo("Test Sonuçları", " %s örnek üzerinde uygulanan test sonucunda %s  doğruluk oranı elde edilmiştir."%(str(len(target_data)),str(acc_rate)) )

def nn_pred():
    e_tolerance = float(e_t.get())
    rand_data = random.randint(0, 4600)
    input_data = np.load("nn_input2.npy")[rand_data]
    target_data = np.load("nn_class2.npy")[rand_data]


    input_weight = np.load("nn_weight/input_weight.npy")
    h1_weight = np.load("nn_weight/h1_weight.npy")
    h1_bias = np.load("nn_weight/h1_bias.npy")
    output_bias = np.load("nn_weight/output_bias.npy")
    h1_out = feedforward(input_data, input_weight, h1_bias[0])
    n_out = feedforward(h1_out, h1_weight, output_bias[0])
    n_err = er_rmse(target_data[0], n_out)
    target_com = ""
    if target_data == "0":
        target_com = "Spam değil"
    elif target_data == "1":
        target_com = "Spam"

    print("Tahmin Edilen Örnek Numarası : " + str(rand_data) + "  Örnek Durumu : " + target_com)

    pre_com = ""
    if n_err < e_tolerance:
        if n_out[0] == 0:
            pre_com = "Spam değil"
        elif n_out[0] == 1:
            pre_com = "Spam"
    else:
        pre_com = "Hata : " + str(n_err)

    print("Tahmin Sonucu : " + pre_com)
    if target_com == pre_com:
        m_t = "DOĞRUDUR"
        print("Tahmin Doğru")
    else:
        m_t = "YANLIŞTIR"
        print("Tahmin Yanlış")
    messagebox.showinfo("Tahmin Sonuçları",
                        " %s.örnek üzerinde uygulanan tahmin sonucunda '%s' elde edilmiştir ve tahmin %s." % (
                        str(rand_data),pre_com,m_t))


def draw_neural_net():
    fig = plt.figure(figsize=(18, 9))
    ax = fig.gca()
    ax.axis('off')
    left = .1
    right = .9
    bottom = 0
    top = 1
    layer_sizes = [57, 40, 1]
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
    plt.show()

def draw_h_layer():
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

    x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]
    plt.ylabel("Accuracy")
    plt.xlabel("Hidden Layer")
    plt.plot(x, acc_array)
    plt.show()

if __name__ == "__main__":


    master = Tk()
    master.title("YSA Spam Mail Projesi")
    master.geometry("500x680")

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    lb5 = Label(master, text="Toplam Örnek Sayısı : 4601")
    lb_edu = Label(master, text="Eğitim Örnek Sayısı")
    edu_count = IntVar(value=2700)
    edu_c = Entry(master, textvariable=edu_count)

    h_u_data = IntVar()
    u_data = Checkbutton(master, text="Verileri Karışık Kullan",variable=h_u_data)

    rand_weight = IntVar()
    r_w = Checkbutton(master, text="Eğitimde Kayıtlı Ağırlıkları Kullan", variable=rand_weight)

    lb_hidden = Label(master, text="Arka Katman Nöron Sayısı")
    hi_count = IntVar(value=40)
    hidden_c = Entry(master, textvariable=hi_count)

    lb1 = Label(master,text="Epoch Sayısı")
    e_count = IntVar(value=10)
    e_c = Entry(master,textvariable=e_count)

    lb2 = Label(master, text="Öğrenme Katsayısı")
    l_rate = IntVar(value=0.1)
    l_r = Entry(master, textvariable=l_rate)

    lb3 = Label(master, text="Momentum Katsayısı")
    mome = IntVar(value=0.1)
    m_m = Entry(master, textvariable=mome)

    lb4 = Label(master, text="Hata Toleransı")
    error_t = IntVar(value=0.05)
    e_t = Entry(master, textvariable=error_t)

    lb_step = Label(master, text="Basamak Değeri")
    step_t = IntVar(value=0.35)
    e_step_t = Entry(master, textvariable=step_t)

    education_btn = Button(text="Ağı Eğit", command=nn_edu,padx=20,pady=5)

    test_btn = Button(text="Kayıtlı Ağı Test Et", command=nn_test,padx=20,pady=5)

    pre_btn = Button(text="Kayıtlı Ağda Örnek Tahmin Et", command=nn_pred,padx=20,pady=5)

    draw_btn = Button(text="Kayıtlı Ağı Çiz", command=draw_neural_net,padx=20,pady=5)

    draw_h_btn = Button(text="Ara Katman Testini Çiz", command=draw_h_layer,padx=20,pady=5)



    lb5.pack()
    lb_edu.pack()
    edu_c.pack()
    u_data.pack()
    r_w.pack()
    lb_hidden.pack()
    hidden_c.pack()
    lb1.pack()
    e_c.pack()
    lb2.pack()
    l_r.pack()
    lb3.pack()
    m_m.pack()
    lb4.pack()
    e_t.pack()
    lb_step.pack()
    e_step_t.pack()
    education_btn.pack(side="top")
    test_btn.pack(side="top")
    pre_btn.pack(side="top")
    draw_btn.pack(side="top")
    draw_h_btn.pack(side="top")

    mainloop()