# Veri seti dosyasını okuyarak giriş ve çıkış matrislerini oluşturma
import numpy as np
import os

data= list()
out= list()
file = open("spambase_mix2.data", "r")
for line in file:
    data.append(line.split(',')[0:57])
    out.append(line.split(',')[57][0])

# print(out[1200:2600])

max_column = list()
for i in range(len(data[0])):
    max = 0

    for j in range(len(data)):
        if float(data[j][i]) > max:
            max = float(data[j][i])
    max_column.append(max)

for i in range(len(data[0])):
    for j in range(len(data)):
        data[j][i] = float(data[j][i]) / max_column[i]



# input_data = np.array(data).astype(np.float)
output_data = np.array(out)


# for i in range(len(input_data)):
#     max = np.amax(input_data[i].astype(np.float))
#     for y in range(len(input_data[i])):
#         input_data[i][y] = np.true_divide(np.float64(input_data[i][y]),max)
#
# print(input_data)
# print(output_data)
#
#
np.save("nn_input2",data)
np.save("nn_class2",output_data)