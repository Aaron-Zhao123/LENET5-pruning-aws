import os
import training_v3
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')

acc_list = []
count = 6
while (count < 10):
    param = [
        ('-p',count),
        ('-tc1',0.12),
        ('-tc2',0.14),
        ('-tfc1',0.15),
        ('-tfc2',0.15)]
    acc = training_v3.main(param)
    print (acc)
    if (acc >= 0.99):
        acc_list.append(acc)
        count = count + 1
print('accuracy summary: {}'.format(acc_list))
