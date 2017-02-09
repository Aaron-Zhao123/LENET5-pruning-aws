import os
import training_v5
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')

acc_list = []
count = 0
pcov = 95
pfc = 95
retrain = 0
model_tag = 'pcov'+str(pcov)+'pfc'+str(pfc)
while (count < 5):
    pcov = pcov+1
    pfc = pfc+1
    param = [
    ('-pcov',pcov),
    ('-pfc',pfc),
    ('-m',model_tag)
    ]
    acc = training_v5.main(param)
    model_tag = 'pcov'+str(pcov)+'pfc'+str(pfc)
    acc_list.append(acc)
    count = count + 1
    print (acc)

print('accuracy summary: {}'.format(acc_list))
