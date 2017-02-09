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
pcov = 90
pfc = 90
retrain = 0
model_tag = 'pcov'+str(pcov)+'pfc'+str(pfc)
while (count < 9):
    if (retrain == 0):
        pcov = pcov+1
        pfc = pfc+1
    else:
        pass
    param = [
        ('-pcov',pcov),
        ('-pfc',pfc),
        ('-m',model_tag)
        ]
    acc = training_v5.main(param)
    if (acc >= 0.99 or retrain >=3):
        model_tag = 'pcov'+str(pcov)+'pfc'+str(pfc)
        acc_list.append(acc)
        retrain = 0
        count = count + 1
    else:
        retrain += 1
    print (acc)
print('accuracy summary: {}'.format(acc_list))
