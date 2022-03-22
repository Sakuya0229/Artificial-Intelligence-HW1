import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
gt = pd.read_csv('GroundTruth.txt', sep=" ", header=None)

## adaboost prediction
adaboost = pd.read_csv('Adaboost_pred.txt', sep=" ", header=None)
adaboost_m = pd.read_csv('Adaboost_pred_m.txt', sep=" ", header=None)

## yolo prediction
yolo = pd.read_csv('Yolov5_pred.txt', sep=" ", header=None)

gt_sum = gt.sum(axis = 1)
adaboost_sum = adaboost.sum(axis = 1)
adaboost_m_sum = adaboost_m.sum(axis = 1)
yolo_sum = yolo.sum(axis = 1)

plt.figure()
plt.plot(gt_sum, label='Ground truth', color='red')
plt.plot(adaboost_sum, label='Adaboost', color='blue')
plt.plot(adaboost_m_sum, label='Adaboostmean', color='green')
plt.plot(yolo_sum, label='Yolo', color='black')

plt.xlabel('time')
plt.ylabel('# car')
plt.xlim((0,49))
plt.ylim((0,76))
plt.title('Parking Slots Occupation')
plt.legend()
plt.savefig('Parking_Slots_Occupation.png')
plt.show()

def get_accuracy(df):
  return np.sum(np.equal(gt, df), axis=1) / gt.shape[1]

adaboost_acc = get_accuracy(adaboost)
adaboost_m_acc = get_accuracy(adaboost_m)
yolo_acc = get_accuracy(yolo)

plt.figure()
plt.plot(adaboost_acc, label='Adaboost', color='blue')
plt.plot(adaboost_m_acc, label='Adaboostmean', color='green')
plt.plot(yolo_acc, label='Yolo', color='black')

plt.xlabel('time')
plt.ylabel('accuracy')
plt.xlim((0,49))
plt.ylim((0,1))
plt.title('Accuracy')
plt.legend()
plt.savefig('Accuracy.png')
plt.show()
