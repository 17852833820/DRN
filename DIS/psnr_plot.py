import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
# fig_size = [fig_width, fig_height]

# fig, ax1 = plt.subplots(1, 1)
# acc = torch.load('acc47.pth')
# psnr = torch.load('psnr47.pth')
# plain_acc = torch.load('plain_acc47.pth')
# a = acc/plain_acc.reshape(-1, 1)
#
# accb = torch.load('acc_167.pth')
# psnrb = torch.load('psnr_167.pth')
# plain_acc = torch.load('plain_acc_167.pth')
# ab = accb/plain_acc.reshape(-1, 1)
#
#
#
# x = []
# y = []
# for i in range(47):
#     if i not in [5, 15, 21, 37, 42, 45] and np.random.random() > 0.5:
#         for m, n in zip(psnr[i], a[i]):
#             x.append(m)
#             y.append(n)
#
# for i in range(accb.shape[0]):
#     if i in [12, 19, 21, 34, 37, 40, 44, 47, 51, 53, 55, 59, 61, 63, 67, 69,76,77,78,79,82]:
#         for m, n in zip(psnrb[i], ab[i]):
#             x.append(m)
#             y.append(n)
# plt.scatter(x, y)
# plt.show()

x = np.load('x.npy')
y = np.load('y.npy')
indexs = x.argsort()
x = x[indexs]
y = y[indexs]
# ax1.scatter(x, y, s=20, label='Actual results')

xf = np.arange(10, 43, 0.1)
a = 0.2
b = 1.5
yf = 1-b*np.e**(-a*xf)

sio.savemat('psnr-acc.mat', {'psnr': x, 'acc': y, 'psnr_mean': xf, 'acc_mean': yf})
# ax1.plot(xf, yf, color='red', label='Average results')
# ax1.legend(loc='lower right')
# plt.xlabel('PSNR1[dB]')
# plt.ylabel(r'Normalized[%]')
# plt.show()