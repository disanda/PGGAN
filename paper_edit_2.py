from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os

#-------------------------------从集合图片中截取1行的部分列--------------------------
#image = Image.open('./experiment/dataSet/actions/149-10-5.png')
#image = Image.open('./experiment/dataSet/3d-face/cd20-cc10.png')
# image = Image.open('./experiment/dataSet/shape/seq/3-3.jpg')
# #image = Image.open('./a2.png')
# #image = image.resize((300,375))
# #image.save('./a2-2.png')
# image = image.resize((1280,1280))
# array = np.array(image)

##选择所需行列
# num = 20 #起点为1
# #index = [1,3,5,7,9,12,14,16,18,20]#起点为1
# imgs=[]

# flag = array[(num-1)*64:num*64,:,:] #选择行
# print(flag.shape)

# def edit(*args): #选择列元素的函数
# 	print(type(args[0]))
# 	for j in args[0]:
# 		imgs.append(flag[:,(j-1)*64:j*64,:]) 



#---------------------------画 多 线 图-------------------

# x = np.array([1,2,3,4,5,6,7,8,9,10])  #numpy.linspace(开始，终值(含终值))，个数)
# y1 = np.array([0.44,0.41,0.4008,0.3996,0.3951,0.3795,0.3854,0.3754,0.3711,0.3704])
# y2 = np.array([0.4502,0.4468,0.4286,0.4199,0.4142,0.4104,0.4173,0.404,0.4012,0.4067])
# y3 = np.array([0.4794,0.4955,0.4852,0.4851,0.4797,0.4854,0.4785,0.4727,0.4696,0.4688])

# #画图
# plt.title('Learned Perceptual Image Patch Similarity Metric (VGG)')  #标题
# #plt.plot(x,y)
# #常见线的属性有：color,label,linewidth,linestyle,marker等
# plt.plot(x, y3, 'red', label='$G(E_s)$',linestyle='-.',marker='o')#'b'指：color='blue'
# plt.plot(x, y2, 'magenta', label='$G(E_r)$',linestyle=':',marker='o')#'b'指：color='blue'
# plt.plot(x, y1, color='blue', label='$G(E_w)$',marker='o')
# plt.legend()  #显示上面的label
# plt.xlabel('Epoch (5000*iters/ep)')
# plt.ylabel('LPIPS')
# plt.axis([1, 10, 0.35, 0.525])#设置坐标范围axis([xmin,xmax,ymin,ymax])
# #plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.show()

x = np.array([1,2,3,4,5,6,7,8,9,10])
y1 = np.array([6.6267,5.9984,5.6813,5.97,5.4128,5.58,5.1087,5.1536,4.86388,5.2483])
y2 = np.array([7.6939,8.2202,7.7461,7.43,7.317,6.9419,7.0191,7.0282,6.7789,6.5425])
y3 = np.array([6.8621,7.6169,7.8526,8.4357,8.3255,8.813,8.6035,8.5767,8.4295,8.6775])

plt.title('Fréchet Inception Distance (FID score)')
plt.plot(x, y3, 'red', label='$G(E_s)$',linestyle='-.',marker='o')#'b'指：color='blue'
plt.plot(x, y2, 'magenta', label='$G(E_r)$',linestyle=':',marker='o')#'b'指：color='blue'
plt.plot(x, y1, color='blue', label='$G(E_w)$',marker='o')
plt.legend()  #显示上面的label
plt.xlabel('Epoch (5000*iters/ep)')
plt.ylabel('FID')
plt.axis([1, 10, 4, 10])#设置坐标范围axis([xmin,xmax,ymin,ymax])
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()