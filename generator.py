import matplotlib.pyplot as plt
import numpy as np
shade0=255
shade1=180
shade2=100
shade3=50
shade4=10
#print("[shade{s},shade{s},shade{s}]".format(s=3))
rad=2
n=2*rad-1
colors=[1,0.7,0.5,0.2,0.1]
print('[',end='')
for i in range(n):
    print('[',end='')
    for j in range(n):
        print(colors[int(abs(i-rad+1)+abs(j-rad+1))],end='')
        if j!=n-1:
            print(',',end='')
    print(']',end='')
    if i!=n-1:
            print(',',end='')
print(']',end='')

plt.imshow(np.array([[[10, 10, 10],[50, 50, 50],[100, 100, 100],[50, 50, 50],[10, 10, 10]],[[50, 50, 50],[100, 100, 100],[180, 180, 180],[100, 100, 100],[50, 50, 50]],[[100, 100, 100],[180, 180, 180],[255, 255, 255],[180, 180, 180],[100, 100, 100]],[[50, 50, 50],[100, 100, 100],[180, 180, 180],[100, 100, 100],[50, 50, 50]],[[10, 10, 10],[50, 50, 50],[100, 100, 100],[50, 50, 50],[10, 10, 10]]]))
plt.show()