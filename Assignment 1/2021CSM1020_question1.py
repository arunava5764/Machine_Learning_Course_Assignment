# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:48:00 2022

@author: Ananya_Chaudhuri
"""
import pandas as pd
import random
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# =============================================================================
# numberList = [0,1] 
# Y=random.choices(numberList, weights=(50,50), k=10)
# print(Y)
# =============================================================================
def y_vector(a,b,n):
    Y=[]
    for i in range(n):
        Y.append(random.randint(0,1))
    #print(Y)
    #Y = np.array([Y]).T
    return Y
Y=y_vector(0,1,30)


X=[]
for i in Y:
    p=[]
    if i==1:
       p.append(np.random.uniform(2, 7))
       p.append(np.random.uniform(4, 6))
       X.append(p)
    else:
        s=[]
        s.append(np.random.uniform(0, 2))
        s.append(np.random.uniform(7, 9))
        p.append(random.choice(s))
        s=[]
        s.append(np.random.uniform(1, 3))
        s.append(np.random.uniform(6, 8))
        p.append(random.choice(s))
        X.append(p)
X1=np.mat(X)
#print(X)
#print(Y)
df = pd.DataFrame(data = X1, 
                  index = None, 
                  columns = ['X1','X2'])
s = pd.Series(Y, index=None)
df = pd.concat([df, s.rename('Y')], axis=1)
#print(df)
colors = np.where(df['Y']==1,'r','g')
df.plot.scatter(x="X1",y="X2",c=colors)
plt.title("Data points Plot")
plt.show()
x1,y1,x0,y0=[],[],[],[]
for i in range(len(Y)):
    if Y[i]==1:
        x1.append(X[i][0])
        y1.append(X[i][1])
    if Y[i]==0:
        x0.append(X[i][0])
        y0.append(X[i][1])

# Most specific Hypothesis for all possible rectangles
cord_x,cord_y=min(x1),min(y1)
#print(cord_x,cord_y)
width,height=max(x1)-min(x1),max(y1)-min(y1)
#print(width,height)

# Most general Hypothesis for for all possible rectangles
cord_x0,cord_y0=[],[]
diag_x0,diag_y0=[],[]
for i in x0:
    if i<cord_x:
        cord_x0.append(i)
    if i>max(x1):
        diag_x0.append(i)
for i in y0:
    if i <cord_y:
        cord_y0.append(i)
    if i>max(y1):
        diag_y0.append(i)
#print(diag_x0)
#print(diag_y0)
cord_x0,cord_y0=max(cord_x0),max(cord_y0)
diag_x0,diag_y0=min(diag_x0),min(diag_y0)
#print(cord_x0,cord_y0)
#print(diag_x0,diag_y0)
#side=(1/np.sqrt(2))*np.linalg.norm(np.array((cord_x0,cord_y0)) - np.array((diag_x0,diag_y0)))
#side=(1/np.sqrt(2))*np.sqrt((cord_y0-diag_y0)**2+(cord_x0-diag_x0)**2)
side1,side2=diag_x0-cord_x0,diag_y0-cord_y0
#print(side1,side2)
#print(side)
fig, ax = plt.subplots()
ax.scatter(x="X1",y="X2",c=colors,data=df)
#add rectangle to plot
ax.add_patch(Rectangle((cord_x,cord_y), width,height,
             edgecolor='r', facecolor='none',
             fill=False,
             lw=1))
ax.add_patch(Rectangle((cord_x0,cord_y0), max(side1,side2),max(side1,side2),
             edgecolor='g', facecolor='none',
             fill=False,
             lw=1))
#display plot
plt.show()



# Most specific Hypothesis for all possible circles
center_x,center_y=min(x1)+(max(x1)-min(x1))/2,min(y1)+(max(y1)-min(y1))/2
dist1,dist0=[],[]
for i in range(len(Y)):
    if Y[i]==1:
        dist1.append(np.sqrt((X[i][0]-center_x)**2+(X[i][1]-center_y)**2))
#print(center_x,center_y)
radius1=max(dist1)
#print(radius1)

# Most general Hypothesis for for all possible circles
side=np.linalg.norm(np.array((cord_x0,cord_y0)) - np.array((diag_x0,diag_y0)))
fig, ax = plt.subplots()
ax.scatter(x="X1",y="X2",c=colors,data=df)
circle1 = plt.Circle((center_x,center_y), radius1, color='r', fill=False)
circle0 = plt.Circle((center_x,center_y), side/2, color='g', fill=False)
ax.add_patch(circle1)
ax.add_patch(circle0)
plt.show()
