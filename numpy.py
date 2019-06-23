#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
data = np.random.randn(2, 3)


# In[2]:


data


# In[3]:


data * 10


# In[4]:


data + data


# In[5]:


data.shape


# In[6]:


data.dtype


# In[7]:


data1 = [6, 7.5, 8, 0, 1]
arrl = np.array(data1)


# In[8]:


arrl


# In[9]:


data2 = [1, 2, 3, 4],[5, 6, 7, 8]
arr2 = np.array(data2)


# In[19]:


arr2


# In[11]:


arr2.ndim


# In[12]:


data.ndim


# In[13]:


arr2.dtype


# In[14]:


np.zeros(10)


# In[15]:


np.zeros((3, 6))


# In[17]:


np.empty((2,3,2))


# In[18]:


np.arange(15)


# In[20]:


arr = np.array([1, 2, 3, 4, 5])


# In[21]:


arr.dtype


# In[22]:


float_arr = arr.astype(np.float64)


# In[23]:


float_arr.dtype


# In[24]:


arr


# In[29]:


arr = np.array([[1, 2, 3], [4, 5, 6]])


# In[30]:


arr.astype(np.float64)


# In[31]:


arr *arr


# In[32]:


1 / arr


# In[34]:


arr2 = np.array([[0, 4, 1], [7, 2, 12]], dtype = np.float64)


# In[35]:


arr2 > arr


# In[37]:


arr = np.arange(10)


# In[38]:


arr


# In[39]:


arr[5]


# In[40]:


arr[0]


# In[41]:


arr[5:8]


# In[42]:


arr[5:8] = 12


# In[43]:


arr


# In[44]:


arr_slice = arr[5:8]


# In[45]:


arr_slice[1] = 12345


# In[46]:


arr


# In[47]:


arr_slice[:] = 64


# In[48]:


arr


# In[49]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# In[50]:


arr2d[2]


# In[51]:


arr2d[0][2]


# In[52]:


arr2d[0, 2]


# In[56]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


# In[57]:


arr3d


# In[59]:


arr3d.ndim


# In[60]:


arr3d[0]


# In[61]:


arr3d[0,1,2]


# In[62]:


old_value = arr3d[0].copy()


# In[63]:


arr3d[0] = 42


# In[64]:


arr3d


# In[65]:


old_value


# In[67]:


arr3d[0] = old_value


# In[69]:


arr3d


# In[70]:


arr


# In[71]:


arr2d


# In[74]:


arr2d[:2]


# In[76]:


arr2d[:2, 1:]


# In[77]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])


# In[78]:


data = np.random.randn(7,4)


# In[79]:


data


# In[80]:


names


# In[81]:


names == 'Bob'


# In[84]:


data[names == 'Bob',3]


# In[86]:


names != 'Bob'


# In[88]:


data[~(names == 'Bob')]


# In[90]:


mask = (names == 'Bob') | (names == 'Will')


# In[91]:


data[mask]


# In[92]:


data[data > 0] = 0


# In[93]:


data


# In[95]:


data[names != 'Joe'] = 7


# In[96]:


data


# In[97]:


arr = np.empty((8, 4))


# In[98]:


arr


# In[99]:


for i in range(8):
    arr[i] = i


# In[100]:


arr


# In[101]:


arr[[4, 3, 0, 6]]


# In[102]:


arr[[-3, -5, -7]]


# In[103]:


arr = np.arange(32).reshape(8, 4)


# In[104]:


arr


# In[105]:


arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[106]:


arr = np.arange(15).reshape((3, 5))


# In[107]:


arr


# In[108]:


arr.T


# In[109]:


arr = np.random.randn(6,3)


# In[110]:


arr


# In[111]:


np.dot(arr.T, arr)


# In[112]:


arr = np.arange(16).reshape((2, 2, 4))


# In[113]:


arr


# In[114]:


arr.transpose((0, 1, 2))


# In[116]:


arr.transpose((1, 0, 2))


# In[117]:


arr


# In[118]:


arr.T


# In[120]:


arr.swapaxes(1,2)


# In[121]:


arr = np.arange(10)


# In[122]:


arr


# In[123]:


np.sqrt(arr)


# In[125]:


np.exp(arr)


# In[126]:


x = np.random.randn(8)


# In[127]:


y = np.random.randn(8)


# In[128]:


x


# In[129]:


y


# In[130]:


np.maximum(x, y)


# In[131]:


arr = np.random.randn(7) * 5


# In[132]:


arr


# In[133]:


remainder, whole_part = np.modf(arr)


# In[134]:


remainder


# In[135]:


whole_part


# In[137]:


arr


# In[138]:


np.sqrt(arr)


# In[139]:


points = np.arange(-5, 5, 0.01)


# In[141]:


xs, ys = np.meshgrid(points,points)


# In[142]:


ys


# In[143]:


z = np.sqrt(xs ** 2 + ys ** 2)


# In[144]:


z


# In[32]:


import matplotlib.pyplot as plt


# In[158]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])


# In[159]:


yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])


# In[160]:


cond = np.array([True, False, True, True, False])


# In[161]:


resule = np.where(cond, xarr, yarr)


# In[162]:


resule


# In[164]:


arr = np.random.randn(4, 4)


# In[165]:


arr


# In[166]:


arr > 0


# In[168]:


np.where(arr > 0, 2 ,-2)


# In[169]:


np.where(arr > 0, 2 ,arr)


# In[170]:


arr = np.random.randn(5, 4)


# In[171]:


arr.mean()


# In[172]:


np.mean(arr)


# In[173]:


arr.sum()


# In[176]:


arr.mean(axis=0)


# In[177]:


arr.cumsum()


# In[187]:


np.arange(9).reshape((3, 3))


# In[188]:


arr = np.random.randn(100)


# In[189]:


arr


# In[191]:


(arr > 0).sum()


# In[192]:


arr = np.random.randn(6)


# In[193]:


arr


# In[194]:


arr.sort()


# In[195]:


arr


# In[196]:


arr = np.random.randn(5, 3)


# In[197]:


arr


# In[198]:


arr.sort(1)


# In[199]:


arr


# In[200]:


large_arr = np.random.randn(1000)


# In[201]:


large_arr.sort()


# In[202]:


large_arr[int(0.05 * len(large_arr))]


# In[203]:


names


# In[204]:


np.unique(names)


# In[205]:


arr = np.arange(10)


# In[206]:


np.save('some_array', arr)


# In[207]:


np.load('some_array.npy')


# In[6]:


from numpy.linalg import inv, qr


# In[7]:


X = np.random.randn(5, 5)


# In[8]:


mat =X.T.dot(X)


# In[9]:


mat


# In[10]:


inv(mat)


# In[11]:


q, r = qr(mat)


# In[12]:


r


# In[17]:


samples = np.random.normal(size = (4, 4))


# In[18]:


samples


# In[19]:


from random import normalvariate


# In[20]:


N = 10000


# In[24]:


np.random.seed(1234)


# In[27]:


rng = np.random.RandomState(1234)


# In[28]:


rng.randn(10)


# In[30]:


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


# In[33]:


plt.plot(walk[:100])


# In[37]:


nsteps = 1000


# In[38]:


draws = np.random.randint(0, 2, size=nsteps)


# In[39]:


steps = np.where(draws > 0, 1, -1)


# In[40]:


walk = steps.cumsum()


# In[41]:


walk.min()


# In[42]:


walk.max()


# In[44]:


np.abs(walk >=10).argmax()


# In[45]:


nwalks = 5000


# In[46]:


nsteps = 1000


# In[49]:


draws = np.random.randint(0, 2, size=(nwalks, nsteps))


# In[50]:


nsteps = steps = np.where(draws > 0, 1, -1)


# In[51]:


walks = steps.cumsum(1)


# In[52]:


walks


# In[53]:


walks.min()


# In[54]:


walks.max()


# In[55]:


hits30 = (np.abs(walks) >= 30).any(1)


# In[56]:


hits30


# In[58]:


hits30.sum()


# In[60]:


crossing_times = (np.abs(walks[hits30]) >=30).argmax(1)


# In[62]:


crossing_times.mean()


# In[64]:


steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))


# In[ ]:




