import numpy as np
import H_makerr_v_1_1 as H_m

np.set_printoptions(threshold=np.inf)




def make_s(alpha,beta,xp,w):
    print("make s_aim")
    SIZE_x = xp[0].size
    h=H_m.make_h(xp,w)
    s = np.mat(np.array([[0.0 for xs in range(0, SIZE_x)] for ls in range(0, SIZE_x)]))
    a = np.mat(np.array([[0.0 for xa in range(0, SIZE_x)] for la in range(0, SIZE_x)]))
    a1 = np.mat(np.array([[0.0 for xa1 in range(0, SIZE_x)] for la1 in range(0, SIZE_x)]))
    a2 = np.mat(np.array([[0.0 for xa2 in range(0, SIZE_x)] for la2 in range(0, SIZE_x)]))
    temp=xp.T.dot(xp)
    temp1 = h.T.dot(h)
    for i in range(0,SIZE_x):
        for j in range(0, SIZE_x):
            if(i!=j):
               a1[i, j] = temp[i, i] + temp[j, j] - 2 * temp[i, j]
               a2[i, j] = temp1[i, i] + temp1[j, j] - 2 * temp1[i, j]
               if((alpha*a1[i,j]+beta*a2[i,j])!=0):
                    a[i,j]=1.0/(alpha*a1[i,j]+beta*a2[i,j])
               else:
                   a[i,j]=0.0
            else:
                a[i,j]=0.0
    a_sum = np.sum(a, axis=1)
    for j in range(0,SIZE_x):
        #print("making s"+str(i))
        for i in range(0, SIZE_x):
            if(i!=j and a_sum[i,0]!=0):
                s[i,j]=((a[i,j])/(a_sum[i,0]))
            else:
                s[i,j]=0.0
    #print(s)
    print("s_aim ok")
    return s

if __name__ == '__main__':
    x=np.mat("1.0 2 1;0 2 8;4 5 9;4 5 9")
    w=np.mat("1 1 ;0 2 ;4 3;4 3")
    print(make_s(1.0,1.0,x,w))
