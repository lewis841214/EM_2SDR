import torch
from tomosipo.torch_support import to_autograd
import time
import numpy as np
import astra
import tomosipo as ts
import numpy as np
import pickle
import mrcfile
import numpy as np
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd  
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
import os
import matplotlib.pyplot as plt
import astra
from tomosipo.geometry.parallel_vec import *
from scipy.ndimage import zoom
"""
First step. write out the
1. Expectation
2. likelihood
Then use .backward solve it


For AU with U with batch n^3, to_autograd take batch input, which is actually what I want!!!!, so no need for one by one concatenate!!!

"""
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
def Trace_bradcast(a):
    b = torch.einsum('bii->b', a)
    return b



class EM_2SDR():
    def __init__(self, ProjSize, num_image, n_component, op, Images,Orientation_Vectors, exp_name,mean_subtracted_strcut, true_index ,  batch_size = 1000, n_iter = 20, PCA_n = 5):
        #init
        self.exp_name = exp_name
        self.ProjSize  = ProjSize
        self.z_size = n_component # First set 1 for simplicity
        self.num_image = num_image
        self.n_component = n_component
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.PCA_n = PCA_n

        """
        The answer 
        """
        self.true_index = true_index
        self.mean_subtracted_strcut = mean_subtracted_strcut
        """
        projection operator and images
        """
        self.op = to_autograd(op)
        Images = torch.tensor(Images)
        self.All_Image = Images
        self.I = reshape_fortran(Images, (self.num_image, self.ProjSize*self.ProjSize , 1)).float() # N x D^2 x 1
        self.All_I = self.I
        self.Orientation_Vectors = Orientation_Vectors
        """
        paramters we want to find
        """
        #x = int((ProjSize - self.z_size ) /2)
        self.U1 = torch.eye(self.ProjSize)[:, :self.z_size] +1
        self.U2 = torch.eye(self.ProjSize)[:, :self.z_size] +1 
        self.U3 = torch.eye(self.ProjSize)[:, :self.z_size] +1
        #print('self.U1',self.U1.shape)
        self.U1.data = torch.qr(self.U1.data)[0]
        self.U2.data = torch.qr(self.U2.data)[0]
        self.U3.data = torch.qr(self.U3.data)[0]
        #print('ortho_U1',ortho_U1.shape)
        #print(ortho_U1.T@ortho_U1)
        #self.mask = torch.eye(self.ProjSize)[:, x : x + self.z_size]
        
        self.U1.requires_grad =True
        self.U2.requires_grad =True
        self.U3.requires_grad =True
        self.All_U = torch.kron(self.U1, torch.kron(self.U2, self.U3))
        self.Sigma_Z_inv = torch.eye(self.n_component ** 3) # size: p1 x p2 x p3 
        self.sigma = torch.ones(self.batch_size).reshape(-1, 1 ,1) * 1
        #print('sigma monotoring',torch.mean(self.sigma))
        """
        Expectation term
        or so called missing data
        """
        self.mu = torch.abs(torch.ones((self.batch_size, n_component **3,1)))
        self.BigSigma_I = torch.abs(torch.ones((self.batch_size, self.n_component**3, self.n_component**3 )))

       

    def A_pro_All_U(self, detach_list = []):
        #print('in A por U')
        #self.All_U # shape = Ps ^3 x n  ^3: which we should make it Ps x Ps x Ps x n ^3 to process
        #A: Ps x Ps x Ps x 1 ->  Ps x Ps x 1
        
        #So overal, we have Mu = N x D^3 x n^3
        #AMu = N x D^2 x n^3
        """
        for i in range(self.n_component**3):
            Vol_i = self.All_U[:, i].reshape(self.ProjSize, self.ProjSize,self.ProjSize)
            Proj_i = self.op(Vol_i).permute(1, 0, 2)
            Proj_i = reshape_fortran(Proj_i, (self.num_image, self.ProjSize*self.ProjSize ,1))
            
            #print(Proj_i.shape)
            if i == 0:
                self.AU = Proj_i
            else:
                self.AU = torch.cat((self.AU, Proj_i), 2)
        #print('self.AU',self.AU.shape)
        
        #Broad cast: torch.matmul
        # (AU).T(AU) = torch.matmul(self.AU.permute(0,2,1), self.AU.permute(0,2,1))
        #torch.matmul(self.AU.permute(0,2,1), self.AU)
        """
        U1 , U2, U3 = self.U1, self.U2, self.U3
        if 1 in detach_list:
            U1 = U1.detach()
        if 2 in detach_list:
            U2 = U2.detach()
        if 3 in detach_list:
            U3 = U3.detach()
        
        
        self.All_U = torch.kron(U1, torch.kron(U2, U3))
        self.All_U_kron_shape = self.All_U
        self.All_U = reshape_fortran(self.All_U, (self.ProjSize, self.ProjSize, self.ProjSize, self.n_component**3)) #D^3 x n^3 -> D xD x D x n^3
        self.All_U = self.All_U.permute(3, 0, 1, 2) # make it n x Dx Dx D
        #print(self.All_U.shape)
        self.AU = self.op( self.All_U) # n x Dx Dx D -> n x D x N x D
        #print('self.AU',self.AU.shape)
        self.AU = self.AU.permute(2, 0, 1, 3) # N x n x D x D
        #print( self.AU.permute(2, 0, 1, 3).shape)
        self.AU = reshape_fortran(self.AU, (self.batch_size, self.n_component**3, self.ProjSize**2))
        self.AU = self.AU.permute(0, 2, 1) # N x n x D x D
        
        #print('self.AU',self.AU.shape)
        
    def Expectation(self):
        #print('in Exp')
        
        """
        First, get all projection information
        
        ro_j = 1/sigma^2 * (Sigmz_Z_inv + 1/sigma^2 * (Au)^TAu)^(-1)(Au)^TI
        
        #We expect to get a n_image x n_component matrix for ro
        
        BigSigma = I - 1 / sigma^2 * (I + 1/sigma^2 * (Au)^TAu)^(-1) (Au)^TAu9htu
        
        Here, the shape of All_U is PS^3 x n_component^3, but A can only process a 3d volume PS^3 at once
        
        """
        #print(torch.unsqueeze(self.Sigma_Z_inv,0  ).shape)
        #print((1 / self.sigma **2).shape)
        
        UTATAU = torch.matmul(self.AU.permute(0,2,1), self.AU)
        #print(UTAAU.shape)
        #print((1 / self.sigma **2 * UTAAU ).shape)
        self.mu = torch.inverse(torch.unsqueeze(self.Sigma_Z_inv,0  ) + 1 / self.sigma **2 * UTATAU ) * 1 / self.sigma **2 
        self.mu = torch.matmul(self.mu , self.AU.permute(0,2,1))
        self.mu = torch.matmul(self.mu, self.I)
        #print('self.mu.shape',self.mu.shape)
        
        
        """
        So we found Mu
        Then we found Sigma_I
        torch.inverse can broadcast through batch, so just use it
        """
        self.BigSigma_I = torch.inverse((torch.unsqueeze(self.Sigma_Z_inv,0  ) + 1 / self.sigma **2 * UTATAU )) + torch.matmul(self.mu, self.mu.permute(0,2,1))
        #print('self.BigSigma_I', self.BigSigma_I.shape)
        self.mu = self.mu.detach()
        self.BigSigma_I = self.BigSigma_I.detach()
        #print('torch.mean(self.mu)', torch.mean(self.mu))
        #print('torch.mean(self.BigSigma_I)', torch.mean(self.BigSigma_I))
        
    def Q_Function(self, ):
        #print('in Q')
        """
        #left term
        #print('self.AU', torch.mean(self.AU))
        print('in Q')
        print('self.mu ', torch.mean(self.mu ))
        print('self.BigSigma_I', torch.mean(self.BigSigma_I))
        #print('self.sigma', torch.mean(self.sigma))
        print('self.Sigma_Z_inv',torch.mean(self.Sigma_Z_inv))
        print('self.U1', torch.mean(self.U1))
        print('self.U2', torch.mean(self.U2))
        print('self.U3', torch.mean(self.U3))
        
        try:
            print('self.U1.grad.data',torch.mean(self.U1.grad.data))
            print('self.U2.grad.data',torch.mean(self.U2.grad.data))
            print('self.U3.grad.data',torch.mean(self.U3.grad.data))
        
        except:
            pass
        """
        
        left = -2 * self.I.permute(0, 2, 1) @ self.AU @ self.mu 
        #UTATAU = torch.matmul(self.AU.permute(0,2,1), self.AU)
        #print('self.AU', self.AU.shape)
        #print('self.BigSigma_I', self.BigSigma_I.shape)
        #print('self.AU.permute(0,2,1)', self.AU.permute(0,2,1).shape)
        left = left + Trace_bradcast(self.AU @ self.BigSigma_I @ self.AU.permute(0,2,1))   #self.mu.permute(0, 2, 1) @  UTATAU @ self.mu
        left = 1 / self.sigma **2 *left
        
        #middle term
        #print('self.mu', self.mu.shape)
        #print('torch.unsqueeze(self.Sigma_Z_inv,0  )', torch.unsqueeze(self.Sigma_Z_inv,0  ).shape)
        #middle = self.mu.permute(0, 2, 1)  @torch.unsqueeze(self.Sigma_Z_inv,0  ) @self.mu
        middle = Trace_bradcast( torch.unsqueeze(self.Sigma_Z_inv,0  ) @ self.BigSigma_I )
        
        
        right = torch.log(torch.det(torch.inverse(self.Sigma_Z_inv)))
        self.Q = -1/2 * torch.sum(left + middle + right)/100000000
        #print('self.Q',self.Q.shape,self.Q)
        #print(self.Q)
    def Maximization(self, ratio):
        #print('in max')
        """
        Update U1, U2, U3 by gradient assent and use svd to make sure orthogonality
        self.U1.grad.data.zero_() # conduct zero grad
        self.Q.backward()
        
        """
        self.Sigma_Z_inv = torch.inverse(torch.mean(self.BigSigma_I, 0)).detach()
        #print('self.Sigma_Z_inv ', torch.mean(self.Sigma_Z_inv ))
        
        start = time.time()
        
        self.A_pro_All_U([2,3])
        self.Q_Function()
        #print('q fun ction time', time.time()- start)
        self.Q.backward()
        #print('backward time', time.time()- start)
        
        lr = torch.mean(torch.abs(self.U1)) /  torch.mean(torch.abs(self.U1.grad)) * ratio# 0.01
        #print('torch.mean(torch.abs(self.U1.grad))', torch.mean(torch.abs(self.U1.grad)))
        self.U1.data +=  lr * self.U1.grad.data
        self.U1.data = torch.qr(self.U1.data)[0] 
        
        self.U1.grad.data.zero_()

        #print('update time', time.time()- start)
        
        self.A_pro_All_U()
        #print('AU time', time.time()- start)
        ##self.Expectation()
        #print('expectation time', time.time()- start)
        
        
        self.A_pro_All_U([1,3])
        self.Q_Function()
        self.Q.backward()
        lr = torch.mean(torch.abs(self.U2)) /  torch.mean(torch.abs(self.U2.grad))* ratio#0.01
        self.U2.data += lr * self.U2.grad.data
        self.U2.data = torch.qr(self.U2.data)[0] 
        
        self.U2.grad.data.zero_()
        self.A_pro_All_U()
        #self.Expectation()
        
        
        self.A_pro_All_U([1,2])
        self.Q_Function()
        self.Q.backward()
        
        lr = torch.mean(torch.abs(self.U3)) /  torch.mean(torch.abs(self.U3.grad))* ratio#0.01
        self.U3.data += lr * self.U3.grad.data
        self.U3.data = torch.qr(self.U3.data)[0] 
        
        self.U3.grad.data.zero_()
        self.A_pro_All_U()
        #self.Expectation()
        print(time.time()-start, 'sec')
        #lr = np.abs(np.mean(self.mu) / np.mean(DeltaMu) )*0.001
        #self.mu = self.mu + lr * DeltaMu
        """
        lr = torch.mean(torch.abs(self.U1)) /  torch.mean(torch.abs(self.U1.grad)) * ratio# 0.01
        self.U1.data +=  lr * self.U1.grad.data
        
        lr = torch.mean(torch.abs(self.U2)) /  torch.mean(torch.abs(self.U2.grad))* ratio#0.01
        self.U2.data += lr * self.U2.grad.data
        
        lr = torch.mean(torch.abs(self.U3)) /  torch.mean(torch.abs(self.U3.grad))* ratio#0.01
        self.U3.data += lr * self.U3.grad.data
        
        self.U1.data = torch.qr(self.U1.data)[0]
        self.U2.data = torch.qr(self.U2.data)[0]
        self.U3.data = torch.qr(self.U3.data)[0]
        #print('self.U1.grad.data',torch.mean(self.U1.grad.data))
        ##print('self.U2.grad.data',torch.mean(self.U2.grad.data))
        #print('self.U3.grad.data',torch.mean(self.U3.grad.data))
        
        self.U1.grad.data.zero_()
        self.U2.grad.data.zero_()
        self.U3.grad.data.zero_()
        self.A_pro_All_U()
        """
        #print('self.I.permute(0, 2, 1) @ self.I', (self.I.permute(0, 2, 1) @ self.I).shape)
        #print('-2 * self.I.permute(0, 2, 1) @ self.AU @ self.mu', (-2 * self.I.permute(0, 2, 1) @ self.AU @ self.mu).shape)
        #print('Trace_bradcast(self.AU @ self.BigSigma_I @ self.AU.permute(0,2,1))', (Trace_bradcast(self.AU @ self.BigSigma_I @ self.AU.permute(0,2,1))).shape)
        self.sigma = self.I.permute(0, 2, 1) @ self.I -2 * self.I.permute(0, 2, 1) @ self.AU @ self.mu  + (Trace_bradcast(self.AU @ self.BigSigma_I @ self.AU.permute(0,2,1))).reshape(-1, 1, 1)
        #print('self.sigma',self.sigma.shape)
        self.sigma = self.sigma.detach() / self.ProjSize ** 2
        #print('sigma monotoring',torch.mean(self.sigma))
        #print('sigma monotoring min',torch.min(self.sigma))
        #print('sigma monotoring max',torch.max(self.sigma))
        #print(' self.I.permute(0, 2, 1) @ self.I',  (self.I.permute(0, 2, 1) @ self.I).shape)
        #print('self.sigma',self.sigma.shape)
    def Plot_temp(self, num = 10):
        #print('monotor sigma j max', torch.max(self.sigma))
        fig,axes=plt.subplots(1,num, figsize=(20, 200))
        print('generated')
        for i in range(num):
            est_I = self.AU @ self.mu
            AU = est_I.detach().numpy()
            image = AU[i].reshape(self.ProjSize,self.ProjSize, order = 'F')
            axes[i].imshow(image)
        plt.show()
        fig,axes=plt.subplots(1,num , figsize=(20, 200))
        print('real')
        for i in range(num):
            #est_I = self.AU @ self.mu
            #AU = est_I.detach().numpy()
            axes[ i].imshow(self.Images[i])
        plt.show()
        
        """
        c = 0
        print('generated')
        est_I = self.AU @ self.mu
        AU = est_I.detach().numpy()
        #print(AU.shape)
        image = AU[c].reshape(self.ProjSize,self.ProjSize, order = 'F')
        plt.imshow(image) # quarter rotation
        plt.show()
        #MP3.Q_Function()

        #MP3.Maximization()
        print('real')
        plt.imshow(Images[c])
        plt.show()
        
        c = 11
        print('generated')
        #AU = self.AU.detach().numpy()
        #print(AU.shape)
        image = AU[c].reshape(self.ProjSize,self.ProjSize, order = 'F')
        plt.imshow(image) # quarter rotation
        plt.show()
        #MP3.Q_Function()

        #MP3.Maximization()
        print('real')
        plt.imshow(Images[c])
        plt.show()
        """
        
        
        
    def fit(self, start_ratio, estimate = False):
        
        """
        If doing SGD, set self.I as part of the images, then the following code run aotumatically.
        Use np.random.permutation(5000)
        self.All_I 
        """
        
        ratio = start_ratio
        """
        ratio = start_ratio
        self.A_pro_All_U()
        self.Expectation()
        print('oritinal')
        self.Plot_temp()
        """
        
        temp = []
        for i in range(2):
            
            order = np.random.permutation(self.num_image)
            
            if estimate == False:
                pass
            #else:
            #    self.Draw_Kmean_tsne(0)
            #self.A_pro_All_U()
            #self.Plot_temp()
            #for j in range(int(self.num_image / self.batch_size)):
            for j in range(1, self.n_iter+1):
                #continue
                batch_order = order[self.batch_size * j : self.batch_size * (j + 1) ]
                self.I = self.All_I[batch_order] 
                self.Images = self.All_Image[batch_order]
                #print('self.I.shape', self.I.shape)
                #print('batch_order', batch_order.shape)
                
                #Update op
                Proj_geom = astra.create_proj_geom('parallel3d_vec', self.ProjSize, self.ProjSize, self.Orientation_Vectors[batch_order])
                Vol_geom = astra.create_vol_geom(self.ProjSize, self.ProjSize, self.ProjSize)
                vg = ts.from_astra(Vol_geom)
                pg = ts.from_astra(Proj_geom)
                op = ts.operator(vg, pg)
                self.op = to_autograd(op)
                
                self.A_pro_All_U()
                #self.Q_Function()
                #self.Plot_temp()
                #self.Q_Function()
                self.Expectation()
                self.Q_Function()
                self.Maximization(ratio)
                print(f'{i} batch {j} iteration')
                self.Plot_temp()
                torch.save(self.U1, f'./snap_shot/{self.exp_name}_{i}th_{j}_U1.pt')
                torch.save(self.U2, f'./snap_shot/{self.exp_name}_{i}th_{j}_U2.pt')
                torch.save(self.U3, f'./snap_shot/{self.exp_name}_{i}th_{j}_U3.pt')
                torch.save(self.Sigma_Z_inv, f'./snap_shot/{self.exp_name}_{i}th_{j}_Sigma_Z_inverse.pt')
                torch.save(self.sigma, f'./snap_shot/{self.exp_name}_{i}th_{j}_sigma.pt')
                self.iter = j+self.n_iter * i
                if j % 3 == 1:
                    if estimate == False:
                        pass
                    else:
                        self.Draw_Kmean_tsne(j+self.n_iter * i, pca_n = self.PCA_n)
                    #self.Draw_Kmean_tsne(j+self.n_iter * i )
                #self.Plot_temp()
                #if j % 6 == 5:
                #    print('ratio / 5')
                #    ratio = ratio / 10
            if i ==2:
                ratio = ratio / 10
            print(f'the {i} th iter, ratio = {ratio}', )
            #self.Plot_temp()
            
            temp.append([self.U1.data, self.U2.data, self.U3.data])
            with open(f'./snap_shot/{self.exp_name}_MPCA.pkl', 'wb') as f:
                pickle.dump(temp, f)
            torch.save(self.U1, f'./snap_shot/{self.exp_name}_{i}th_U1.pt')
            torch.save(self.U2, f'./snap_shot/{self.exp_name}_{i}th_U2.pt')
            torch.save(self.U3, f'./snap_shot/{self.exp_name}_{i}th_U3.pt')
            #use torch save to save torch tensor
            self.Draw_Kmean_tsne(j , pca_n = self.PCA_n)
            
    def Output_colection(self, n = 8):
        
        """
        If doing SGD, set self.I as part of the images, then the following code run aotumatically.
        Use np.random.permutation(5000)
        self.All_I 
        """
       
        
        temp = []

        order = np.array([i for i in range(self.num_image)])[ :n * self.batch_size]
        for j in range(n):
            print(f'output_collection {j}-th batch')
            batch_order = order[self.batch_size * j : self.batch_size * (j + 1) ]
            self.I = self.All_I[batch_order] 
            self.Images = self.All_Image[batch_order]
            #print('self.I.shape', self.I.shape)
            #print('batch_order', batch_order.shape)

            #Update op
            Proj_geom = astra.create_proj_geom('parallel3d_vec', self.ProjSize, self.ProjSize, self.Orientation_Vectors[batch_order])
            Vol_geom = astra.create_vol_geom(self.ProjSize, self.ProjSize, self.ProjSize)
            vg = ts.from_astra(Vol_geom)
            pg = ts.from_astra(Proj_geom)
            op = ts.operator(vg, pg)
            self.op = to_autograd(op)

            self.A_pro_All_U()
            #self.Q_Function()
            #self.Plot_temp()
            #self.Q_Function()
            self.Expectation()
            #self.Plot_temp()
            temp.extend(self.mu.detach().numpy().tolist())
        self.All_mu = temp
        with open(f'./snap_shot/{self.exp_name}_all_mu_{self.iter}.pkl', 'wb') as f:
            pickle.dump(self.All_mu , f)
    def Draw_Kmean_tsne(self, iter_, num_batch = 1 , pca_n = 5):
        self.iter = iter_
        true_index = self.true_index
        #num_batch = int(9200 / self.batch_size)

        num_batch = num_batch
        self.Output_colection(num_batch)
        indexs = true_index[:num_batch * self.batch_size]
        np.random.seed(0)
        Coef = np.array(self.All_mu)
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        Coef = Coef.reshape(Coef.shape[0], -1)
        pca = PCA(n_components=pca_n)
        pca.fit(Coef)
        P_Coef = pca.transform(Coef)
        self.P_Coef = P_Coef
        self.PCs = pca.components_
        #Save components
        with open(f'./snap_shot/{self.exp_name}_PCA_W_iter_{self.iter}.pkl', 'wb') as f:
            pickle.dump(self.PCs, f)
        z = tsne.fit_transform(P_Coef) 
        self.Projected_var()
        
        kmeans = KMeans(n_clusters=pca_n, random_state=0).fit(P_Coef) #k-mean perform on PCA coef
        kmeans.labels_
        vs = v_measure_score(indexs, kmeans.labels_)
        df = pd.DataFrame()
        df["y"] = kmeans.labels_
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", pca_n),
                        data=df).set(title=f"n = {self.z_size} 5Ribsome data k-mean at iter-{iter_} \n vm = {str(vs)[:6]} log_proj_var ={torch.log(self.projected_var):.2f} log_ori_var={torch.log(self.ori_var):.2f}") 
        os.makedirs(os.path.dirname(f'./TSNE_result/{self.exp_name}/n = {self.z_size} 5Ribsome data k-mean at iter-{iter_} vm = {vs} log_proj_var ={torch.log(self.projected_var):.2f}.jpg'), exist_ok = True)
        plt.savefig(f'./TSNE_result/{self.exp_name}/n = {self.z_size} 5Ribsome data k-mean at iter-{iter_} vm = {str(vs)[:6]} log_proj_var ={torch.log(self.projected_var):.2f}.jpg')
        plt.show()
        
        plt.close()
       # print(indexs[:10])
        df = pd.DataFrame()
        df["y"] = indexs
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", pca_n),
                        data=df).set(title=f"n = {self.z_size} 5Ribsome data true_index at iter-{iter_}\n vm = {str(vs)[:6]} log_proj_var ={torch.log(self.projected_var):.2f} log_ori_var={torch.log(self.ori_var):.2f}") 
        plt.savefig(f'./TSNE_result/{self.exp_name}/n = {self.z_size} 5Ribsome data true_index at iter-{iter_} vm = {str(vs)[:6]} log_proj_var ={torch.log(self.projected_var):.2f}.jpg')
        plt.show()
        plt.close()
        
        
    def Projected_var(self):
        """
        self.All_U # V^3 x n_component ^ 3
        self.PCs # n_component ^ 3 x n_pcs
        ms_strucs # 5 x v x v x v
        return :
        self.I = reshape_fortran(Images, (self.num_image, self.ProjSize*self.ProjSize , 1)).float()
        """
        ms_strucs = self.mean_subtracted_strcut
        self.PCs = torch.permute(torch.tensor(self.PCs), (1, 0)).float()
        print('self.All_U_kron_shape', self.All_U_kron_shape.shape)
        print('self.PCs.shape', self.PCs.shape)
        
        self.two_SDR_PCs = self.All_U_kron_shape @ self.PCs #V^3 x n_pcs
        torch.save(self.two_SDR_PCs, f'./snap_shot/{self.exp_name}_{self.iter}th_2SDR_PCs.pt')
        #print(self.two_SDR_PCs.T @self.two_SDR_PCs  ) # chekced! orthonormal!
        ms_strucs = torch.tensor(ms_strucs)
        #print('ms_strucs', ms_strucs.shape)
        #print('self.n_component', self.n_component)
        self.ms_strucs = reshape_fortran(ms_strucs, (self.PCA_n, self.ProjSize*self.ProjSize*self.ProjSize)).float()
        self.ms_strucs = torch.permute(torch.tensor(self.ms_strucs), (1, 0))
        
        #print('original variance = ',torch.sum(self.ms_strucs * self.ms_strucs))
        
        projected_struct = self.two_SDR_PCs.T @ self.ms_strucs
        #print('original variance = ',torch.sum(projected_struct * projected_struct))
        self.ori_var = torch.sum(self.ms_strucs * self.ms_strucs)
        self.projected_var = torch.sum(projected_struct * projected_struct)
        
        self.ori_var_each = torch.sum(self.ms_strucs * self.ms_strucs, axis = 0)
        self.projected_var_each = torch.sum(projected_struct * projected_struct, axis = 0)
        print('self.ori_var_each', self.ori_var_each/(1e12))
        print('self.projected_var_each', self.projected_var_each/(1e12))
        
        
