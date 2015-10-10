'''
Created on Oct 10, 2015

@author: karishma
'''
import numpy as np
import theano.tensor as T
import theano
import scipy.misc as misc
import os
# import Image
class data_reader():
    def __init__(self,path,batch_size):
        self.dir_path = path
        self.batch_size = batch_size
        self.file_list = list()
        for dirname,_,filename in os.walk(self.dir_path):
            for fname in filename:
                self.file_list.append(os.path.join(dirname,fname))
        self._batch_str = 0
    def reinit(self):
        self._batch_str = 0
      
    def normalize(self,img):
        return (img - np.min(img))/(np.max(img)-np.min(img))
       
    def _to_gray(self,img):
        img = np.sum(img,axis=2)
        return np.array(img,dtype=np.float32)
    
    def _get_patches(self,img,size=32,):
        sz = img.itemsize
        h,w = img.shape
        bh,bw=size,size
        shape=(h/bh,w/bh,bh,bw)
        strides = sz*np.array([w*bh,bw,w,1])
        blocks=np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        return np.reshape(blocks,(-1,32*32))

    def next_batch(self):
        data_list = list()
        for fp in self.file_list[self._batch_str:self._batch_str+self.batch_size]:
            img = misc.imread(fp)
            img = self._to_gray(img)
            img = self.normalize(img)
            patch = self._get_patches(img, size=32)
            data_list.append(patch)
        data_list = np.array(data_list,dtype=np.float32)
        
        data_list = np.reshape(data_list,(-1,32*32))
        shared_list=theano.shared(data_list)
        return shared_list
    
def main():
    
    path = "/Users/karishma/Dropbox/CMU/fall_2015/deep_learning/hw2/data_rescaled/train"
    dr=data_reader(path,100)
    print dr.next_batch().shape
    
if __name__ == '__main__':
    main()
