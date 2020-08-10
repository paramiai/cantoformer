#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: profile=False, embedsignature=True, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=2, language=c++
'''
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

cdef long randint(int m):
    return <long>(drand48()*m)

'''
from __future__ import print_function
    
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np

cimport numpy as np
np.import_array()


import os
import io
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

cdef np.int32_t CLS = 1
cdef np.int32_t SEP = 2
cdef np.int32_t NULL_LABEL = -100

cdef long find_first_zero_from_bytes(a, long start, long size):
    size -= 2
    while size >= 0:
        if a[start+size] == a[start+size+1] == 0:
            size -= 2
            continue
        return size + 2
    return size

        
from fairseq.data import FairseqDataset

class textDataset(FairseqDataset):
    def __init__(self, 
                 data_path, 
                 seq_length,
                 batch_size, 
                 ram=False, 
                 eval=False, 
                 eval_num_samples=100000, 
                 cooked=None):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.eval_num_samples = (eval_num_samples // batch_size) * batch_size
        self.eval = eval
        
        try:
            if cooked == False:
                raise

            self.length = os.stat(data_path+"data_original_cooked").st_size//(seq_length*2)
            path = data_path+"data_original_cooked"
            with open(path, 'rb') as f:
                pass
            cooked = True
        except:
            if cooked == True:
                raise Exception ('Not yet cooked')
            self.length = os.stat(data_path+"data_original").st_size//(seq_length*2)
            cooked = False


        self.actual_length = self.length


        if self.length < eval_num_samples:
            self.eval_num_samples = eval_num_samples = ((self.length // 10) // batch_size) * batch_size


        if eval:
            self.length = eval_num_samples
        

        print(
            'loaded dataet (cooked: %s, eval:%s, len:%s, eval_len:%s, tot_len:%s)'%(
                cooked, eval, self.length, self.eval_num_samples, self.actual_length), flush=True)

            
        self.cooked = cooked

        self.ids_bin_buffer = None

    def __del__(self):
        try:
            self.ids_bin_buffer_mmap._mmap.close()
            del self.ids_bin_buffer_mmap
            self.masked_bin_buffer_mmap._mmap.close()
            del self.masked_bin_buffer_mmap
            self.labels_bin_buffer_mmap._mmap.close()
            del self.labels_bin_buffer_mmap
            self.attention_mask_bin_buffer_mmap._mmap.close()
            del self.attention_mask_bin_buffer_mmap
            
        except:
            pass
                
                
    def ordered_indices(self):
        return np.arange(len(self))
    
    def collater(self, samples):
        return tuple(map(torch.stack, zip(*samples)))
    
    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        batches = np.array_split(indices, indices.shape[0] // self.batch_size)
        return batches

    def __len__(self):
        return self.eval_num_samples if self.eval else self.length - self.eval_num_samples
    
    def num_tokens(self, index):
        return self.seq_length
    
    def size(self, index):
        return self.seq_length
    
    def dump(self):
        cdef long shift, start, seq_length, b_length, end, length, mult = 100, i=-1, b_seq_length, count, max_bytes
        cdef unsigned int j
        cdef const np.int16_t[:] ids_buffer
        cdef const np.int16_t[:] masked_buffer
        cdef const np.int16_t[:] labels_buffer
        cdef np.ndarray[np.int64_t, ndim=1] indices
        
        cdef np.ndarray[np.int16_t, ndim=1] ids
        cdef np.ndarray[np.int16_t, ndim=1] masked
        cdef np.ndarray[np.int16_t, ndim=1] labels 
        cdef np.ndarray[np.int8_t, ndim=1] attention_mask
        seq_length = self.seq_length
        b_seq_length = self.seq_length * 2
        data_path = self.data_path
        f1_temp = io.BytesIO()
        f2_temp = io.BytesIO()
        f3_temp = io.BytesIO()
        f4_temp = io.BytesIO()
        
        
        with open(data_path+"data_original", 'rb') as f1, \
             open(data_path+"data_masked", 'rb') as f2, \
             open(data_path+"data_labels", 'rb') as f3, \
             open(data_path+"data_original_cooked", 'wb') as f1_real, \
             open(data_path+"data_masked_cooked", 'wb') as f2_real, \
             open(data_path+"data_labels_cooked", 'wb') as f3_real, \
             open(data_path+"data_attn_mask_cooked", 'wb') as f4_real \
        :
            steps = self.length // mult
            with tqdm(total=steps) as pbar:
                while True:
                    max_bytes = b_seq_length*mult
                    ids_b = f1.read(max_bytes)
                    masked_b = f2.read(max_bytes)
                    labels_b = f3.read(max_bytes)
                    if len(ids_b) == 0 :
                        break
                    for j in range(len(ids_b) // b_seq_length):


                        shift = 0 
                        start = (b_seq_length*j + shift)

                        b_length = find_first_zero_from_bytes(ids_b, start, b_seq_length - shift)
                        if b_length > b_seq_length - 4:
                            b_length = b_seq_length - 4
                        length = b_length // 2
                        
                        
                        ids_buffer = np.frombuffer(ids_b, dtype=np.int16, count=length, offset=start)
                        masked_buffer = np.frombuffer(masked_b, dtype=np.int16, count=length, offset=start)
                        labels_buffer = np.frombuffer(labels_b, dtype=np.int16, count=length, offset=start)

                        end = start + b_length

                        ids = np.zeros(seq_length, dtype=np.int16)
                        masked = np.zeros(seq_length, dtype=np.int16)
                        labels = np.full(seq_length, NULL_LABEL, dtype=np.int16)
                        attention_mask = np.zeros(seq_length, dtype=np.int8)

                        ids[0] = CLS
                        masked[0] = CLS
                        for j in range(length):
                            ids[j+1] = ids_buffer[j]
                            masked[j+1] = masked_buffer[j]
                            labels[j+1] = labels_buffer[j]
                            attention_mask[j] = 1
                        ids[length+1] = SEP
                        masked[length+1] = SEP
                        attention_mask[j+1] = 1
                        attention_mask[j+2] = 1

                        f1_temp.write(ids.tobytes())
                        f2_temp.write(masked.tobytes())
                        f3_temp.write(labels.tobytes())
                        f4_temp.write(attention_mask.tobytes())

                    pbar.update(1)
                    
            indices = np.random.permutation(self.length)
            with tqdm(total=steps) as pbar:
                for j in range(self.length):
                    start = indices[j]*b_seq_length
                    f1_temp.seek(start)
                    f1_real.write(f1_temp.read(b_seq_length))
                    f2_temp.seek(start)
                    f2_real.write(f2_temp.read(b_seq_length))
                    f3_temp.seek(start)
                    f3_real.write(f3_temp.read(b_seq_length))
                    f4_temp.seek(start // 2)
                    f4_real.write(f4_temp.read(seq_length))

                    pbar.update(1)
                    
                
                
    def __getbatch__(self, long i, long size):
        cdef long shift, start, seq_length = self.seq_length, b_length, end, length
        cdef unsigned int j = 0
        cdef np.ndarray[np.int16_t, ndim=2] ids_buffer
        cdef np.ndarray[np.int16_t, ndim=2] masked_buffer
        cdef np.ndarray[np.int16_t, ndim=2] labels_buffer
        cdef np.ndarray[np.int8_t, ndim=2] attention_mask_buffer
        assert self.cooked
        if self.ids_bin_buffer is None:
            cooked = self.cooked
            data_path = self.data_path
            path = data_path+"data_original" + ('_cooked' if cooked else '')
            self.ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.ids_bin_buffer = self.ids_bin_buffer_mmap #memoryview(self.ids_bin_buffer_mmap)

            path = data_path+"data_masked" + ('_cooked' if cooked else '')
            self.masked_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.masked_bin_buffer = self.masked_bin_buffer_mmap #memoryview(self.masked_bin_buffer_mmap)

            path = data_path+"data_labels" + ('_cooked' if cooked else '')
            self.labels_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.labels_bin_buffer = self.labels_bin_buffer_mmap #memoryview(self.labels_bin_buffer_mmap)
        
            if cooked:
                path = data_path+"data_attn_mask_cooked"
                self.attention_mask_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
                self.attention_mask_bin_buffer = self.attention_mask_bin_buffer_mmap # memoryview(self.attention_mask_bin_buffer_mmap)
                
        if self.eval:
            i += self.actual_length - self.eval_num_samples
            
        if self.cooked:
            start = seq_length*i*2
            shape = (size,self.seq_length)
            
            ids_buffer = np.frombuffer(self.ids_bin_buffer, dtype=np.int16, count=seq_length*size, offset=start).reshape(shape)
            masked_buffer = np.frombuffer(self.masked_bin_buffer, dtype=np.int16, count=seq_length*size, offset=start).reshape(shape)
            labels_buffer = np.frombuffer(self.labels_bin_buffer, dtype=np.int16, count=seq_length*size, offset=start).reshape(shape)
            attention_mask_buffer = np.frombuffer(self.attention_mask_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)

            return (
                torch.LongTensor(masked_buffer), 
                torch.LongTensor(labels_buffer), 
                torch.LongTensor(attention_mask_buffer), 
                torch.LongTensor(ids_buffer)
            )
            
    
    def __getitem__(self, long i):
        cdef long shift, start, seq_length = self.seq_length, b_length, end, length
        cdef unsigned int j = 0
        cdef np.ndarray[np.int32_t, ndim=1] ids
        cdef np.ndarray[np.int32_t, ndim=1] masked
        cdef np.ndarray[np.int32_t, ndim=1] labels
        cdef np.ndarray[np.int32_t, ndim=1] attention_mask
        cdef const np.int16_t[:] ids_buffer
        cdef const np.int16_t[:] masked_buffer
        cdef const np.int16_t[:] labels_buffer
        cdef const np.int8_t[:] attention_mask_buffer
        if self.ids_bin_buffer is None:
            cooked = self.cooked
            data_path = self.data_path
            path = data_path+"data_original" + ('_cooked' if cooked else '')
            self.ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.ids_bin_buffer = self.ids_bin_buffer_mmap #memoryview(self.ids_bin_buffer_mmap)

            path = data_path+"data_masked" + ('_cooked' if cooked else '')
            self.masked_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.masked_bin_buffer = self.masked_bin_buffer_mmap #memoryview(self.masked_bin_buffer_mmap)

            path = data_path+"data_labels" + ('_cooked' if cooked else '')
            self.labels_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.labels_bin_buffer = self.labels_bin_buffer_mmap #memoryview(self.labels_bin_buffer_mmap)
        
            if cooked:
                path = data_path+"data_attn_mask_cooked"
                self.attention_mask_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
                self.attention_mask_bin_buffer = self.attention_mask_bin_buffer_mmap # memoryview(self.attention_mask_bin_buffer_mmap)
                

        if self.eval:
            i += self.actual_length - self.eval_num_samples
            
        if self.cooked:
            start = seq_length*i*2
            
            ids_buffer = np.frombuffer(self.ids_bin_buffer, dtype=np.int16, count=seq_length, offset=start)
            masked_buffer = np.frombuffer(self.masked_bin_buffer, dtype=np.int16, count=seq_length, offset=start)
            labels_buffer = np.frombuffer(self.labels_bin_buffer, dtype=np.int16, count=seq_length, offset=start)
            attention_mask_buffer = np.frombuffer(self.attention_mask_bin_buffer, dtype=np.int8, count=seq_length, offset=start // 2)

            return torch.LongTensor(masked_buffer), torch.LongTensor(labels_buffer), torch.LongTensor(attention_mask_buffer), torch.LongTensor(ids_buffer)
            
        else:
            ids = np.zeros(seq_length, dtype=np.int32)
            masked = np.zeros(seq_length, dtype=np.int32)
            labels = np.full(seq_length, NULL_LABEL, dtype=np.int32)
            attention_mask = np.zeros(seq_length, dtype=np.int32)

            shift = 0
            start = (seq_length*i + shift)*2

            b_length = find_first_zero_from_bytes(self.ids_bin_buffer, start, seq_length*2 - shift*2)
            if b_length > seq_length*2 - 4:
                b_length = seq_length*2 - 4
            length = b_length // 2
            end = start + b_length

            ids_buffer = np.frombuffer(self.ids_bin_buffer, dtype=np.int16, count=length, offset=start)
            masked_buffer = np.frombuffer(self.masked_bin_buffer, dtype=np.int16, count=length, offset=start)
            labels_buffer = np.frombuffer(self.labels_bin_buffer, dtype=np.int16, count=length, offset=start)

            ids[0] = CLS
            masked[0] = CLS
            for j in range(length):
                ids[j+1] = ids_buffer[j]
                masked[j+1] = masked_buffer[j]
                labels[j+1] = labels_buffer[j]
                attention_mask[j] = 1
            ids[length+1] = SEP
            masked[length+1] = SEP
            attention_mask[j+1] = 1
            attention_mask[j+2] = 1
        return torch.LongTensor(masked), torch.LongTensor(labels), torch.LongTensor(attention_mask), torch.LongTensor(ids)
    


cdef bint _too_many_repeat(char* chars, 
                      size_t length) :
    cdef:
        unsigned char lb
        size_t cursor = 0
        unsigned char size = 0
        int code
        int a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0, j=0, k=0, accum=0

    while True:
        if cursor >= length:
            lb = 32
            size = 1
            code = lb
        else:
            lb = chars[cursor]

            if (lb - 0xc2) > (0xf4-0xc2):
                return -1

            if lb < 0x80:
                size = 1
                code = lb
                
            elif lb < 0xE0:
                size = 2
                if cursor + size > length:
                    return -1
                
                code = ((lb & 0x1f)<<6) | (chars[cursor+1] & 0x3f)
                
                
            elif lb < 0xF0:
                size = 3
                if cursor + size > length:
                    return -1
                
                code = ((lb & 0xf)<<12) | ((chars[cursor+1] & 0x3f)<<6) | (chars[cursor+2] & 0x3f);
                
            elif ( lb & 0xF8 ) == 0xF0:
                size = 4
                if cursor + size > length:
                    return -1
                
                code = ((lb & 7)<<18) | ((chars[cursor+1] & 0x3f)<<12) | ((chars[cursor+2] & 0x3f)<<6) | (chars[cursor+3] & 0x3f)
                
            else:
                return -2
            
        if code < 33 or code == 160:
            pass
        else:
            #print(code, accum)
            if code == a or code == b or code == c or code == d or code == e or code == f or code == g:
                accum += 1
                if accum > 16:
                    return 1
            elif code == a or code == b or code == c or code == d or code == e or code == f or code == g or code == h or code == i or code == j or code == k:
                accum += 1
                if accum > 22:
                    return 1

            else:
                a = b
                b = c
                c = d
                d = e
                e = f
                f = g
                g = h
                h = i
                i = j
                j = k
                k = code
                accum = 0

        if cursor >= length:
            break
        cursor += size

    return 0
def too_many_repeat(str text):
    cdef:
        bytes b_text = text.encode()
        
    return _too_many_repeat(b_text, len(b_text))

