import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
from scipy.fftpack import dct

from utils import *


def zigzag_scan(matrix):
    ''' return matrix as an array by zigzaging indices ''' 
    r, c = 0, 0
    r_max, c_max = matrix.shape

    zz_array = [matrix[r, c]]

    while c < c_max and r < r_max:
        # if not on last column, move right
        if c != c_max-1:
            c += 1
            zz_array.append(matrix[r, c])
        # if last column, move down unless on corners
        else:
            # if r != 0:
            r += 1
            zz_array.append(matrix[r, c])

        # go diagonally down
        while c > 0 and r < r_max-1:
            r += 1
            c -= 1
            zz_array.append(matrix[r, c])

        if c == c_max-1 and r == r_max-1:
            break

        # if on last row, go right
        if r == r_max-1:
            c += 1
            zz_array.append(matrix[r, c])
        else:
            r += 1
            zz_array.append(matrix[r, c])

        if c == c_max-1 and r == r_max-1:
            break

        # go diagonally up 
        while c < c_max-1 and r > 0:
            r -= 1
            c += 1
            zz_array.append(matrix[r, c])

    return np.array(zz_array)

def quantization(matrix, quality):
    ''' return quantized matrix ''' 
    Q = get_Q(quality)
    r, c = matrix.shape
    qmatrix = matrix/Q[:r,:c]
    return np.round(qmatrix)

def calculate_fd(qarray, base):
    ''' calculate first digit using Eqn '''
    fd_list = []
    for i in range(len(qarray)):
        qdct = np.abs(qarray[i])
        fd = np.floor(qdct/(base**np.floor(np.log(qdct)/np.log(base))))
        fd_list.append(fd)
    return np.array(fd_list)

def image_fd_pmd(image, freq=0):
    r, c = image.shape
    n_blocks = int(np.floor(r/8)*np.floor(c/8))
    fd_matrix = np.zeros((n_blocks, 64))
    pmf = np.zeros(9)
    k = 0
    for i in range(int(np.floor(r/8))):
        for j in range(int(np.floor(c/8))):
            subsection = image[i*8:i*8+8, j*8:j*8+8]
            dct2d = dct(dct(subsection.T).T)
            qmatrix = quantization(dct2d, 90)
            qarray = zigzag_scan(qmatrix)
            fd_array = []
            for item in qarray:
                fd = str(item)[0]
                if fd == '-':
                    fd = str(item)[1]
                fd_array.append(int(fd))
            fd_matrix[k, :] = np.array(fd_array)
            # fd_matrix[k, :] = calculate_fd(qarray)           # list of FD from 8x8
            k += 1

    print(fd_matrix)
    # curr_freq = fd_matrix[:, freq]
    # print(curr_freq.shape)
    # removednan = curr_freq[~np.isnan(curr_freq)]
    # print(len(removednan))

    # print(fd_matrix.shape)
    # fd_matrix = fd_matrix.reshape(np.size(fd_matrix))
    # print(fd_matrix.shape)
    # fd_matrix = fd_matrix[~np.isnan(fd_matrix)]
    # print(fd_matrix.shape)

    vals, counts = np.unique(fd_matrix, return_counts=True)
    print(vals)
    print(counts)
    pmf[vals.astype(int)-1] += counts

    d = np.linspace(1,9,9)
    plt.plot(d, pmf/np.size(fd_matrix))
    plt.plot(d, np.log10(1+(1/d)))
    plt.show()
    return fd, pmf

if __name__ == "__main__":
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
    ])

    dataset = datasets.ImageFolder('data2', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    image, label = next(iter(dataloader))
    image = np.squeeze(np.array(image))
    all_fd, pmf = image_fd_pmd(image)

    # plt.hist(all_fd, bins = [1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    # plt.title("histogram") 
    # plt.show()
            
        
        

