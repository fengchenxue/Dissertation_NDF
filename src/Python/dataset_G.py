from re import X
import numpy as np
from scipy.stats import alpha, beta
from scipy.interpolate import interp1d

import ndf_py


def generate_GGX(theta_vals):
    alpha = np.random.uniform(0.05, 0.5)
    
    tan_theta_sq = np.tan(theta_vals)**2
    cos_theta = np.clip(np.cos(theta_vals), 1e-6, 1.0)
    D = alpha**2 / (np.pi * cos_theta**4 * (alpha**2 + tan_theta_sq)**2)
    
    D[-1] = 0.0
    return D

def generate_Beckmann(theta_vals):
    alpha = np.random.uniform(0.05, 0.5)
    
    tan_theta_sq = np.tan(theta_vals)**2
    cos_theta = np.clip(np.cos(theta_vals), 1e-6, 1.0)
    D = np.exp(-tan_theta_sq / alpha**2) / (np.pi * alpha**2 * cos_theta**4)
    
    D[-1] = 0.0
    return D


def generate_student_t(theta_vals):
    alpha = np.random.uniform(0.05, 0.5)
    gamma=np.random.uniform(1.2, 6.0)

    epsilon=1e-6
    cos_theta = np.clip(np.cos(theta_vals), epsilon, 1.0)
    tan_theta_sq = np.tan(theta_vals) ** 2
    D = ((gamma - 1) / (np.pi * alpha**2 * cos_theta**4)) * (1 + tan_theta_sq / alpha**2) ** (-gamma)
    
    D[-1] = 0.0
    return D

def generate_sample(num_control=128):
    theta_vals = np.linspace(0, np.pi / 2, num_control)
    r1=np.random.rand()
    r2=np.random.rand()

    Dx = None
    Dy = None

    if r1<0.5:
        #GGX
        Dx= generate_GGX(theta_vals)
    elif r1<0.75:
        #Beckmann
        Dx = generate_Beckmann(theta_vals)
    else:
        #Student-t
        Dx = generate_student_t(theta_vals)

    if r2<0.5:
        #GGX
        Dy= generate_GGX(theta_vals)
    elif r2<0.75:
        #Beckmann
        Dy = generate_Beckmann(theta_vals)
    else:
        #Student-t
        Dy = generate_student_t(theta_vals)
     
    theta= np.random.rand() * np.pi / 2
    phi= np.random.rand() * np.pi * 2

    w = ndf_py.Vec3f(np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta))
    # in fact, G1 function doesn't use wh
    wh = ndf_py.Vec3f(0.0, 0.0, 1.0)
    
    ndf = ndf_py.PiecewiseLinearNDF(Dx.tolist(), Dy.tolist())
    g1= ndf.G1(w,wh)
    
    x_feature = np.concatenate((
        [np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)],
        Dx,
        Dy
    )).astype(np.float32)

    return x_feature, g1

def generate_dataset_G (N=1000, num_control=128):
    x_data= np.zeros((N, 4+2*num_control),dtype=np.float32)
    y_data= np.zeros(N,dtype=np.float32)

    for i in range(N):
        x_feature, g1 = generate_sample(num_control)
        x_data[i] = x_feature
        y_data[i] = g1
        
    return x_data, y_data

if __name__ == "__main__":
    data_count= 100000
    x_data,y_data=generate_dataset_G(data_count, 128)
    
    file_count = data_count // 1000
    np.savez_compressed(f"data/dataset/dataset_G_{file_count}k.npz", x=x_data, y=y_data)