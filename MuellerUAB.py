import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

##################
### READ FILES ###
##################

def read_file(file_name):
    
    """Reads the polarimetric information contained in a .mat file.
    
    Parameters:
        file_name ('str'): Name of the file to read
    
    Returns:
        MM: Mueller matrix of all N pixels in a (4,4,N) shape
        M00: Intensity of the image
        Nx: Number of columns (of pixels) in the image
        Ny: Number of rows (of pixels) in the image
        Feasibility (np.array): Array with the physical feasibility of each pixel 
                               (0 means feasible, 1 means M00=0 i.e not feasible) 
    """
    
    print('Reading .mat file...')
    mat_file = file_name # Name of the file
    mat_contents = sio.loadmat(mat_file)
    sorted(mat_contents.keys())

    # Extract all the Mueller matrix elements:
    M00=mat_contents['M00']; M01=mat_contents['M01']; M02=mat_contents['M02']; M03=mat_contents['M03']; 
    M10=mat_contents['M10']; M11=mat_contents['M11']; M12=mat_contents['M12']; M13=mat_contents['M13']; 
    M20=mat_contents['M20']; M21=mat_contents['M21']; M22=mat_contents['M22']; M23=mat_contents['M23']; 
    M30=mat_contents['M30']; M31=mat_contents['M31']; M32=mat_contents['M32']; M33=mat_contents['M33']; 
    
    Nx = len(M00); Ny = len(M00[0])
    
    print('.mat file loaded')
    
    MM_img = np.array([[M00, M01, M02, M03], # Create the Mueller matrix 
                       [M10, M11, M12, M13], # of the whole image
                       [M20, M21, M22, M23], 
                       [M30, M31, M32, M33]])
    
    N = Nx*Ny # Number of pixels
    MM = MM_img.reshape(4,4,N) # Reshape in order to have the MM of each pixel
    
    # Physical feasibility
    M00 = np.ravel(M00) # Reshape M00 image into a 1D vector
    Feasibility = np.zeros((N,), dtype=int) # Initialize array assuming all pixels are feasible
    
    epsilon = 1/4096 # Condition of 0 intensity pixel
    NotFeasible = np.where(M00 < epsilon) # Array with the index of 0 intensity pixels
    Feasibility[NotFeasible] = 1 # Update feasibility of each pixel
    
    if len(NotFeasible[0]) == 0: # Give info to the user
        print('All pixels have appropriate intensity')
        
    else:
        print('Some pixels have 0 intensity')
    
    return MM, M00, Nx, Ny, Feasibility


def read_bytesfile(file_name):
    
    """Reads the MM's elements contained in a bytes file 
    (the old-fashioned file format that the polarimeter used to output).
    
    Parameters:
        file_name ('str'): Name of the file to read
        
    Returns:
        MM: Mueller matrix of all N pixels in a (4,4,N) shape
        M00: Intensity of the image
        Nx: Number of columns (of pixels) in the image
        Ny: Number of rows (of pixels) in the image
        Feasibility (np.array): Array with the physical feasibility of each pixel 
                               (0 means feasible, 1 means M00=0 i.e. not feasible)
    """
    
    print('Reading bytes file...')
    
    with open(file_name, 'rb') as file:
        Ny = np.fromfile(file, np.int32, 1)[0]  # Extract dimensions of the image
        Nx = np.fromfile(file, np.int32, 1)[0] 
        
        M00 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)  # Extract all
        M01 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)  # the MM elements
        M02 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M03 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        
        M10 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M11 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M12 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M13 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        
        M20 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M21 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M22 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M23 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        
        M30 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M31 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M32 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        M33 = np.fromfile(file, float, Nx*Ny).reshape(Nx,Ny)
        
    print('bytes file loaded')
    
    MM_img = np.array([[M00, M01, M02, M03],  # Create the Mueller matrix
                       [M10, M11, M12, M13],  # of the whole image
                       [M20, M21, M22, M23], 
                       [M30, M31, M32, M33]])
    
    N = Nx*Ny  # Number of pixels
    MM = MM_img.reshape(4,4,N)  # Reshape in order to have the MM of each pixel
    
    # Physical feasibility
    M00 = np.ravel(M00)  # Reshape M00 image into a 1D vector
    Feasibility = np.zeros((N,), dtype=int)  # Initialize array assuming all pixels are feasible
    epsilon = 1/4096 # Condition of 0 intensity pixel
    NotFeasible = np.where(M00 < epsilon)  # Array with the index of 0 intensity pixels
    Feasibility[NotFeasible] = 1  # Update feasibility of each pixel
    
    if len(NotFeasible[0]) == 0:  # Give info to the user
        print('All pixels have appropriate intensity')
        
    else:
        print('Some pixels have 0 intensity')
    
    return MM, M00, Nx, Ny, Feasibility

##################################
####  MM BASIC MANIPULATIONS  ####
##################################

def PhysicalFeasibility_filter(M, Feasibility):
    
    """Converts a given Mueller matrix into a physically feasible one.
    Before filtering, MM should be normalized with .normalize().

    Parameters: 
        Feasibility (1D array): Outputted when using .normalize() function.
        
    Returns:
        MM_filt (4,4,N) array: Physically feasible Mueller matrix
        H_filt (4,4,N) array: Filtered Covariance matrix
        Feasibility (1D array): Updated with not feasible pixels (0 -> feasible
                                                                  1 -> M00=0
                                                                  2 -> eigenvalue<0)
    """

    # Filter not physically feasible pixels (negative eigenvalues)
    H = covariance_matrix(M)
    H = np.moveaxis(H, -1, 0)  # Reshape to (N,4,4)
    eigenvalues_H, U = np.linalg.eig(H)  # Eigenvalues & Eigenvectors matrix
    U_dag = np.conjugate(np.transpose(U, axes=[0,2,1]))  # Dagger (transpose & conjugate) of eigenvectors matrix U

    NotFeasible = np.where(eigenvalues_H < 0)  # Not feasible pixels (lambda<0)
    eigenvalues_H[NotFeasible] = 0  # If eigenvalue is negative, make it 0
    Feasibility[NotFeasible[0]] = 2  # pixels with <0 eigenvalues, mark them with number 2

    # Recompute H and M with the filtered eigenvalues
    diag_lambdas, H_filt = np.zeros_like(H), np.zeros_like(H)  # Initialize
    N = np.size(H, axis=0)  # nº of pixels

    for i in range(0,N):  # Puts the eigenvalues of each pixel in a diagonal matrix
        diag_lambdas[i,:,:] = np.diag(eigenvalues_H[i,:])

    H_filt = U @ diag_lambdas @ U_dag  # Recompose H with the filtered eigenvalues
    H = H_filt  # Update H with filtered values
    H = np.moveaxis(H, 0, -1)  # (4,4,N) shape

    MM_filt = np.array([[H[0,0,:] + H[1,1,:] + H[2,2,:] + H[3,3,:],  # J.J. Gil, Polarized Light (pag. 172, eq 5.18)
                         H[0,0,:] - H[1,1,:] + H[2,2,:] - H[3,3,:],
                         H[0,1,:] + H[1,0,:] + H[2,3,:] + H[3,2,:],
                    -1j*(H[0,1,:] - H[1,0,:] + H[2,3,:] - H[3,2,:])],
                  
                        [H[0,0,:] + H[1,1,:] - H[2,2,:] - H[3,3,:], 
                         H[0,0,:] - H[1,1,:] - H[2,2,:] + H[3,3,:],
                         H[0,1,:] + H[1,0,:] - H[2,3,:] - H[3,2,:],
                    -1j*(H[0,1,:] - H[1,0,:] - H[2,3,:] + H[3,2,:])],
                  
                        [H[0,2,:] + H[2,0,:] + H[1,3,:] + H[3,1,:], 
                         H[0,2,:] + H[2,0,:] - H[1,3,:] - H[3,1,:],
                         H[0,3,:] + H[3,0,:] + H[1,2,:] + H[2,1,:],
                    -1j*(H[0,3,:] - H[3,0,:] - H[1,2,:] + H[2,1,:])],
                  
                    [1j*(H[0,2,:] - H[2,0,:] + H[1,3,:] - H[3,1,:]), 
                     1j*(H[0,2,:] - H[2,0,:] - H[1,3,:] + H[3,1,:]),
                     1j*(H[0,3,:] - H[3,0,:] + H[1,2,:] - H[2,1,:]),
                         H[0,3,:] + H[3,0,:] - H[1,2,:] - H[2,1,:]]])

    MM_filt = MM_filt.real  # Get rid of imaginary part in Mueller matrix

    return MM_filt, H, Feasibility

        
def normalize(M, Feasibility=None):
    
    """Normalizes a given MM object dividing by its total intensity, M00.
    Shape of MM object should be (4,4,N).
        
    Parameters:
        Feasibility (1D array): Array outputted in read_file() function
    
    Returns:
        MM (4,4,N) array: Normalized Mueller matrix
    """
    
    if Feasibility is None:
        M00 = M[0,0,:]
        N = len(M00)
        
        Feasibility = np.zeros((N,), dtype=int)  # Initialize array assuming all pixels are feasible
        Feasible = np.where(Feasibility == 0)  # Search for feasible (and not) pixels
        epsilon = 1/4096  # Condition of 0 intensity pixel
        NotFeasible = np.where(M00 < epsilon)  # Array with the index of NOT feasible pixels
        Feasibility[NotFeasible] = 1  # Update feasibility of each pixel
        
        M[:,:,Feasible] = M[:,:,Feasible] / M00  # Normalizes all physically feasible pixels
        M[:,:,NotFeasible] = 0  # Anulates the whole MM for a not feasible pixel
        
        return M
        
    Feasible = np.where(Feasibility == 0)  # Search for feasible (and not) pixels
    NotFeasible = np.where(Feasibility == 1)
    
    M00 = M[0,0,:]
    M[:,:,Feasible] = M[:,:,Feasible] / M00  # Normalizes all physically feasible pixels
    M[:,:,NotFeasible] = 0  # Anulates the whole MM for a not feasible pixel
    
    return M


def covariance_matrix(M): 
    """Computes the covariance matrix H of a given Mueller matrix.
    
    Parameters: 
        M (4,4,N) array: Mueller matrix
        
    Returns:
        H (4,4,N) array: Covariance matrix of M
    
    """
    
    H = np.array([[M[0,0,:] + M[0,1,:] +     M[1,0,:] + M[1,1,:],  # J.J. Gil, Polarized Light (pag. 171, eq 5.16)
                   M[0,2,:] + M[1,2,:] + 1j*(M[0,3,:] + M[1,3,:]),
                   M[2,0,:] + M[2,1,:] - 1j*(M[3,0,:] + M[3,1,:]),
                   M[2,2,:] + M[3,3,:] + 1j*(M[2,3,:] - M[3,2,:])],
            
                  [M[0,2,:] + M[1,2,:] - 1j*(M[0,3,:] - M[1,3,:]), 
                   M[0,0,:] - M[0,1,:] +     M[1,0,:] - M[1,1,:],
                   M[2,2,:] - M[3,3,:] - 1j*(M[2,3,:] + M[3,2,:]),
                   M[2,0,:] - M[2,1,:] - 1j*(M[3,0,:] - M[3,1,:])],
            
                  [M[2,0,:] + M[2,1,:] + 1j*(M[3,0,:] + M[3,1,:]), 
                   M[2,2,:] - M[3,3,:] + 1j*(M[2,3,:] + M[3,2,:]),
                   M[0,0,:] + M[0,1,:] -     M[1,0,:] - M[1,1,:],
                   M[0,2,:] - M[1,2,:] + 1j*(M[0,3,:] - M[1,3,:])],
          
                  [M[2,2,:] + M[3,3,:] - 1j*(M[2,3,:] - M[3,2,:]), 
                   M[2,0,:] - M[2,1,:] + 1j*(M[3,0,:] - M[3,1,:]),
                   M[0,2,:] - M[1,2,:] - 1j*(M[0,3,:] - M[1,3,:]),
                   M[0,0,:] - M[0,1,:] -     M[1,0,:] + M[1,1,:]]])
    
    N = len(M[0,0])  # Number of pixels
    a = np.repeat(0.25, N)  # Make an array of 0.25's
    H = np.moveaxis(H, -1, 0)  # (N,4,4) shape
    H = np.einsum('i,ijk->ijk', a, H)  # Multiply 0.25*H
    
    H = np.moveaxis(H, 0, -1)  # (4,4,N) reshape back to original
    
    return H


def coherency_matrix(M):
    """Computes the coherency matrix C of a given Mueller matrix.
    
    Parameters: 
        M (4,4,N) array: Mueller matrix
        
    Returns:
        C (4,4,N) array: Coherency matrix of M
    
    """
    
    C = np.array([[M[0,0,:] + M[1,1,:] +     M[2,2,:] + M[3,3,:],  # J.J. Gil, Polarized Light (pag. 174, eq 5.23)
                   M[0,1,:] + M[1,0,:] - 1j*(M[2,3,:] - M[3,2,:]),
                   M[0,2,:] + M[2,0,:] + 1j*(M[1,3,:] - M[3,1,:]),
                   M[0,3,:] + M[3,0,:] - 1j*(M[1,2,:] - M[2,1,:])],
            
                  [M[0,1,:] + M[1,0,:] + 1j*(M[2,3,:] - M[3,2,:]), 
                   M[0,0,:] + M[1,1,:] -     M[2,2,:] - M[3,3,:],
                   M[1,2,:] + M[2,1,:] + 1j*(M[0,3,:] - M[3,0,:]),
                   M[1,3,:] + M[3,1,:] - 1j*(M[0,2,:] - M[2,0,:])],
        
                  [M[0,2,:] + M[2,0,:] - 1j*(M[1,3,:] - M[3,1,:]), 
                   M[1,2,:] + M[2,1,:] - 1j*(M[0,3,:] - M[3,0,:]),
                   M[0,0,:] - M[1,1,:] +     M[2,2,:] - M[3,3,:],
                   M[2,3,:] + M[3,2,:] + 1j*(M[0,1,:] - M[1,0,:])],
          
                  [M[0,3,:] + M[3,0,:] + 1j*(M[1,2,:] - M[2,1,:]), 
                   M[1,3,:] + M[3,1,:] + 1j*(M[0,2,:] - M[2,0,:]),
                   M[2,3,:] + M[3,2,:] - 1j*(M[0,1,:] - M[1,0,:]),
                   M[0,0,:] - M[1,1,:] -     M[2,2,:] + M[3,3,:]]])
    
    N = len(M[0,0])  # Number of pixels
    a = np.repeat(0.25, N)  # Make an array of 0.25's
    C = np.moveaxis(C, -1, 0)  # (N,4,4) shape
    C = np.einsum('i,ijk->ijk', a, C)  # Multiply 0.25*C
    
    C = np.moveaxis(C, 0, -1)  # (4,4,N) reshape back to original
    
    return C


def identity(Mdim, N):
    """Creates an array of N elements, where each of them is a MxM identity matrix.
        
    Parameters: 
        Mdim (int): Dimensions of the identity matrix.
        N (int): Number of matrices in the array.
    
    Returns:
        I_M (NxMxM array): N identity matrices of size MxM.
    """ 
    shape = (Mdim,Mdim,N)
    I_M = np.zeros(shape)
    idx = np.arange(shape[0])
    I_M[idx, idx, :] = 1
    
    return I_M


def flatten(M):
    
    """Reshapes a (4,4,N) MM object into shape (16,N).
        
    Returns:
        Same MM object reshaped into (16, N).   
    """
    
    N = len(M[0,0,:])
    MM = M.reshape(16,N)
    
    return MM

###################################
####  POLARIMETRIC PARAMETERS  ####
###################################
       
def Polarizance(M):
    
    """Calculates all parameters related to polarizance from a (4x4,N) normalized MM.
        
    Returns:
        P (1xN array): Polarizance
        Pc (1xN array): Circular polarizance
        Pc (1xN array): Linear polarizance
        P_vect (3xN array): Polarizance vector
        Pazi (1xN array): Azimuth angle of the polarizance vector
        Pelip (1xN array): Elipticity angle of the polarizance vector
    """
    
    P_vect = np.array([M[1,0,:], M[2,0,:],M[3,0,:]])
    P = np.sqrt(P_vect[0,:]**2 + P_vect[1,:]**2 + P_vect[2,:]**2)  
    
    Pl = np.sqrt(P_vect[0,:]**2 + P_vect[1,:]**2)  # Pag 204 J.J. Gil Polarized Light
    Pc = P_vect[2,:]
    
    Pazi = 0.5 * np.arctan(P_vect[1,:]/P_vect[0,:])/np.pi*180  # Wikipedia: https://en.wikipedia.org/wiki/Stokes_parameters
    Pelip = 0.5 * np.arctan(Pc/Pl)/np.pi*180
    
    return P_vect, P, Pc, Pl, Pazi, Pelip


def Diattenuation(M):
    """Calculates all parameters related to diattenuation from a (4x4,N) normalized MM.
        
    Returns:
        D (1xN array): Diattenuation
        Dc (1xN array): Circular Diattenuation
        Dl (1xN array): Linear diattenuation
        D_vect (3xN array): Diattenuation vector
        Dazi (1xN array): Azimuth angle of the diattenuation vector
        Delip (1xN array): Elipticity angle of the diattenuation vector
    """
    
    D_vect = np.array([M[0,1,:], M[0,2,:],M[0,3,:]])
    D = np.sqrt(D_vect[0,:]**2 + D_vect[1,:]**2 + D_vect[2,:]**2)  
    
    Dl = np.sqrt(D_vect[0,:]**2 + D_vect[1,:]**2)  # Pag 204 J.J. Gil Polarized Light
    Dc = D_vect[2,:]
    
    Dazi = 0.5 * np.arctan(D_vect[1,:]/D_vect[0,:])/np.pi*180  # Wikipedia: https://en.wikipedia.org/wiki/Stokes_parameters
    Delip = 0.5 * np.arctan(Dc/Dl)/np.pi*180
    
    return D_vect, D, Dc, Dl, Dazi, Delip


def Retardance(M, Mr=None):
    """Calculates all parameters related to retardance, 
    once the MM has been Lu-Chipman decomposed.
        
    Parameters:
        Mr (4x4xN array): Retarder's MM outputted in .LuChipman_decomposition()
        
    Returns:
        R (1xN array): Retardance
        delta (1xN array): Linear retardance
        Psi (1xN array): Retarder's rotation
    """
    
    if Mr is None:
        Mr = LuChipman_decomposition(M)[1]  # If no Mr is give, make the Lu-Chipman decomposition
        
        mr = Mr[1:,1:,:]  # Small 3x3 sub-matrix of Mr
        tr_mr = np.matrix.trace(mr)  # Trace of mr
       
        R = np.arccos(0.5*(tr_mr - 1))/np.pi*180  # Global retardance
        
        delta = np.arccos(np.sqrt((Mr[1,1,:] + Mr[2,2,:])**2 +  # Linear retardance
                                  (Mr[2,1,:] - Mr[1,2,:])**2) - 1)/np.pi*180 

        Psi = np.arctan((Mr[2,1,:] - Mr[1,2,:]) /   # Retarder's rotation
                         (Mr[1,1,:] + Mr[2,2,:]))/np.pi*180
        
        return R, delta, Psi
        
    mr = Mr[1:,1:,:] # Small 3x3 sub-matrix of Mr
    tr_mr = np.matrix.trace(mr)  # Trace of mr
   
    R = np.arccos(0.5*(tr_mr - 1))/np.pi*180  # Global retardance
    
    delta = np.arccos(np.sqrt((Mr[1,1,:] + Mr[2,2,:])**2  # Linear retardance
                              + (Mr[2,1,:] - Mr[1,2,:])**2) - 1)/np.pi*180 

    Psi = np.arctan((Mr[2,1,:] - Mr[1,2,:]) /   # Retarder's rotation
                     (Mr[1,1,:] + Mr[2,2,:]))/np.pi*180
    
    return R, delta, Psi
    

def IPPs(M, H_filt=None, lambdas=False):
    
    """Computes IPPs and H eigenvalues. M should be already filtered and normalized.
        
    Parameters:
        H_filt (4x4xN array): Covariance matrix filtered. Outputted in 
                              PhysicalFeasibility_filter() function.
        lambdas (bool): If True, also returns the eigenvalues of H.
        
    Returns:
        (P1, P2, P3) tuple: Polarimetric purity indices
        (lambda1, lambda2, lambda3, lambda4) tuple: Eigenvalues of the filtered H
    """

    if H_filt is None: # If no covariance matrix is given, compute it
        
        H = covariance_matrix(M)
        H = np.moveaxis(H, -1, 0)  # Reshape to (N,4,4)
        eigenvalues_H = np.linalg.eig(H)[0]
        eigenvalues_H = np.real(eigenvalues_H)  # Get rid of imaginary part
        eigenvalues_H = -np.sort(-eigenvalues_H, axis = 1)  # Order manually (decreasing order)
        H_real = H.real  # Make elements of H real to calculate its trace
        trH = H_real.trace(axis1=1, axis2=2)        
        
        lambda0 = eigenvalues_H[:,0]
        lambda1 = eigenvalues_H[:,1]
        lambda2 = eigenvalues_H[:,2]
        lambda3 = eigenvalues_H[:,3]
        
        P1 = (lambda0 - lambda1)/trH
        P2 = (lambda0 + lambda1 - 2*lambda2)/trH
        P3 = (lambda0 + lambda1 + lambda2 - 3*lambda3)/trH
        
        if lambdas==False:
            return P1, P2, P3
        
        else:
            return P1, P2, P3, lambda0, lambda1, lambda2, lambda3
    # TODO: No funciona cuando le inputs una H
    
    # If H is user input, start from here
    H = H_filt
    H = np.moveaxis(H, -1, 0)  # Reshape to (N,4,4)
    eigenvalues_H = np.linalg.eig(H)[0]
    eigenvalues_H = np.real(eigenvalues_H)  # Get rid of imaginary part
    eigenvalues_H = -np.sort(-eigenvalues_H, axis = 1)  # Order manually (decreasing order)
    H_real = H_filt.real  # Make elements of H real to calculate its trace
    trH = H_real.trace(axis1=1, axis2=2)

    lambda0 = eigenvalues_H[:,0]
    lambda1 = eigenvalues_H[:,1]
    lambda2 = eigenvalues_H[:,2]
    lambda3 = eigenvalues_H[:,3]

    P1 = (lambda0 - lambda1)/trH
    P2 = (lambda0 + lambda1 - 2*lambda2)/trH
    P3 = (lambda0 + lambda1 + lambda2 - 3*lambda3)/trH

    if lambdas is False:
        return P1, P2, P3

    else:
        return P1, P2, P3, lambda0, lambda1, lambda2, lambda3
    

def Pdelta(M):
    
    """Computes the parameter Pdelta, defined as the Euclidean distance
    between the given MM and an ideal depolarizer.
        
    Returns:
        Pdelta: array with (4,4,N) shape
    """
    quadratic_sum = np.sum(M**2, axis = 1); 
    quadratic_sum = np.sum(quadratic_sum, axis = 0)
    Pdelta = np.sqrt((quadratic_sum - M[0,0]**2)/3) / M[0,0]
    
    return Pdelta


def Ps(M):

    """Computes the degree of spherical purity (Ps) as done in 
    Polarized Light (JJ Gil), Pag 199, eq. (6.1).
        
    Returns:
        Ps (1xN) array: Degree of spherical purity
    """
    quadratic_sum = np.sum(M[1:,1:,:]**2, axis=1)
    quadratic_sum = np.sum(quadratic_sum, axis=0)
    Ps = np.sqrt(quadratic_sum/3)
    
    return Ps

###########################
###  MM DECOMPOSITIONS  ###
###########################

def LuChipman_decomposition(M):
    """Returns the three pure matrices (Depolarizer, Retarder, Diattenuator)
    from the Lu-Chipman decomposition of M: M = Mp*Mr*Md
        
    Returns:
        (Mp, Mr, Md): Tuple where element [0] is Mp, [1] is Mr and [2] is Md. 
                      Each of them is an array of size (4x4xN).
    """    
    # Extract info from M
    N = len(M[0,0])
    P = M[1:,0,:]  # Polarizance vector
    m = M[1:,1:,:]  # 3x3 sub-matrix of M

    # Initialize some matrices
    Mp, Mr, Md = np.zeros_like(M), np.zeros_like(M), np.zeros_like(M)
    Mp[0,0,:], Mr[0,0,:], Md[0,0,:] = 1,1,1  # M00 = 1 since M should be normalized

    # Start by calculating the diattenuator
    D_vect = np.array([M[0,1,:], M[0,2,:],M[0,3,:]])  # Diattenuation vector
    DT_vect = np.transpose(D_vect, axes=(1,0))  # D^T, transpose vector of D
    D = np.sqrt(D_vect[0,:]**2 + D_vect[1,:]**2 + D_vect[2,:]**2)  # Diattenuation
    
    a = np.sqrt(1-D**2)  # Parameter used later

    I_3 = identity(3,N)  # Creates N identity 3x3 matrices
    I_3 = np.moveaxis(I_3, -1, 0)  # Shape like (N,3,3)
    D_tensor_DT = np.zeros_like(M[1:,1:,:])  # Initialize tensor product of D*D^T
    D_tensor_DT = np.moveaxis(D_tensor_DT, -1, 0)  # Reshape to (N,3,3)
    
    if D.all() != 0:  # Avoid dividing by 0
        
        b = (1-a)/D**2  # Parameter used later

        for i in range(0,N):  # Tensor product D*D^T
            D_tensor_DT[i,:,:] = np.tensordot(D_vect[0:3,i], DT_vect[i,0:3], axes=0)

        aI_3 = np.einsum('i,ijk->ijk', a, I_3)  # a*Identity(3). Multiply i element of a with i matrix of I_3
        bD_tensor_DT = np.einsum('i,ijk->ijk', b, D_tensor_DT)  # Same with b and D*D^T

        mD = aI_3 + bD_tensor_DT  # 3x3 sub-matrix
        
    else:
        b = 0

        mD = I_3  # just if D = 0

    Md[1:,0,:] = D_vect  # Fill with D and D^T vectors
    Md[0,1:,:] = D_vect
    Md[1:,1:,:] = np.moveaxis(mD, 0, -1)  # Fill 3x3 sub-matrix with mD

    # Calculate the inverse of the diattenuator
    M1 = np.zeros_like(Md)  # Create matrices to operate
    M2 = np.zeros_like(Md)
    M1[0,0,:]= 1

    M1[1:,0,:] = -1*D_vect  # Fill with -D and -D^T vectors
    M1[0,1:,:] = -1*D_vect
    M1[1:,1:,:] = np.moveaxis(I_3, 0, -1)  # Fill 3x3 sub-matrix with Identity(3)
    M2[1:,1:,:] = np.moveaxis(D_tensor_DT, 0, -1)  # Fill 3x3 sub-matrix with D*D^T, rest is 0

    M1 = np.moveaxis(M1, -1, 0)
    M2 = np.moveaxis(M2, -1, 0) 

    M1 = np.einsum('i,ijk->ijk', 1/(a**2), M1)  # 1/a^2 * M1
    M2 = np.einsum('i,ijk->ijk', 1/(a**2*(a+1)), M2)

    Md_I = M1 + M2  # Inverse matrix of Md, Md^-1
    Md_I = np.moveaxis(Md_I, 0, -1)

    # Extract Md^I from the total matrix M
    M_ = np.zeros_like(M)  # M_ (M' in the book) is M with the information of Md substracted

    M_ = np.moveaxis(M_, -1, 0)  # Shape like (N,4,4)
    M = np.moveaxis(M, -1, 0)
    Md_I = np.moveaxis(Md_I, -1, 0)

    M_ = M @ Md_I

    # Extract useful info from M_
    m_ = M_[:,1:,1:]  # (Nx3x3) sub-matrix of M_
    m_T = np.transpose(m_, axes=(0,2,1))  # m^T, transposed matrix of m_

    m_mT = m_ @ m_T # Matrix multiplication

    lambdas = np.linalg.eig(m_mT)[0]  # Eigenvalues of m_ * m_^T
    lambdas = -np.sort(-lambdas)  # Decreasing order

    l1 = lambdas[:,0]  # l1>l2>l3
    l2 = lambdas[:,1]
    l3 = lambdas[:,2]

    k1 = np.sqrt(l1) + np.sqrt(l2) + np.sqrt(l3)  # Parameters used later
    k2 = np.sqrt(l1*l2) + np.sqrt(l2*l3) + np.sqrt(l3*l1)
    k3 = np.sqrt(l1*l2*l3)

    # Calculate the depolarizer, Mp
    mD_vect = np.zeros_like(P)  # Initialize multiplication of m * D_vect. It is an array of size (3,N)
    for i in range(0,N):
        mD_vect[:,i] = np.dot(m[:,:,i], D_vect[:,i])  # Multiplication

    Vd = P - mD_vect  # P - m * D_vect
    Pdelta = 1/(a**2) * Vd[:3,:]  # 1/a^2 * (P-m*D)

    k2I_3 = np.einsum('i,ijk->ijk', k2, I_3)  # k2 * I_3
    Mp1 = m_mT + k2I_3
    k1m_mT = np.einsum('i,ijk->ijk', k1, m_mT)
    k3I_3 = np.einsum('i,ijk->ijk', k3, I_3)
    Mp2 = k1m_mT + k3I_3

    sign = np.linalg.slogdet(m_)[0]  # sign of the determinant of m'
    Mp1_I = np.einsum('i,ijk->ijk', sign, np.linalg.inv(Mp1))  # Inverse of Mp1, multiplied by the sign of m'

    mp = Mp1_I @ Mp2  # Nx3x3 sub-matrix of Mp

    Mp[1:,0,:] = Pdelta  # Build Mp
    Mp[1:,1:,:] = np.transpose(mp, (1,2,0))  # Reshape mp to 3x3xN and insert it as sub-matrix
    
    # Calculate the retarder, Mr
    mp_I = np.linalg.inv(mp)  # Inverse of mp
    mr = mp_I @ m_
    
    Mr[1:,1:,:] = np.transpose(mr, (1,2,0))  # Build Mr
    
    return Mp, Mr, Md


def Arrow_decomposition(M):
    
    """Returns the small matrices (mro, mA,mri) from the Arrow decomposition: MM = M_RO*M_A*M_RI.
        
    Parameters: 
        M (4x4xN array): Mueller matrix to decompose.
    
    Returns:
        mro (3x3xN array): small MM of retarders retarder 
        mA (3x3xN array): diag(a1, a2, epsilon*a3)
        mri (3x3xN array): small MM of entrance retarder 
    """ 

    m = M[1:,1:,:]  # Get the small 3x3 (right-under) submatrix from the MM
    N =  len(m[0,0])  # nº of pixels
    m = np.moveaxis(m, -1, 0)  # Move N pixels to first dimension. Now shape of m is Nx3x3

    mro, mA, mri = np.linalg.svd(m)  # svd performs a singular value decomposition of the submatrix m, such that (A = U*S*V')

    mA_diag = np.zeros_like(mro)  # Initialize
    
    for i in range(0,N):  # Puts the values of mA in a diagonal matrix of shape Nx3x3
        mA_diag[i,:,:] = np.diag(mA[i,:])

    det_mri = np.linalg.det(mri)  # Determinants
    det_mro = np.linalg.det(mro)

    mA_diag[:,2,2] = det_mri*det_mro*mA_diag[:,2,2]  # The sign of the third component depends on the determinants of mri and mro
    for i in range(1,3):
        mri[:,i,2] = det_mri*mri[:,i,2]
        mro[:,2,i] = det_mro*mro[:,2,i]
     
    mA = mA_diag
    
    mro = np.moveaxis(mro, 0, -1)  # Reshape back to 3x3xN
    mA = np.moveaxis(mA, 0, -1)
    mri = np.moveaxis(mri, 0, -1)
    
    return mro, mA, mri


#################
####  PLOTS  ####
#################

def plot_Pobs(param, Nx=None, Ny=None, title=None, save=False): 
    
    """Plots a given polarimetric observable.
        
    Parameters: 
        param (1D array): Polarimetric observable to plot
        Nx, Ny (int): Size of the image
        title (str): Title of the plot. Default no title
        save (bool): True if want to save the plot. Default False
    
    Returns:
        Plot of the polarimetric observable
    """ 
    
    plt.rcParams['figure.dpi'] = 300  # Change to get a good quality plot. Otherwise is very blurry
    plt.rcParams['savefig.dpi'] = 300
    
    if Nx and Ny is None:
        n = int(np.sqrt(len(param)))  # Size of the image. Only works for square images
        param = np.reshape(param, (n,n))  # Reshape to the original image size
    
    else:
        nx = Nx; ny = Ny  # Size of the image. In case it's not square
        param = np.reshape(param, (nx,ny))  # Reshape to the original image size
        
    param = param.T  # Transpose

    plt.imshow(param, cmap='gray')
    plt.colorbar()

    if title is not None:  # Title of the image
        plt.title(title)

    if save is True:
        plt.savefig(f"{title}.png")  # Save the plot
    
    plt.show()
    
    
def plot_Mueller(M, Nx=None, Ny=None, title='Mueller matrix', save=False):
    
    """Plots the whole Mueller matrix as: M00 M01 M02 M03 
                                          M10 M11 M12 M13
                                          M20 M21 M22 M23 
                                          M30 M31 M32 M33            
    Parameters: 
        M (4x4xN array): Mueller matrix to plot.
        title (str): Set plot title. Default: 'Mueller matrix'
        save (bool): If True downloads the plot. Default False
    
    Returns:
        Plot of the Mueller matrix
    """ 
    
    if Nx and Ny is None:
        n = int(np.sqrt(len(M)))  # Size of the image. Only works for square images
    
        M00 = M[0,0,:].reshape((n,n)).T; M01 = M[0,1,:].reshape((n,n)).T; M02 = M[0,2,:].reshape((n,n)).T; M03 = M[0,3,:].reshape((n,n)).T
        M10 = M[1,0,:].reshape((n,n)).T; M11 = M[1,1,:].reshape((n,n)).T; M12 = M[1,2,:].reshape((n,n)).T; M13 = M[1,3,:].reshape((n,n)).T
        M20 = M[2,0,:].reshape((n,n)).T; M21 = M[2,1,:].reshape((n,n)).T; M22 = M[2,2,:].reshape((n,n)).T; M23 = M[2,3,:].reshape((n,n)).T
        M30 = M[3,0,:].reshape((n,n)).T; M31 = M[3,1,:].reshape((n,n)).T; M32 = M[3,2,:].reshape((n,n)).T; M33 = M[3,3,:].reshape((n,n)).T
        
    else: 
        nx = Nx  # Size of the image. In case it's not square
        ny = Ny
    
        M00 = M[0,0,:].reshape((nx,ny)).T; M01 = M[0,1,:].reshape((nx,ny)).T; M02 = M[0,2,:].reshape((nx,ny)).T; M03 = M[0,3,:].reshape((nx,ny)).T
        M10 = M[1,0,:].reshape((nx,ny)).T; M11 = M[1,1,:].reshape((nx,ny)).T; M12 = M[1,2,:].reshape((nx,ny)).T; M13 = M[1,3,:].reshape((nx,ny)).T
        M20 = M[2,0,:].reshape((nx,ny)).T; M21 = M[2,1,:].reshape((nx,ny)).T; M22 = M[2,2,:].reshape((nx,ny)).T; M23 = M[2,3,:].reshape((nx,ny)).T
        M30 = M[3,0,:].reshape((nx,ny)).T; M31 = M[3,1,:].reshape((nx,ny)).T; M32 = M[3,2,:].reshape((nx,ny)).T; M33 = M[3,3,:].reshape((nx,ny)).T

    row0 = np.concatenate((M00,M01,M02,M03), axis=1)
    row1 = np.concatenate((M10,M11,M12,M13), axis=1)
    row2 = np.concatenate((M20,M21,M22,M23), axis=1)
    row3 = np.concatenate((M30,M31,M32,M33), axis=1)

    MM = np.concatenate((row0,row1,row2,row3), axis=0)

    plt.rcParams['figure.dpi'] = 300  # Change to get a good quality plot. Otherwise is very blurry
    plt.rcParams['savefig.dpi'] = 300
        
    plt.imshow(MM)
    plt.colorbar()
    plt.title(title)
    
    if save is True:
        plt.savefig(f"{title}.png")  # Save the plot

    plt.show()

    
def plot_IPPtetrahedron(P1, P2, P3): 
    
    """Plots the IPPs distribution in the tetrahedral physically feasible region.
           
    Parameters: 
        P1, P2, P3 (1D array) each: IPPs    

    Returns:
        Plot of the IPPs distribution. and the feasible region
    """ 
    
    ### Customization ###
    points_color = 'winter' # Color of the IPP points
    tetraedron_color = '#800000' 
    line_color = 'orange' # Color of the edges of the tetraedron
    ####################
    
    # Create 3D figure
    plt.figure('SPLTV',figsize=(10,5))
    ax = plt.axes(projection='3d')

    ax.axes.set_xlim3d(left=0, right=1) # Axis limits
    ax.axes.set_ylim3d(bottom=0, top=1) 
    ax.axes.set_zlim3d(bottom=0, top=1) 

    # Vertices of the tetraedron
    p0 = np.array([0,0,0]) # Coordinates
    p1 = np.array([0,0,1])
    p2 = np.array([0,1,1])
    p3 = np.array([1,1,1])

    # Edges of the tetraedron
    x1, y1, z1 = [p0[0],p2[0]], [p0[1],p2[1]], [p0[2],p2[2]] # Line from p0 to p2
    x2, y2, z2 = [p0[0],p3[0]], [p0[1],p3[1]], [p0[2],p3[2]] # p0-p3
    x3, y3, z3 = [p2[0],p3[0]], [p2[1],p3[1]], [p2[2],p3[2]] # p2-p3
    x4, y4, z4 = [p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]] # p0-p1
    x5, y5, z5 = [p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]] # p1-p2
    x6, y6, z6 = [p1[0],p3[0]], [p1[1],p3[1]], [p1[2],p3[2]] # p1-p2

    # Plot lines created above
    ax.plot(x1, y1, z1, color=line_color); ax.plot(x2, y2, z2, color=line_color) 
    ax.plot(x3, y3, z3, color=line_color); ax.plot(x4, y4, z4, color=line_color) 
    ax.plot(x5, y5, z5, color=line_color); ax.plot(x6, y6, z6, color=line_color) 

    # Planes of the tetraedron
    verts1 = [((p0), (p2), (p3))]  # Vertices of the planes
    verts2 = [((p0), (p1), (p2))] 
    verts3 = [((p0), (p1), (p3))]
    verts4 = [((p1), (p2), (p3))]

    srf1 = Poly3DCollection(verts1, alpha=.25, facecolor=tetraedron_color)  # Create the surfaces
    srf2 = Poly3DCollection(verts2, alpha=.25, facecolor=tetraedron_color)
    srf3 = Poly3DCollection(verts3, alpha=.25, facecolor=tetraedron_color)
    srf4 = Poly3DCollection(verts4, alpha=.25, facecolor=tetraedron_color)

    plt.gca().add_collection3d(srf1)  # Add surfaces to the figure
    plt.gca().add_collection3d(srf2)
    plt.gca().add_collection3d(srf3)
    plt.gca().add_collection3d(srf4)

    # Labels
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    # IPPs data
    ax.scatter3D(P1, P2, P3, c=P3, cmap=points_color, s=1, alpha=0.5);
    
    # Show plot
    # ax.view_init(30, 0)  # Change view uncommenting and changing angles
    plt.show()
    
    # For a 3d interactive plot:
    # Before running the function plot_IPPtetrahedron(), run in the CONSOLE the
    # following command: %matplotlib qt

###################
###  P3 FILTER  ###
###################

def P3_filter(M):
    
    """Returns the Mueller matrix with the P3 filter applied.
        
    Parameters: 
        M (4x4xN array): Mueller matrix to filter.
    
    Returns:
        M (4x4xN array): Mueller matrix P3-filtered.
    """ 
    
    N = len(M[0,0])  # Number of pixels
    P3 = IPPs(M)[2]
    one = np.ones_like(P3)  # Array of 1's
    Mdiag = identity(4, N)
    
    one_P3 = one - P3
    Mdiag = np.einsum('i,ijk->ijk', one_P3, Mdiag)  # (1-P3) * Mdiag
    
    M_filt = M - Mdiag
    
    return M_filt

