#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree
# Import functions to read and write ply files
from ply import write_ply, read_ply
from plyfile import PlyData, PlyElement

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):
    
    points_mean = np.mean(points , axis =0)
    centre = points - points_mean
    
    cov = np.dot(centre.T, centre)/len(points)
    
   
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
 

    return  eigenvalues, eigenvectors



#def compute_local_PCA(query_points, cloud_points, radius):
def compute_local_PCA(query_points, cloud_points):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    tree = KDTree(cloud_points)

    #for ind, query_point in enumerate(query_points):
        #indices = tree.query_radius([query_point], r=radius, return_distance=False)[0]

        # Neighborhood points
        #if len(indices) > 0:  # Check if there are any points in the neighborhood
            #neighborhood_points = cloud_points[indices]

        
            #eigenvalues, eigenvectors = PCA(neighborhood_points)

            # Store the results in descending order of eigenvalues
            #sorted_indices = np.argsort(eigenvalues)[::-1]
            #all_eigenvalues[ind] = eigenvalues[sorted_indices]
            #all_eigenvectors[ind] = eigenvectors[:, sorted_indices]

    for ind, query_point in enumerate(query_points):
        # Use KNN to find the indices of the k nearest neighbors
        distances, indices = tree.query([query_point], k=30)
        
        neighborhood_points = cloud_points[indices[0]] # Indices[0] to get the first query result

       
        eigenvalues, eigenvectors = PCA(neighborhood_points)

        # Store the results
        sorted_indices = np.argsort(eigenvalues)[::-1]
        all_eigenvalues[ind] = eigenvalues[sorted_indices]
        all_eigenvectors[ind] = eigenvectors[:, sorted_indices]

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    
    # Initialize features
    verticality = np.zeros(query_points.shape[0])
    linearity = np.zeros(query_points.shape[0])
    planarity = np.zeros(query_points.shape[0])
    sphericity = np.zeros(query_points.shape[0])
    
    # Compute features
    for i in range(query_points.shape[0]):
        eigenvalues = all_eigenvalues[i]
        eigenvectors = all_eigenvectors[i]
        
        # Ensure eigenvalues are sorted
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        
        # Avoid division by zero
        epsilon = 1e-6
        lambda_1, lambda_2, lambda_3 = eigenvalues_sorted + epsilon
        
        # Compute descriptors
        linearity[i] = (lambda_1 - lambda_2) / lambda_1
        planarity[i] = (lambda_2 - lambda_3) / lambda_1
        sphericity[i] = lambda_3 / lambda_1
        
        # Compute verticality, assuming the last eigenvector is the normal
        normal = eigenvectors[:, -1]
        verticality_angle = np.arccos(np.abs(normal[2]))  # Assuming Z is up
        verticality[i] = 1 - 2 * verticality_angle / np.pi
    
    return verticality, linearity, planarity, sphericity

 
# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud) #compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('Lille_street_small_normals.ply_3', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])


      # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud , 0.5) 
        print(verticality)
        print(linearity)
        print(planarity)
        print(sphericity)

        sphericity_normalized = (sphericity - np.min(sphericity)) / (np.max(sphericity) - np.min(sphericity))

        # Using a colormap (e.g., 'jet') to map normalized 'verticality' to RGB colors
        cmap = plt.get_cmap('jet')
        colors = cmap(sphericity_normalized)

        # Convert colors to 8-bit RGB values    
        colors_rgb = (colors[:, :3] * 255).astype(np.uint8)

        # Prepare the data for saving, including RGB color based on 'verticality'
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        enhanced_cloud_with_sphericity_color = np.empty(cloud.shape[0], dtype=dtype)
        enhanced_cloud_with_sphericity_color['x'], enhanced_cloud_with_sphericity_color['y'], enhanced_cloud_with_sphericity_color['z'] = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        enhanced_cloud_with_sphericity_color['red'], enhanced_cloud_with_sphericity_color['green'], enhanced_cloud_with_sphericity_color['blue'] = colors_rgb[:, 0], colors_rgb[:, 1], colors_rgb[:, 2]

        # Save the enhanced cloud as a new .ply file with verticality color
        el = PlyElement.describe(enhanced_cloud_with_sphericity_color, 'vertex')
        PlyData([el], text=True).write('Lille_street_small_with_sphericity_color.ply')

