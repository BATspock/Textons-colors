'''
Create a 3D voxel data from the image
The top-most (#1) and bottom-most (#13) layer will contain all zeros
The middle 10 layers (#3 to #12) contain the same pixel values as the gray scale image
There is an additional layer(#2) for the base of the model 
'''
import cv2
import numpy as np 
from stl import mesh
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt 
from skimage import measure

image1 = cv2.imread("center: 2.png",0)
image2 = cv2.imread("center processing: 0.png",0)
show = np.zeros_like(image1)
show[image1 == 0] = 255
show[image2 == 0] = 127
show = cv2.medianBlur(show,5)

cv2.imshow("check", show)
cv2.waitKey(0)
cv2.destroyAllWindows()

rows, cols = show.shape
threshold = threshold_otsu(show)

layers = 13
rows += 2
cols += 2
voxel = np.zeros((rows, cols, layers))
voxel[:, :, 1] = np.ones((rows, cols)).astype('float32')

# making the boundary voxel values to be zero, for the marching cubes algorithm to work correctly
voxel[0, :, :] = np.zeros((cols, layers)).astype('float32')
voxel[(rows - 1), :, :] = np.zeros((cols, layers)).astype('float32')

voxel[:, 0, :] = np.zeros((rows, layers)).astype('float32')
voxel[:, (cols - 1), :] = np.zeros((rows, layers)).astype('float32')


'''
Create the middle 10 layers from the image
Based on the pixel values the layers are created to assign different heights to different regions in the image
'''

for level in range(1, 10):
    level_threshold = level * 0.1
    for j in range(0, rows - 2):
        for k in range(0, cols - 2):
            pixel_value = show[j][k]
            if pixel_value > level_threshold:
                voxel[j + 1][k + 1][level + 1] = pixel_value


print(voxel.shape)
#plt.plot(voxel[:,:,:1])
#plt.show()

verts, faces, normals, values = measure.marching_cubes_lewiner(volume=voxel, level=threshold,spacing=(1., 1., 1.), gradient_direction='descent')

# Export the mesh as stl
mymesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

for i, f in enumerate(faces):
    for j in range(3):
        mymesh.vectors[i][j] = verts[f[j], :]


mymesh.save('test.stl')
