# from PIL import Image
#
# # Load the image
# image = Image.open("input_image.jpg")
#
# # Create a new image with the same dimensions and mode as the input image
# output_image = Image.new(image.mode, image.size)
#
# # Set the depth value for each pixel
# for y in range(image.size[1]):
#     for x in range(image.size[0]):
#         depth = x // 10
#         color = image.getpixel((x, y))
#         output_image.putpixel((x, y), (color[0], color[1], depth))
#
# # Save the output image
# output_image.save("output_image.jpg")




# attempt 2

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imread
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.ndimage as ndimage
#
# imageFile = 'input_image.jpg'
# mat = imread(imageFile)
# mat = mat[:, :, 0]  # get the first channel
# rows, cols = mat.shape
# xv, yv = np.meshgrid(range(cols), range(rows)[::-1])
#
# blurred = mat
# fig = plt.figure(figsize=(6, 6))
#
# ax = fig.add_subplot(221)
#
# ax.imshow(mat, cmap='gray')
#
# ax = fig.add_subplot(222, projection='3d')
# plt.title("3d")
# ax.elev = 75
# ax.plot_surface(xv, yv, mat)
#
# ax = fig.add_subplot(223)
# ax.imshow(blurred, cmap='gray')
#
# ax = fig.add_subplot(224, projection='3d')
# plt.title("3d")
# ax.elev = 75
# ax.plot_surface(xv, yv, blurred)
# plt.show()




#attempt 3


import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

import numpy as np
import open3d as o3d


feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# load and resize the input image
image = Image.open("brain_mri.jpeg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# get the prediction from the model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# remove borders
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# visualize the prediction
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
# plt.pause(5)






width, height = image.size

depth_image = (output * 255 / np.max(output)).astype('uint8')
image = np.array(image)

# create rgbd image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

# camera settings
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

# create point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# o3d.visualization.draw_geometries([pcd])


cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
pcd = pcd.select_by_index(ind)

# estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

# surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# save the mesh
o3d.io.write_triangle_mesh(f'./mesh.obj', mesh)

# visualize the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)