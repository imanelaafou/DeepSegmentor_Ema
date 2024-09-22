import math

import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
#from scipy.misc import imresize
from PIL import Image
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError
from sklearn.cluster import KMeans
import cv2
import numpy as np
from PIL import Image
from skimage import color


def apply_kmeans_to_thresholded_image(image):
    gray_image = (color.rgb2gray(image))
    image_flat = gray_image.astype('float32').reshape((-1, 1))
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(image_flat)
    segmented_image = kmeans.labels_.reshape(gray_image.shape)

    if segmented_image.dtype != np.uint8:
       segmented_image = segmented_image.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_image, connectivity=8)

    output = np.zeros_like(segmented_image, dtype=np.uint8)
    h,w = output.shape

    for i in range(1, num_labels):
        componentMask = (labels == i).astype(np.uint8) * 255
        if stats[i, cv2.CC_STAT_AREA] > 0.08 * math.sqrt(h*h + w*w):
            output = cv2.bitwise_or(output, componentMask)
    output = (output > 0).astype(np.uint8) * 255
    return output


def find_coordinates(visuals, image_path, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        h, w, _ = im.shape
        im = Image.fromarray(im)
        if aspect_ratio > 1.0:
            #im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            im = im.resize((int(w * aspect_ratio), h), Image.BICUBIC)
        if aspect_ratio < 1.0:
            #im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            im = im.resize((w, int(h / aspect_ratio)), Image.BICUBIC)
        im = np.array(im, dtype='uint8')


# def create_markers_in_agisoft(image_path, image_name):
#     chunk = Metashape.app.document.chunk
#     surfrace_model = chunk.model
#     crs = chunk.crs
#     T = chunk.transform.matrix
#     cameras = [camera for camera in chunk.cameras if camera.transform and camera.type == Metashape.Camera.Type.Regular]  # list of aligned cameras
#     for camera in cameras:
#         if camera.label == image_name:
#             binary_image = cv2.imread(image_path, 0)
#             contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             contour_id = 1
#             for contour in contours:
#                 for point in contour:
#                     x_coord = point[0][0]
#                     y_coord = point[0][1]
#                     marker = chunk.addMarker()
#                     ray_origin = camera.unproject(Metashape.Vector([x_coord, y_coord, 0]))
#                     ray_target = camera.unproject(Metashape.Vector([x_coord, y_coord, 1]))
#                     picked_point = surfrace_model.pickPoint(ray_origin, ray_target)
#                     if picked_point is None:
#                         print(f"Warning: No point found in cloud for camera {image_name} at ({x_coord}, {y_coord})")
#                         picked_point = surfrace_model.pickPoint(ray_origin, ray_target)
#                         if picked_point is None:
#                             print(f"Warning: No point found in model for camera {image_name} at ({x_coord}, {y_coord})")
#                     coord = crs.project(T.mulp(picked_point))
#                     marker.label = image_name
#                     marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([x_coord, y_coord]), True)
#                     marker.reference.location = coord
#                     marker.reference.enabled = True
#                     # output_file.write("{:s},{:.6f},{:.6f},{:.6f}\n".format(marker.label, coord.x, coord.y, coord.z))
#                     break
#                     # file.write(f"{camera_name}, {real_x:.2f}, {real_y:.2f}\n")
#                 contour_id += 1

def save_fused_only_binary(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        im = Image.fromarray(im)
        #im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio >= 1.0:
            # im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            im = im.resize((h, w), Image.BICUBIC)
        if aspect_ratio < 1.0:
            # im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            im = im.resize((int(h / aspect_ratio), int(w / aspect_ratio)), Image.BICUBIC)
        im = np.array(im, dtype='uint8')
        if label == 'fused':
           im = apply_kmeans_to_thresholded_image(im)
           util.save_image(im, save_path)


        ims.append(image_name)
        txts.append(label)
        links.append(image_name)

    webpage.add_images(ims, txts, links, width=width)