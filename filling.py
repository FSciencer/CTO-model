
import os
import random
import numpy as np
from numpy import linalg as lg
from matplotlib import pylab as plt
from skimage import morphology
from scipy.ndimage import binary_dilation
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from glob import glob
import skimage
import cc3d


def plot_3d(image, threshold=0):
    p = image.transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=1.0)
    face_color = [1, 0., 0.]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.axis('off')
    plt.show()


def skeletonize_filling(img, mask):
    centerline = morphology.skeletonize(mask.astype(np.uint8))

    depth, height, width = centerline.shape[0], centerline.shape[1], centerline.shape[2]
    tmp = centerline.copy()
    point_list = []
    raudis = 4
    threshold = 4
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                if (tmp[d][h][w]) == 0:
                    continue
                count = 0
                for k in range(d - raudis, d + raudis + 1):
                    for i in range(h - raudis, h + raudis + 1):
                        for j in range(w - raudis, w + raudis + 1):
                            if k < 0 or i < 0 or j < 0 or k > depth - 1 or i > height - 1 or j > width - 1:
                                continue
                            elif tmp[k][i][j] == 1:
                                count += 1
                if count < threshold:
                    point = (d, h, w)
                    point_list.append(point)

    patch_size = 64
    candidate_point = list()
    for ind in range(len(point_list) - 1):
        distance = lg.norm(np.array([point_list[ind][0] - point_list[ind + 1][0], point_list[ind][1] - point_list[ind + 1][1], point_list[ind][2] - point_list[ind + 1][2]]), axis=0)
        if distance <= patch_size:
            candidate_point.append([point_list[ind], point_list[ind + 1]])

    patches = list()
    index = list()
    for k in range(len(candidate_point)):
        point1, point2 = candidate_point[k]
        z = (point1[0] + point2[0]) // 2
        x = (point1[1] + point2[1]) // 2
        y = (point1[2] + point2[2]) // 2
        scale = patch_size // 2
        lesion = img[int(z - scale):int(z + scale), int(x - scale):int(x + scale), int(y - scale):int(y + scale)]
        patches.append(lesion)
        index.append([int(z - scale), int(z + scale), int(x - scale), int(x + scale), int(y - scale), int(y + scale)])

    global_mask = np.zeros(mask.shape)
    for k in range(len(patches)):
        local_mask = regionGrow(patches[k])
        global_mask[index[k]] = local_mask
    combined_mask = global_mask + mask
    combined_mask = cc3d.connected_components(combined_mask)
    filling_mask = cc3d.largest_k(combined_mask, k=2)
    filling_mask[filling_mask > 1] = 1

    lesions = filling_mask - mask
    structure = skimage.morphology.ball(radius=7)
    lesion_dilate = binary_dilation(lesions, structure, iterations=1)

    return lesion_dilate


class Stack:
    def __init__(self):
        self.item = []
        self.obj = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []


class regionGrow:
    def __init__(self, img, th=100):
        self.im = img
        self.h, self.w, self.d = self.im.shape
        self.passedBy = np.zeros((self.h, self.w, self.d), np.double)
        self.currentRegion = 0
        self.iterations = 0
        self.SEGS = np.zeros((self.h, self.w, self.d, 3), dtype='uint8')
        self.stack = Stack()
        self.thresh = float(th)

    def getNeighbour(self, x0, y0, z0):
        neighbour = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if (i, j, k) == (0, 0, 0):
                        continue
                    x = x0 + i
                    y = y0 + j
                    z = z0 + k
                    if self.limit(x, y, z):
                        neighbour.append((x, y, z))
        return neighbour

    def ApplyRegionGrow(self):
        randomseeds = [[self.h / 2, self.w / 2, self.d / 2], [self.h / 3, self.w / 3, self.d / 3],
                       [2 * self.h / 3, self.w / 3, self.d / 3], [self.h / 3, 2 * self.w / 3, self.d / 3], [self.h / 3, self.w / 3, 2 * self.d / 3],
                       [self.h / 3 - 10, self.w / 3, self.d / 3], [self.h / 3, self.w / 3 - 10, self.d / 3], [self.h / 3, self.w / 3, self.d / 3 - 10],
                       [self.h / 3 - 10, self.w / 3 - 10, self.d / 3], [self.h / 3, self.w / 3 - 10, self.d / 3 - 10], [self.h / 3 - 10, self.w / 3, self.d / 3 - 10],
                       [self.h / 3 - 10, self.w / 3 - 10, self.d / 3 - 10],
                       [2 * self.h / 3, 2 * self.w / 3, self.d / 3], [self.h / 3, 2 * self.w / 3, 2 * self.d / 3], [2 * self.h / 3, self.w / 3, 2 * self.d / 3],
                       [2 * self.h / 3, 2 * self.w / 3, 2 * self.d / 3],
                       [2 * self.h / 3, self.w / 3 - 10, self.d / 3 - 10], [self.h / 3 - 10, 2 * self.w / 3, self.d / 3 - 10], [self.h / 3 - 10, self.w / 3 - 10, 2 * self.d / 3],
                       [2 * self.h / 3, 2 * self.w / 3, self.d / 3 - 10], [self.h / 3 - 10, 2 * self.w / 3, 2 * self.d / 3], [2 * self.h / 3, self.w / 3 - 10, 2 * self.d / 3],
                       ]
        np.random.shuffle(randomseeds)
        for x0 in range(self.h):
            for y0 in range(self.w):
                for z0 in range(self.d):
                    if self.passedBy[x0, y0, z0] == 0 and (int(self.im[x0, y0, z0, 0]) * int(self.im[x0, y0, z0, 1]) * int(self.im[x0, y0, z0, 2]) > 0):
                        self.currentRegion += 1
                        self.passedBy[x0, y0] = self.currentRegion
                        self.stack.push((x0, y0))
                        self.prev_region_count = 0
                        while not self.stack.isEmpty():
                            x, y, z = self.stack.pop()
                            self.BFS(x, y, z)
                            self.iterations += 1
                        if self.PassedAll():
                            break
                        if self.prev_region_count < 8 * 8:
                            self.passedBy[self.passedBy == self.currentRegion] = 0
                            x0 = random.randint(x0 - 4, x0 + 4)
                            y0 = random.randint(y0 - 4, y0 + 4)
                            z0 = random.randint(z0 - 4, z0 + 4)
                            x0 = max(0, x0)
                            y0 = max(0, y0)
                            z0 = max(0, z0)
                            x0 = min(x0, self.h - 1)
                            y0 = min(y0, self.w - 1)
                            z0 = min(z0, self.d - 1)
                            self.currentRegion -= 1

        for i in range(0, self.h):
            for j in range(0, self.w):
                for k in range(0, self.d):
                    val = self.passedBy[i][j][k]
                    if val == 0:
                        self.SEGS[i][j][k] = 255, 255, 255
                    else:
                        self.SEGS[i][j][k] = val * 35, val * 90, val * 30
        if self.iterations > 1000:
            print("Max Iterations")
        print("Iterations : " + str(self.iterations))
        return self.SEGS

    def BFS(self, x0, y0, z0):
        regionNum = self.passedBy[x0, y0, z0]
        elems = []
        elems.append((int(self.im[x0, y0, z0, 0]) + int(self.im[x0, y0, z0, 1]) + int(self.im[x0, y0, z0, 2])) / 3)
        var = self.thresh
        neighbours = self.getNeighbour(x0, y0, z0)
        for x, y, z in neighbours:
            if self.passedBy[x, y, z] == 0 and self.distance(x, y, z, x0, y0, z0) < var:
                if self.PassedAll():
                    break
                self.passedBy[x, y, z] = regionNum
                self.stack.push((x, y, z))
                elems.append((int(self.im[x, y, z, 0]) + int(self.im[x, y, z, 1]) + int(self.im[x, y, z, 2])) / 3)
                var = np.var(elems)
                self.prev_region_count += 1
            var = max(var, self.thresh)

    def PassedAll(self):
        return self.iterations > 1000 or np.count_nonzero(self.passedBy > 0) == self.d * self.w * self.h

    def limit(self, x, y, z):
        return 0 <= x < self.h and 0 <= y < self.w and 0 <= z < self.d

    def distance(self, x, y, z, x0, y0, z0):
        return ((int(self.im[x, y, z, 0]) - int(self.im[x0, y0, z0, 0])) ** 2 +
                (int(self.im[x, y, z, 1]) - int(self.im[x0, y0, z0, 1])) ** 2 +
                (int(self.im[x, y, z, 2]) - int(self.im[x0, y0, z0, 2])) ** 2) ** 0.5


if __name__ == '__main__':

    img_path = 'dataset/cta'
    mask_path = 'dataset/predicted_mask'
    lesion_mask_path = 'dataset/lesion_mask'
    if not os.path.exists(lesion_mask_path):
        os.makedirs(lesion_mask_path)

    img_list = [d for d in glob(img_path + '/*')]
    mask_list = [d for d in glob(mask_path + '/*')]
    img_list.sort()
    mask_list.sort()

    img_name = os.listdir(img_path)
    img_name.sort()
    case_name = ['_'.join(img_name[i].split('.')[0].split('_')[:-1]) for i in range(len(img_name))]

    for k in range(len(img_list)):
        cn = case_name[k]
        print("Processing case {} ...".format(cn))
        itk_img = sitk.ReadImage(img_list[k])
        itk_mask = sitk.ReadImage(mask_list[k])
        img = sitk.GetArrayFromImage(itk_img)
        mask = sitk.GetArrayFromImage(itk_mask)

        candidate_CTO_lesions = skeletonize_filling(img, mask)

        filename = cn + '.nii.gz'
        mask_ab_path = os.path.join(lesion_mask_path, filename)
        sitk.WriteImage(sitk.GetImageFromArray(candidate_CTO_lesions), mask_ab_path)
        print("Saving CTO lesion mask of {} ...".format(cn))

