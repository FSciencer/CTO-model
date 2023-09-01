
import os
import cv2
import random
import numpy as np
from matplotlib import pylab as plt
from skimage import morphology
from scipy.ndimage import binary_dilation, binary_erosion
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from glob import glob
import skimage
import cc3d
from empatches import EMPatches


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


def filling(mask):
    coronary = morphology.skeletonize(mask.astype(np.uint8))
    structure = skimage.morphology.ball(radius=7)
    dilate = binary_dilation(coronary, structure, iterations=1)
    filling_result = binary_erosion(dilate, structure, iterations=1)
    return filling_result - mask


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
    def __init__(self, im_path, th):
        self.readImage(im_path)
        self.h, self.w, self.d = self.im.shape
        self.passedBy = np.zeros((self.h, self.w, self.d), np.double)
        self.currentRegion = 0
        self.iterations = 0
        self.SEGS = np.zeros((self.h, self.w, self.d, 3), dtype='uint8')
        self.stack = Stack()
        self.thresh = float(th)

    def readImage(self, img_path):
        self.im = cv2.imread(img_path, 1)

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
        cv2.imshow("", self.SEGS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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


def get_patch(img, mask, patchsize=64, overlap=0.2):

    labels_out = cc3d.connected_components(mask)
    labels_out[labels_out > 1] = 1
    stats = cc3d.statistics(labels_out)
    bb = stats['bounding_boxes'][-1]
    zt, zd, yf, yb, xl, xr = bb[0].start, bb[0].stop, bb[1].start, bb[1].stop, bb[2].start, bb[2].stop

    exp_nm = patchsize // 2
    d, h, w = img.shape
    yf = max(int(yf - exp_nm), 0)
    yb = min(int(yb + exp_nm), h)
    xl = max(int(xl - exp_nm), 0)
    xr = min(int(xr + exp_nm), w)
    roi_img = img[zt:zd, yf:yb, xl:xr]
    roi_mask = mask[zt:zd, yf:yb, xl:xr]

    roi_img = np.transpose(roi_img, [1, 2, 0])
    roi_mask = np.transpose(roi_mask, [1, 2, 0])
    emp = EMPatches()
    img_patches, _ = emp.extract_patches(roi_img, patchsize, overlap, vox=True)
    mask_patches, _ = emp.extract_patches(roi_mask, patchsize, overlap, vox=True)

    return img_patches, mask_patches


if __name__ == '__main__':

    global_patchsize = 64

    img_path = 'dataset/cta'
    mask_path = 'dataset/label'
    img_patch_path = 'dataset/cta_patch'
    mask_patch_path = 'dataset/label_patch'
    if not os.path.exists(img_patch_path):
        os.makedirs(img_patch_path)
    if not os.path.exists(mask_patch_path):
        os.makedirs(mask_patch_path)

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
        img_patches, mask_patches = get_patch(img, mask, patchsize=global_patchsize, overlap=0.2)

        for ind in range(len(img_patches)):
            s_idx = str(ind)
            d_idx = '0' * (4 - len(s_idx)) + s_idx  # '0000', '0001', '0002', ...
            filename = cn + '_patch_' + d_idx + '.nii.gz'
            img_ab_path = os.path.join(img_patch_path, filename)
            mask_ab_path = os.path.join(mask_patch_path, filename)
            img_out = np.transpose(img_patches[ind], [2, 0, 1])  # [x,y,z] -> [z,x,y]
            mask_out = np.transpose(mask_patches[ind], [2, 0, 1])
            sitk.WriteImage(sitk.GetImageFromArray(img_out), img_ab_path)
            sitk.WriteImage(sitk.GetImageFromArray(mask_out), mask_ab_path)
            print("Saving patch nii of {}_{} ...".format(cn, d_idx))

