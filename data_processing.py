
import os
import numpy as np
from matplotlib import pylab as plt
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from glob import glob
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
    label_path = 'dataset/label'
    img_patch_path = 'dataset/cta_patch'
    label_patch_path = 'dataset/label_patch'
    if not os.path.exists(img_patch_path):
        os.makedirs(img_patch_path)
    if not os.path.exists(label_patch_path):
        os.makedirs(label_patch_path)

    img_list = [d for d in glob(img_path + '/*')]
    label_list = [d for d in glob(label_path + '/*')]
    img_list.sort()
    label_list.sort()

    img_name = os.listdir(img_path)
    img_name.sort()
    case_name = ['_'.join(img_name[i].split('.')[0].split('_')[:-1]) for i in range(len(img_name))]

    for k in range(len(img_list)):
        cn = case_name[k]
        print("Processing case {} ...".format(cn))
        itk_img = sitk.ReadImage(img_list[k])
        itk_label = sitk.ReadImage(label_list[k])
        img = sitk.GetArrayFromImage(itk_img)
        label = sitk.GetArrayFromImage(itk_label)
        img_patches, label_patches = get_patch(img, label, patchsize=global_patchsize, overlap=0.2)

        for ind in range(len(img_patches)):
            s_idx = str(ind)
            d_idx = '0' * (4 - len(s_idx)) + s_idx  # '0000', '0001', '0002', ...
            filename = cn + '_patch_' + d_idx + '.nii.gz'
            img_ab_path = os.path.join(img_patch_path, filename)
            mask_ab_path = os.path.join(label_patch_path, filename)
            img_out = np.transpose(img_patches[ind], [2, 0, 1])  # [x,y,z] -> [z,x,y]
            mask_out = np.transpose(label_patches[ind], [2, 0, 1])
            sitk.WriteImage(sitk.GetImageFromArray(img_out), img_ab_path)
            sitk.WriteImage(sitk.GetImageFromArray(mask_out), mask_ab_path)
            print("Saving patch nii of {}_{} ...".format(cn, d_idx))
