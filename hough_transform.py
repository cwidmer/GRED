

import cv2
import Image
import numpy as np
import data_processing as dp


def fit_circle_houghtransform(vec_x, vec_y):
    """
    use hough transform to fit circle:
    http://www.janeriksolem.net/2012/08/reading-gauges-detecting-lines-and.html
    """

    #TODO convert vec_x, vec_y to matrix
    # plug in matrix, extract circles

    import ipdb
    ipdb.set_trace()

    prefix="/home/cwidmer/Documents/phd/projects/cell_fitting/data/data/20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9"
    fn = prefix + "/" + "20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9_w617_z08.tif"
    im = np.array(dp.image2array(Image.open(fn)), dtype=np.uint8)


    im = np.array(dp.image2array(Image.open(fn)), dtype=np.uint8)
    m,n = im.shape
    circles = cv2.HoughCircles(im, cv2.cv.CV_HOUGH_GRADIENT, 2, 10, np.array([]), 20, 60, m/10)[0]
    c = circles[0]
    draw_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.circle(draw_im, (c[0],c[1]), c[2], (0,255,0), 2)
    cv2.imshow("circles",im)
    cv2.waitKey()
    cv2.imwrite("res.jpg",draw_im)


def main():
    """
    main
    """

    fit_circle_houghtransform()


if __name__ == "__main__":
    main()
