import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def seg(img_path):

    image = cv2.imread(img_path)
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #finding edges
    edged = cv2.Canny(gray, 30, 200)
    cv2.waitKey(0)

    #finding contour
    contours, hierarchy = cv2.findContours(edged,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.imshow('edges', edged)
    cv2.waitKey(0)

    #print("contours" + str(len(contours)))

    # doing contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edge(path):
    #read img
    img = cv2.imread(path)
    # Apply Canny
    edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)
    plt.figure()
    plt.title(f'{path}')
    #uncomment to save image
    #plt.imsave('imgname.png', edges, cmap='gray', format='png')
    plt.imshow(edges, cmap='gray')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default='example_data/default-trivialseg.jpg', help=f'image path')
    parser.add_argument('--func', default='seg', help='contour or edge')
    args = parser.parse_args()

    img_path = args.i
    do_func = args.func

    if do_func == 'seg':
        seg(img_path)
    elif do_func == 'edge':
        edge(img_path)
    else:
        print('error with function, none chosen')
main()