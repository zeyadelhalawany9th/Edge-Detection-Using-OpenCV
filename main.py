import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

weak = 100
strong = 255
lowThresholdRatio = 0.2
highThresholdRatio = 0.2

class functions():


    # Function that creates gaussian kernel with size length (size) and a sigma of (sigma)
    def gaussianMask(self, size, sigma = 1):

        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

        return kernel / np.sum(kernel)


    def firstDerivativeEdgeDetector(self, image):

        img = cv2.imread(image, 0)
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)

        # Filter for x derivative image
        kernelx = np.zeros((3, 3), np.float32)
        kernelx[0][0] = -1
        kernelx[0][1] = 0
        kernelx[0][2] = 1
        kernelx[1][0] = -1
        kernelx[1][1] = 0
        kernelx[1][2] = 1
        kernelx[2][0] = -1
        kernelx[2][1] = 0
        kernelx[2][2] = 1

        # Filter for y derivative image
        kernely = np.zeros((3, 3), np.float32)
        kernely[0][0] = -1
        kernely[0][1] = -1
        kernely[0][2] = -1
        kernely[1][0] = 0
        kernely[1][1] = 0
        kernely[1][2] = 0
        kernely[2][0] = 1
        kernely[2][1] = 1
        kernely[2][2] = 1

        Gx = cv2.filter2D(img, -1, kernelx)
        Gy = cv2.filter2D(img, -1, kernely)

        edgeDetectedImage = Gx + Gy

        return edgeDetectedImage

    def secondDerivativeEdgeDetector(self, image):


        img = cv2.imread(image, 0)
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)

        # Filter for x derivative image
        kernelx = np.zeros((3, 3), np.float32)
        kernelx[0][0] = 1
        kernelx[0][1] = -2
        kernelx[0][2] = 1
        kernelx[1][0] = 1
        kernelx[1][1] = -2
        kernelx[1][2] = 1
        kernelx[2][0] = 1
        kernelx[2][1] = -2
        kernelx[2][2] = 1

        vertical_edge = cv2.filter2D(img, cv2.CV_32F, kernelx)
        vertical_edge = cv2.convertScaleAbs(vertical_edge)

        # Filter for y derivative
        kernely = np.zeros((3, 3), np.float32)
        kernely[0][0] = 1
        kernely[0][1] = 1
        kernely[0][2] = 1
        kernely[1][0] = -2
        kernely[1][1] = -2
        kernely[1][2] = -2
        kernely[2][0] = 1
        kernely[2][1] = 1
        kernely[2][2] = 1

        horizontal_edge = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernely)
        horizontal_edge = cv2.convertScaleAbs(horizontal_edge)

        cv2.imshow('Vertical Edge', vertical_edge)
        cv2.waitKey(0)
        cv2.imshow('Horizontal Edge', horizontal_edge)
        cv2.waitKey(0)

        result_edge = horizontal_edge + vertical_edge

        return result_edge


    def Prewitt(self, image):

        img = cv2.imread(image)
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)

        height, width, depth = np.shape(img)

        # horizontal edge detector
        hor = [[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]]

        # vertical edge detector
        ver = [[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]]

        H = int(len(ver))

        for i in range(0, height - H):
            for j in range(0, width - H):
                summ = 0
                for k in range(0, H):
                    for l in range(0, H):
                        summ = summ + ver[k][l] * img[i + k][j + l]
                if (summ[0] > 255):
                    summ = [255, 255, 255]
                elif (summ[0] < 0):
                    summ = [0, 0, 0]
                img[i][j] = summ

        return img

    def helperFunction(self, size, sigma=1):

        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def Sobil(self, image):

        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(image, Kx)
        Iy = ndimage.filters.convolve(image, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)

        return (G, theta)

    def non_maximum_suppression(self, image, D):
        M, N = image.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # 0
                    if (0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180):
                        q = image[i, j + 1]
                        r = image[i, j - 1]
                    # 45
                    elif (22.5 <= angle[i, j] <= 67.5):
                        q = image[i + 1, j - 1]
                        r = image[i - 1, j + 1]
                    # 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = image[i + 1, j]
                        r = image[i - 1, j]
                    # 135
                    elif (112.5 <= angle[i, j] <= 157.5):
                        q = image[i - 1, j - 1]
                        r = image[i + 1, j + 1]

                    if (image[i, j] >= q and image[i, j] >= r):
                        Z[i, j] = image[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def double_threshold(self, image):

        highThreshold = image.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        M, N = image.shape
        res = np.zeros((M, N), dtype=np.int32)

        strong_i, strong_j = np.where(image >= highThreshold)
        zeros_i, zeros_j = np.where(image < lowThreshold)

        weak_i, weak_j = np.where((image <= highThreshold) & (image > lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def Edge_Linking(self, image):

        M, N = image.shape

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (image[i, j] == weak):
                    try:
                        if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (
                                image[i + 1, j + 1] == strong) or (
                                image[i, j - 1] == strong) or (image[i, j + 1] == strong) or (
                                image[i - 1, j - 1] == strong) or (
                                image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass

        return image


    def Canny(self, image):

        img_original = cv2.imread(image, 1)
        img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)

        img_smoothed = ndimage.filters.convolve(img, self.helperFunction(3, 1))
        gradients, thetas = self.Sobil(img_smoothed)
        nonMaxImg = self.non_maximum_suppression(gradients, thetas)
        thresholdImg = self.double_threshold(nonMaxImg)
        img_final = self.Edge_Linking(thresholdImg)

        return img_final


def main():

    # First we create an object of the class functions in order to be able to compute the functions of the methods

    object = functions()


    # Then we test each method separately


    # First Derivative Edge Detection
    # After running, one window with the original image will appear then after closing it
    # second window will appear with the image after the first derivative edge detection
    a = object.firstDerivativeEdgeDetector('example.jpg')
    cv2.imshow('Image After Applying First Derivative Edge Detection', a)
    cv2.waitKey(0)

    # Second Derivative Edge Detection
    # After running, one window with the original image will appear then after closing it
    # second window will appear with the vertical edge then
    # a third window will appear with the horizontal edge then a fourth and final
    # window will appear with the image after the second derivative edge detection
    b = object.secondDerivativeEdgeDetector('example.jpg')
    cv2.imshow('Image After Applying Second Derivative Edge Detection', b)
    cv2.waitKey(0)

    # Prewitt Edge Detection
    # After running, one window with the original image will appear then after closing it
    # second window will appear with the image after the Prewitt edge detection
    # This takes a longer period of time to complete than the other methods
    c = object.Prewitt('example.jpg')
    cv2.imshow('Image After Applying Prewitt Edge Detection', c)
    cv2.waitKey(0)

    # Canny Edge Detection
    # After running, one window with the original image will appear then after closing it
    # second window will appear with the image after Canny edge detection
    d = object.Canny('example.jpg')
    plt.imshow(cv2.imread('example.jpg', 1), cmap='gray')
    plt.title('Original Image')
    plt.show()
    plt.imshow(d, cmap='gray')
    plt.title('Image After Canny Edge Detection')
    plt.show()



if __name__ == '__main__':
    main()
