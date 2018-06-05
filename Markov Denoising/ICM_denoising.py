import cv2
import numpy as np
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type = str, default = "lena.jpg", help = "What is the name of the noisy image?")
    parser.add_argument("--iter", type = int, default = 10, help = "Number of iterations you want for ICM")
    parser.add_argument("--beta", type = float, default = 1, help = "Value of regularisation")
    args = parser.parse_args()
    sys.stdout.write(str(ICM(args)))

# potential fonction corresponding to a gaussian markovian model (quadratic function)
def pot(fi, fj):
    return float((fi-fj))**2
	
#ICM : Iterated conditional mode algorithme
def ICM(args):
    NoisyIm = cv2.imread(args.image, 0)
    height, width = NoisyIm.shape

    sigma2 = 5
    beta = args.beta # regularization parameter 

	# Number of iterations : each new image is used as the new restored image
    for iter in range(args.iter):
        print("iteration {}\n".format(iter+1))
        for i in range(height-1):
            print("line {}/{} ok\n".format(i+1, height))
            for j in range(width):
				# We work in 4-connexity here
                xmin = 0
                min = float((NoisyIm[i][j]*NoisyIm[i][j]))/(2.0*sigma2) + beta*(pot(NoisyIm[i][j-1],0)+pot(NoisyIm[i][j+1],0)+pot(NoisyIm[i-1][j], 0)+pot(NoisyIm[i+1][j], 0))

				#Every shade of gray is tested to find the a local minimum of the energie corresponding to a Gibbs distribution
                for x in range(256):
                    proba = float(((NoisyIm[i][j]-x)*(NoisyIm[i][j]-x)))/(2.0*sigma2) + beta*(pot(NoisyIm[i][j-1],x) + pot(NoisyIm[i][j+1],x) + pot(NoisyIm[i-1][j], x) + pot(NoisyIm[i+1][j], x))

                    if(min>proba):
                        min = proba
                        xmin = x
                NoisyIm [i][j] = xmin

        cv2.imwrite("iter_" + str(iter+1) + "_denoised_" + args.image, NoisyIm)


if __name__ == '__main__':
    main()
