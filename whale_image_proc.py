import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import sklearn


def gauss_sum(xs, mu, sig, om):
    """Sums Gaussians. Here mu is the vector of averages, sig the variances, and om the relative amplitudes."""
    assert xs.ndim == 1, "X must be a one dimensional array of inputs"
    ys = np.zeros(len(xs))
    for j, x in enumerate(xs):
        ys[j] = np.sum(om/(np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2)))
    return ys


def grad_loss(xs, ys, ts, mu, sig, om):
    """Calculates the gradient of the loss function for the gaussian fitting.
    Loss: L = (f(xs[1],mu,sig,om) - ts[1])^2 + ... + (f(xs[m],mu,sig,om) - ts[m])^2
    where xs are the grid points, ts are the data values at each grid point, f(xs[i],...) = ys[i] is the sum of
    Gaussians, mu is the vector of averages, sig the variances, and om the relative amplitudes."""
    d_mu = np.zeros(len(mu))
    d_sig = np.zeros(len(mu))
    d_om = np.zeros(len(mu))
    norm = 256**2
    for j in range(len(mu)):
        d_mu[j] = 1/norm*np.sum(om[j]*(xs-mu[j])/(np.sqrt(2*np.pi*sig[j]**2)*sig[j]**2)
                                * np.exp(-(xs-mu[j])**2/(2*sig[j]**2)) * (ys-ts))
        d_sig[j] = 1/norm*np.sum((-om[j]/(np.sqrt(2*np.pi*sig[j]**2))/sig[j]
                                  + om[j]*(xs-mu[j])**2/(np.sqrt(2*np.pi*sig[j]**2))/sig[j]**3)
                                 * np.exp(-(xs-mu[j])**2/(2*sig[j]**2)) * (ys-ts))
        d_om[j] = np.sum(1/(np.sqrt(2*np.pi*sig[j]**2))
                         * np.exp(-(xs-mu[j])**2/(2*sig[j]**2)) * (ys-ts)) \
                  + (np.sum(om)-256**2)

        d_sig[j] = d_sig[j]*(sig[j]**2/(1+sig[j]**3))  # Post derivative factor. Helps with the convergence rate, as the
        #                                                derivative is very large for small or large sig values.
    return d_mu, d_sig, d_om


def gauss_fit(xs, ts, mu, sig, om, prec_thresh, iters_outer, iters_inner):
    """Fits a sum of Gaussians to the data. Seeds with provided parameters."""
    return

def pixel_cluster_gauss(img, n=4, iters=100, prec_thresh=1e-7, iters_inner=25, smooth=3):
    """Clusters pixel values by fitting a sum of n or less Gaussians to the distribution of pixel intensities.

    img: array of pixel values. must be black and white (i.e. dimension n x m x 1) with integer pixel values between
        0 to 255.
    n: Positive integer. Number of Gaussians to start fitting process. After initial convergence, Gaussians that are too
        wide and short will be discarded, and the fitting will be run again on the remaining Gaussians.
    iters: Positive integer. Total permitted number of outer iterations (iterations of the outer for-loop)
    prec_thresh: Positive float. Precision threshold for breaking the iterations. Used for both inner and outer loops.
    iters_inner: Positive integer. Total number of iterations for each inner for-loop.
    smooth: Odd number. Indicates size of moving average for smoothing of pixel distribution. A larger smooth factor
        improves convergence rate of Gaussian fitting, but can damage quality of clustering if too high.
    """
    pix_count = np.zeros(256)
    for j in range(256):
        pix_count[j] = np.count_nonzero(img == j)
    assert (smooth % 2) == 1, "smooth must be an odd number."
    if smooth != 1:
        ext_pix_count = np.append(np.zeros((smooth-1)//2), np.append(pix_count, np.zeros((smooth-1)//2)))
        pix_count = np.zeros(256)
        for j in range(smooth):
            pix_count += ext_pix_count[j:256+j]/smooth
    mu = np.linspace(0, 256, n, endpoint=False) + 128/n
    sig = 64/n*np.ones(n)
    om = 256**2/n*np.ones(n)
    xs = np.arange(0, 256, 1)

    fig = plt.figure(-1)
    fig.clf()
    ax1 = plt.subplot(211)
    ax1.bar(xs, pix_count)

    ys = gauss_sum(xs, mu, sig, om)
    ax1.plot(xs, ys, 'r')

    min_eta_mu = 1e-16/n
    min_eta_sig = 1e-16/n
    min_eta_om = 1e-16/n

    eta_mu = 1e-8
    eta_sig = 1e-8
    eta_om = 1e-8
    d_mu, d_sig, d_om = grad_loss(xs, ys, pix_count, mu, sig, om)
    # ax2 = plt.subplot(212)
    t = time.time()
    for j in range(iters):
        ys_oldest = ys
        mu_oldest, sig_oldest, om_oldest = (mu, sig, om)
        for i in range(iters_inner):  # mu
            d_mu_old = d_mu
            mu_old = mu

            mu = mu_old-eta_mu*d_mu
            mu = np.maximum(mu,0)
            mu = np.minimum(mu,255)
            ys = gauss_sum(xs, mu, sig, om)
            d_mu, _, _ = grad_loss(xs, ys, pix_count, mu, sig, om)

            eta_mu = np.fmax(np.abs(np.dot(mu-mu_old, d_mu-d_mu_old)/np.linalg.norm(d_mu-d_mu_old)**2), min_eta_mu)
            prec_mu = np.linalg.norm((mu-mu_old) / (mu+1e-16))

            # ax2.cla()
            # ax2.bar(xs, pix_count)
            # ax2.plot(xs, ys, 'r')
            # plt.draw()
            # plt.pause(0.01)

            # print(mu, d_mu, eta_mu)
            if prec_mu < n*prec_thresh:
                # print("break")
                break

        for i in range(iters_inner):  # om
            d_om_old = d_om
            om_old = om
            om = np.maximum(om_old - eta_om * d_om, 0)
            ys = gauss_sum(xs, mu, sig, om)
            _, _, d_om = grad_loss(xs, ys, pix_count, mu, sig, om)
            eta_om = np.fmax(np.abs(np.dot(om - om_old, d_om - d_om_old) / np.dot(d_om - d_om_old, d_om - d_om_old)),
                         min_eta_om)
            prec_om = np.linalg.norm((om-om_old) / (om+1e-16))

            # ax2.cla()
            # ax2.bar(xs, pix_count)
            # ax2.plot(xs, ys, 'r')
            # plt.draw()
            # plt.pause(0.01)

            # print(om,d_om,eta_om)
            if prec_om < n*prec_thresh:
                # print("break")
                break

        for i in range(iters_inner):  # sig
            d_sig_old = d_sig
            sig_old = sig
            sig = np.maximum(np.abs(sig_old - eta_sig * d_sig), 0.25)
            ys = gauss_sum(xs, mu, sig, om)
            _, d_sig, _ = grad_loss(xs, ys, pix_count, mu, sig, om)
            eta_sig = np.fmax(
                np.abs(np.dot(sig - sig_old, d_sig - d_sig_old) / np.dot(d_sig - d_sig_old, d_sig - d_sig_old)),
                min_eta_sig)
            # eta_sig = np.fmin(eta_sig, max_eta_sig)
            prec_sig = np.linalg.norm((sig-sig_old) / (sig+1e-16))

            # ax2.cla()
            # ax2.bar(xs, pix_count)
            # ax2.plot(xs, ys, 'r')
            # plt.draw()
            # plt.pause(0.01)

            # print(sig, d_sig, eta_sig)
            if prec_sig < n*prec_thresh:
                # print("break")
                break

        d_mu, d_sig, d_om = grad_loss(xs, ys, pix_count, mu, sig, om)
        eta_mu = np.fmax(np.abs(np.dot(mu - mu_old, d_mu - d_mu_old) / np.dot(d_mu - d_mu_old, d_mu - d_mu_old)),
                     min_eta_mu)
        eta_sig = np.fmax(
            np.abs(np.dot(sig - sig_old, d_sig - d_sig_old) / np.dot(d_sig - d_sig_old, d_sig - d_sig_old)),
            min_eta_sig)
        eta_om = np.fmax(np.abs(np.dot(om - om_old, d_om - d_om_old) / np.dot(d_om - d_om_old, d_om - d_om_old)),
                     min_eta_om)
        prec_all = np.linalg.norm((mu-mu_oldest) / (mu+1e-16)) \
                   + np.linalg.norm((sig-sig_oldest) / (sig+1e-16)) \
                   + np.linalg.norm((om-om_oldest) / (om+1e-16))
        # print(prec_all)
        # print(np.linalg.norm(ys_oldest-ys)/np.linalg.norm(ys))
        # print(np.linalg.norm(pix_count-ys)/np.linalg.norm(ys))
        if (j > 5) \
           & (prec_all < 3*n*prec_thresh):
            print("final break")
            print(j)
            break
    print("Time: {}".format(time.time()-t))


    ax2 = plt.subplot(212)
    ax2.bar(xs, pix_count)
    ax2.plot(xs, ys, 'r')

    return mu, sig, om


def pixel_cluster(img, num_bins=256, smooth=3):
    """Naive clustering of pixels. Splits pixel intensity distribution on minima, and assigns pixel groups the location
    of the intermediate maxima.

    img: Array of pixel values. must be black and white (i.e. dimension n x m x 1) with integer pixel values between
        0 to 255.
    num_bins: Positive integer. Preferably divides 256. Number of bins for histogram distribution of pixel intensities.
    smooth: Odd number. Indicates size of moving average for smoothing of pixel distribution. A larger smooth factor
        decreases the number of maxima and minima, but can damage quality of clustering if too high.
    """
    # Collects pixel counts
    bin_counts = np.zeros([num_bins])
    bins = [round(x*256.0/num_bins) for x in range(num_bins+1)]
    for j in range(num_bins):
        bin_counts[j] = np.count_nonzero((img >= bins[j]) & (img < bins[j+1]))
    # Smooths histogram using moving average of size smooth
    assert (smooth % 2) == 1, "smooth must be an odd number."
    if smooth != 1:
        ext_bin_counts = np.append(np.zeros((smooth-1)//2), np.append(bin_counts, np.zeros((smooth-1)//2)))
        bin_counts = np.zeros(num_bins)
        for j in range(smooth):
            bin_counts += ext_bin_counts[j:num_bins+j]/smooth
    # Finds minima and maxima by searching for zero-crossings in the derivative
    a = np.ones(num_bins)
    der = np.diag(a, 0) - np.diag(a[:-1], -1)  # Derivative operator
    d_bin_counts = np.matmul(der, bin_counts).tolist()
    nonzeros = [(j, x) for j, x in enumerate(d_bin_counts) if x]
    maxs = []
    mins = [bins[0]]  # Always include 0
    for k, (j, x) in enumerate(nonzeros):
        if k == 0:
            if x < 0:
                maxs.append(bins[0])
        else:
            if k == (len(nonzeros) - 1):
                if x > 0:
                    maxs.append((bins[-1]+bins[-2])//2)
            if (x*nonzeros[k-1][1]) < 0:
                if x < 0:
                    maxs.append((bins[j]+bins[nonzeros[k-1][0]])//2)
                elif x > 0:
                    mins.append((bins[j]+bins[nonzeros[k-1][0]])//2)
    fig = plt.figure(0)
    fig.clf()
    ax1 = plt.subplot(211)
    ax1.bar(bins[1:], bin_counts)
    # Group pixels into clusters around maxima. Minima constitute seperation points
    new_img = np.copy(img)
    for j, m in enumerate(mins):
        new_img = np.where(img >= m, maxs[j], new_img)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)
    ax2.imshow(img)
    ax3.imshow(new_img)
    return new_img

# TODO Finish cursory comments
# TODO Add Corner, Blob, Ridge Detection
# TODO Add Scale Invariant Feature Transform
# TODO Investigate more advanced techniques from https://en.wikipedia.org/wiki/Feature_extraction
