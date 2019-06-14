from astropy.io import fits
import numpy as np
import corner
from matplotlib import pyplot as plt
import string
import random
import time
import datetime
import sys
import os
from scipy.optimize import curve_fit
import math

def run_corner(prefix, burn_in, legend, plotlims=1.0, fontsize=28):
    hdulist = fits.open(prefix + '.fits')

    # chains
    chain = hdulist[1].data
    # likelihood
    ndim = len(legend)
    answers = []
    try:
        len(plotlims)
    except:
        plotlims = np.zeros(ndim)+plotlims
    for n in burn_in:
        
        samples = chain[:, n:, :].reshape((-1, ndim))
        fig = corner.corner(samples, labels=legend, show_titles=True,
                            title_kwargs={"fontsize": fontsize}, range=plotlims)
        fig.savefig(prefix + '_' + str(n) + '_corner.pdf', format='PDF')
    return

def plot_walkers(prefix, legend, fontsize = 20):
    """
    Plot and save the evolution of the remainingtimeerent parameters through the markov chain
    """

    hdulist = fits.open(prefix+".fits")
    chain = hdulist[1].data
    adim = chain.shape
    nwalkers = adim[0]
    nit = adim[1]
    ndim = adim[2]
    nlines,ncolumn = plotfactors(ndim)
    if nlines * ncolumn >= ndim:
        X = np.linspace(1, nit, nit)
        fig = plt.figure(figsize=(20,20))

        if len(legend) != ndim:
            print('Watch out for your legend ! \n')
        else:
            for j in range(ndim):
                ax = plt.subplot(nlines, ncolumn, j + 1)
                for i in range(nwalkers):
                    plt.plot(X, chain[i, 0:nit , j])
                plt.ylabel(legend[j],fontsize=fontsize)
                plt.xlabel('number of steps',fontsize=fontsize)
                for label in ax.get_xticklabels()+ax.get_yticklabels():
                    label.set_fontsize(fontsize)

            fig.set_tight_layout(True)
            fig.savefig(prefix + '_walkers.pdf', format='PDF')
    else:
        print('I need more place to plot\n')

def savechain(chain,name,steps,totsteps,initial=None):
    if initial is None:
        if steps == totsteps:
            outputchain = chain
        else:
            adim = chain.shape
            nwalkers = adim[0]
            ndim = adim[2]
            outputchain = np.ndarray([nwalkers,steps,ndim])
            for i in range(nwalkers):
                for j in range(steps):
                    for k in range(ndim):
                        outputchain[i,j,k] = chain[i,j,k]
    else:
        adim = initial.shape
        nwalkers = adim[0]
        isteps = adim[1]
        ndim = adim[2]
        totalsteps = steps+isteps
        outputchain = np.ndarray([nwalkers,totalsteps,ndim])
        for i in range(nwalkers):
            for j in range(isteps):
                for k in range(ndim):
                    outputchain[i, j, k] = initial[i, j, k]
            for j in range(isteps,totalsteps):
                for k in range(ndim):
                    outputchain[i, j, k] = chain[i, j-isteps, k]
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=outputchain))  # save the markov chain in an array (nwalkers, nit,ndim)
    fits_name = name + '.fits'
    if os.path.isfile(fits_name):
        os.remove(fits_name)
    hdul.writeto(fits_name)

def get_pos_and_chain(fitsfile):
    hdulist = fits.open(fitsfile)
    chain = hdulist[1].data
    adim = chain.shape
    nwalkers = adim[0]
    pos = [chain[i][-1] for i in range(nwalkers)]  # initial positions for each walker from fits file
    return pos,chain

def cut_fits(ifits,ofits,limit):
    pos,chain = get_pos_and_chain(ifits)
    savechain(chain,ofits,limit,1e20,None)

def plotfactors(ndim):
    if ndim == 1:
        return 1, 1
    elif ndim == 2:
        return 2, 1
    elif ndim == 3:
        return 3, 1
    for i in range(int(np.sqrt(ndim)),1,-1):
        for j in range(int(np.sqrt(ndim)),1,-1):
            if (i*j)==ndim:
                return i,j
            elif (i*j)<ndim:
                if i*(j+1)>=ndim:
                    return i,j+1
                else:
                    return i+1, j + 1

    return -1,-1

def read_input_sel(filein):
    pars,prinf,prsup=np.loadtxt(filein,usecols=[0,1,2],unpack=True)
    fix,log = np.loadtxt(filein,usecols=[3,4],unpack=True,dtype=bool)
    itheta = pars[~fix]
    fixed = pars[fix]
    priorinf = prinf[~fix]
    priorsup = prsup[~fix]
    ndim = len(itheta)
    return itheta,priorinf,priorsup,ndim,fixed,fix,log

def read_legend(filein):
    legends = np.loadtxt(filein, usecols=[1], unpack=True, dtype=str,delimiter="|")
    fix,log = np.loadtxt(filein, usecols=[3,4], unpack=True, dtype=bool)
    legends = legends[~fix]
    return legends

def completepset(theta,fixed,fix,log):
    ifix = 0
    ipar = 0
    modelpars = []
    for i in range(len(fix)):
        if fix[i]:
            pars = fixed[ifix]
            ifix += 1
        else:
            pars = theta[ipar]
            ipar += 1
        if log[i]:
            pars = 10 ** (pars)
        modelpars.append(pars)
    return modelpars

def random_str(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

def etf(start,nit,ntotal):
    now = time.time()
    duration = now-start
    tottime = ntotal*duration/nit
    remainingtime = (1.0-1.0*nit/ntotal)*tottime
    if remainingtime < 60:
        return "will end in: {0:5.2f} seconds".format(remainingtime)
    elif remainingtime < 3600:
        return "will end in: {0:5.2f} minutes".format(remainingtime/60.)
    elif remainingtime < 86400:
        return "will end in: {0:5.2f}   hours".format(remainingtime/3600.)
    else:
        return "will end in: {0:5.2f}    days".format(remainingtime/86400.)

def took(ref):
    now = time.time()
    diff = now - ref
    if diff < 60:
        print "it took: {0:.4f} seconds.\n".format(diff)
    elif diff < 3600:
        print "it took: {0:.2f} minutes.\n".format(diff/60.)
    elif diff < 86400:
        print "it took: {0:.2f} hours.\n".format(diff/3600.)
    else:
        print "it took: {0:.3f} days.\n".format(diff/86400.)

def progress(ref):
    now = time.time()
    diff = now - ref
    if diff < 60:
        return "Time since start: {0:5.2f} seconds".format(diff)
    elif diff < 3600:
        return "Time since start: {0:5.2f} minutes".format(diff/60.)
    elif diff < 86400:
        return "Time since start: {0:5.2f}   hours".format(diff/3600.)
    else:
        return "Time since start: {0:5.2f}    days".format(diff/86400.)

def initialize(initial, itheta, ndim, nwalkers):
    if initial != None:
        pos, prevchain = get_pos_and_chain(initial)
    else:
        gauss = 1e-2  # we choose to initialize the values in a gaussian ball, this is its width.
        pos = [itheta + gauss * (np.random.rand(ndim)-0.5) for i in range(nwalkers)]  # initial positions for each walker
        prevchain = None

    return pos, prevchain

def run_sampler(sampler, pos, nit, prevchain, outputname):
    start = time.time()
    print("\nStarting MCMC fitting\n")
    print("Time: "+str(datetime.datetime.now()))
    print
    print("**************************************\n")
    print("Progress:\n")
    width = 20
    for i, (pos, lnp, state) in enumerate(sampler.sample(pos, iterations=nit)):
        n = int((width+1) * float(i) / nit)
        sys.stdout.write("\r{3:s}, {2}/{4} - [{0}{1}]"
                         "".format('#' * n, ' ' * (width - n), i+1, etf(start, i+1, nit), nit))
        savechain(sampler.chain, "tmp_fits", i+1, nit, prevchain)
    sys.stdout.write("\r{0:s}, 100.0% - [####################]\n".format(progress(start)))

    savechain(sampler.chain, outputname, nit, nit, prevchain)
    os.system("rm tmp_fits.fits")

    took(start)

def bestfit(prefix, legends, log, burnin=None, save=False, plot=False, fontsize=10):
    hdulist = fits.open(prefix+".fits")
    chain = hdulist[1].data
    adim = chain.shape
    nwalkers = adim[0]
    nsteps = adim[1]
    ndim = adim[2]
    if burnin is None:
        burnin = nsteps/2
    samples = chain[:, burnin:, :].reshape((-1, ndim))

    xs = np.atleast_1d(samples)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"
    if plot:
        figure = plt.figure()         
    val, sigma = [], []
    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()
        yhis,xhis = np.histogram(x,30)
        xhis += (xhis[1]-xhis[0])/2
        xhis = xhis[:-1]
        q_16, q_84 =  quantile(x, (0.16,0.84))
        mindex = np.where(yhis==np.max(yhis))
        mode = xhis[mindex][0]
        mode =  quantile(x, 0.5)[0]
        sig = np.sqrt(np.average((xhis - mode)**2, weights=yhis))
        sig = (q_84-q_16)/2.355
        val.append(mode)
        sigma.append(sig)    
        if plot:
            plt.subplot(*subploting(ndim,i))
            plt.hist(x,bins=30)
            plt.yticks([])
            plt.xticks( fontsize=fontsize)
            preci = int(np.ceil(-np.log10(sigma[i])))
            if preci < 0:
                preci = 0
            title = "={0:."+str(preci)+"f}$\pm${1:."+str(preci)+"f}"
            plt.title(legends[i]+title.format(val[i],sigma[i]),fontsize=fontsize)
            xfit = np.linspace(xhis[0],xhis[-1])
            yfit = gauss(xfit,np.max(yhis),mode,sig)
            plt.plot(xfit,yfit,color="red")
    if save:
        nout = prefix+'_ans.dat'
        outstr = "#Parameter[1] Value[2] Sigma[3]\n"
        for i in range(len(val)):
            outstr += "{0:s}\t{1:.3e}\t{2:.3e}\n".format(legends[i], val[i], sigma[i])
        fout = open(nout,'w')
        fout.write(outstr)
        fout.close()
    if plot:
        figure.set_tight_layout(True)
        figure.savefig(prefix+"_ans.pdf", format="PDF")
    return val, sigma

def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

def subploting(nplots,pos=0):
    if nplots == 1:
        return 1, 1, 1+pos
    elif nplots == 2:
        return 1, 2, 1+pos
    elif 2<nplots<=4:
        return 2, 2, 1+pos
    elif 5<=nplots<=6:
        return 2, 3, 1+pos
    elif 7<=nplots<=9:
        return 3, 3, 1+pos
    elif 10<=nplots<=12:
        return 3, 4, 1+pos
    elif 13<=nplots<=16:
        return 4, 4, 1+pos
    elif 17<=nplots<=20:
        return 4, 5, 1+pos
    elif 21<=nplots<=25:
        return 5, 5, 1+pos
    else:
        return -1
    return -1

def gauss(x,a,x0,sig):
    return a*np.exp(-(x-x0)**2/(2*sig**2))
