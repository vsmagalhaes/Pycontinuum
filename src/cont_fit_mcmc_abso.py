#import my_library as ml
import my_library as ml
import mcmc_utils as mcu
import time
import argparse
import sys
import emcee
from datetime import datetime
from multiprocessing import cpu_count as ncpu
import numpy as np
import os

nthreads = ncpu()

parser = argparse.ArgumentParser(description="MCMC tool to fit the continuum")
parser.add_argument("input", type=str, help="initial guesses, alpha, plateau(''), density ratio")
parser.add_argument("outputname", type=str, help="output fits file containing the markov chain")
parser.add_argument("-n","--nwalkers", type=int, default = 30, help="Number of walkers")
parser.add_argument("-s","--steps", type=int, default = 100, help="Number of MCMC iterations")
parser.add_argument("-i","--initialize", type = str, default = None, help = "Start Markov chain from previous iteration")
parser.add_argument("-p","--positionangle", type = float, default = 0.0, help = "position angle of the aperture ellipsis")
parser.add_argument("-r","--aspectratio", type = float, default = 1.0, help = "aspect ratio of the aperture ellipsis")
parser.add_argument("-c","--cut", default = False, action="store_true", help = "aspect ratio of the aperture ellipsis")

args = parser.parse_args()

inpfile = args.input
nwalkers = args.nwalkers
nit = args.steps
itheta,priorinf,priorsup,ndim,fixed,fix,log = mcu.read_input_sel(inpfile)
legends = mcu.read_legend(inpfile)

### Noise from image headers
# obs250 = ml.fits_obs("her_250_crop_1deg.fits", beam=17.6, freq=1.21514e12, noise=1.477535476189406,
#                      name="Herschel-SPIRE 250 $\mu$m")
# obs350 = ml.fits_obs("her_350_crop_1deg.fits", beam=23.9, freq=0.86543e12, noise=0.7096869232792675,
#                      name="Herschel-SPIRE 350 $\mu$m")
# obs500 = ml.fits_obs("her_500_crop_1deg.fits", beam=35.2, freq=0.61018e12, noise=0.367542589273311,
#                      name="Herschel-SPIRE 500 $\mu$m")
# obs1300 = ml.fits_obs("L1498_1250_fixed.fits", beam=11, freq=240e9, noise=1.8, name="IRAM-30m MAMBO 1.25 mm")

# Noise from gaussian fits to the image histogram
obs250 = ml.fits_obs("her_250_crop_1deg.fits", beam=17.6, freq=1.21514e12, noise=4.1,
                     name="Herschel-SPIRE 250 $\mu$m")
obs350 = ml.fits_obs("her_350_crop_1deg.fits", beam=23.9, freq=0.86543e12, noise=2.2,
                     name="Herschel-SPIRE 350 $\mu$m")
obs500 = ml.fits_obs("her_500_crop_1deg.fits", beam=35.2, freq=0.61018e12, noise=1.0,
                     name="Herschel-SPIRE 500 $\mu$m")
obs1300 = ml.fits_obs("L1498_1250_fixed_sc_rp.fits", beam=11, freq=240e9, noise=0.6,
                     name="IRAM-30m MAMBO 1.25 mm")



# if args.cut:
print "Using cut"
obs = [obs250, obs350, obs500]
#obs=[obs500]
for i in range(len(obs)):
    #        Center in RA DEC (degrees) "direction of cut" Maximal radius
    obs[i].cut_profile([62.7175, 25.17222222], (-1, -2),200)
    print obs[i].radii
obs1300.cut_profile(np.array([62.7175, 25.17222222]), (-1, -2), 130)
print obs1300.radii
obs.append(obs1300)


# else:
#     ### out of date
#     print "not using cut"
#     obs = [obs250,obs350,obs500]
#     centers = [(301,301),(181,181),(130,130)]
#     for i in range(3):
#         obs[i].radial_profile(centers[i],17,500,pa=args.positionangle,ratio=args.aspectratio)
#
#     obs1300.radial_profile([55.5,62.0],11,150,pa=args.positionangle,ratio=args.aspectratio)
#     obs.append(obs1300)

initial = args.initialize
#exit()
if initial != None:
    pos, prevchain = mcu.get_pos_and_chain(initial)
else:
    gauss = 1e-3  # we choose to initialize the values in a gaussian ball, this is its width.
    pos = [itheta + gauss * np.random.rand(ndim) for i in range(nwalkers)]  # initial positions for each walker
    prevchain = None

#start = time.time()

sampler = emcee.EnsembleSampler(nwalkers, ndim, ml.lnprob_abso,args=(priorinf,priorsup,fixed,fix,log,obs),threads=nthreads)

mcu.run_sampler(sampler,pos,nit,prevchain,args.outputname)

print "\n\nPlotting walkers\n"

mcu.plot_walkers(args.outputname, legends)

print "Making corner plot\n"

mcu.run_corner(args.outputname,[nit/2],legends)

answer = mcu.bestfit(args.outputname, legends, log[~fix], burnin=int(nit/2), save=True, plot=True,fontsize=10)

#mcu.took(start)
