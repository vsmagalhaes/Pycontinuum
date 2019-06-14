import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from scipy.ndimage.filters import generic_filter as gf
from matplotlib import pyplot as plt
import mcmc_utils as mcu
from reproject import reproject_exact
from matplotlib.ticker import FormatStrFormatter
import multiprocessing as mp

dash = '-'
gdf = '.gdf'
dat = '.dat'
spc = ' '

kboltz = 1.380658e-16  # ergs/kelvin
cc = 2.99792458e10  # cm/s
hplanck = 6.6260755e-27  # ergs*second
au2cm = 1.496e+13 # cm

gas2dust = 1 / 100. # Gas to dust ratio
mu = 2.3 # Mean molecular weight
mh = 1.6737236e-24 # hydrogen atom mass in grams

class Observation:
    def __init__(self, wavelength, direction, source):
        self.fname = source + dash + wavelength + gdf + dash + direction + dat
        self.name = source+dash+wavelength+dash+direction
        self.source = source
        self.direction = direction
        self.wavelength = wavelength
        self.dists, self.vals = np.loadtxt(self.fname, dtype=float, comments='#', unpack=True)
        #zero = np.array([0.0])
        #self.dists = np.setdiff1d(self.dists,zero)
        #self.vals = np.setdiff1d(self.vals,zero)
        self.zeroclean()
        self.sampling = np.unique(self.dists)

    def printobs(self):
        print self.source, self.wavelength, self.direction
        for i in range(len(self.dists)):
            print self.dists[i], self.vals[i]

    def zeroclean(self):
        new_vals = []
        new_dists = []
        for i in range(len(self.dists)):
            if self.dists[i] != 0.0:
                new_dists.append(self.dists[i])
                new_vals.append(self.vals[i])
        self.dists = np.array(new_dists)
        self.vals = np.array(new_vals)

class model:
    ###
    ### All distance units in arcseconds
    ###

    def __init__(self, alpha, plateau, densratio, sampling):
        self.alpha = alpha
        self.plateau = plateau
        self.densratio = densratio
        self.coluna = np.linspace(0,720,num=200)
        self.coldens = np.array(sampling)
        self.dens = np.array(sampling)
        self.sampling = np.array(sampling)
        self.chi2 = 0
        self.densityprofile = np.array(sampling)

    def calcColDens(self):
        for i in range(len(self.sampling)):
            xpos = self.sampling[i]
            dist = np.sqrt(xpos**2+self.coluna**2)
            den = 2.0 * self.denProfile(dist)
            self.coldens[i] = np.sum(den)
        colmax = np.amax(self.coldens)
        self.coldens/=colmax

    def calcDensity(self):
        self.densityprofile = self.denProfile(self.sampling)

    def denProfile(self, dist):
        ### From Tafalla et al. 2004
        return 1.0/(1.0+(dist/self.plateau)**self.alpha)+self.densratio

    def printColdens(self):
        for i in range(len(self.coldens)):
            print self.sampling[i], self.coldens[i]

    def plotDens(self):
        myfig = plt.figure(1,figsize=(9,9))
        plt.clf()
        plt.yscale('log')
        plt.xscale('log')

        plt.plot(self.sampling,self.densityprofile, label='Density')
        plt.xlabel("eixo X")
        plt.ylabel("eixo y")
        #plt.show()
        plt.legend(loc='lower left',numpoints=1)
        plt.savefig('densi.png')
        return

    def plotColDens(self):
        myfig = plt.figure(1,figsize=(9,9))
        plt.clf()
        plt.yscale('log')
        plt.xscale('log')

        plt.plot(self.sampling,self.coldens, label='Column Density')
        plt.xlabel("eixo X")
        plt.ylabel("eixo y")
        #plt.show()
        plt.legend(loc='lower left',numpoints=1)
        plt.savefig('col.png')
        return

    def plotColdenswithobs(self,name,*observations):
        myfig = plt.figure(1,figsize=(9,9))
        #plt.clf()
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Model: \nPlateau = {0:.1f}, $\\alpha$ = {1:.2f}, $\\chi^2$ = {2:.4f}"\
                  .format(self.plateau,self.alpha,self.chi2))
        colorshapes = ['ro','bo','go','r^','g^','b^']
        for i in range(len(observations)):
            observation = observations[i]
            if isinstance(observation,Observation):
                label = observation.name
                plt.plot(observation.dists,observation.vals,colorshapes[i], label=label)
            elif isinstance(observation,fits_obs):
                label = observation.name
                plt.plot(observation.radii, observation.radial, colorshapes[i], label=label)
            else:
                print "Cannot plot observation {0:d}, it is not an observation.".format(i)
        plt.plot(self.sampling,self.coldens, label=name)
        plt.xlabel("Distance [arcsecs]")
        plt.ylabel("Normalized Column Density or Intensity")
        plt.ylim([0.4,1.5])
        plt.xlim([4.0,200])
        plt.legend(loc='lower left',numpoints=1)
        plt.savefig(name+'.png')
        return

class absolute_model:
    """
    All distances inside class are in cm!
    All input distances expected in arc seconds!
    """
    def __init__(self, sampling, densitypars, temppars, size, dist,templaw="tanh"):
        self.centraldens, self.densext = densitypars[0:2]
        self.plateau,self.alpha = densitypars[2:]
        self.templaw = templaw
        if templaw== "tanh" or templaw=="arctan":
            self.centraltemp,self.exttemp = temppars[0:2]
            self.rtjump, self.deltartjump = temppars[2:]
            self.rtjump *= dist * au2cm
            self.deltartjump *= dist * au2cm
        elif templaw== "power":
            self.centraltemp,self.tempgain,self.tempexpo = temppars
        elif templaw == "Hocuk":
            temppars = None
        self.sampling = np.array(sampling)*dist*au2cm
        self.coldens = np.array(sampling)
        self.emission = np.array(sampling)

        self.plateau *= dist*au2cm

        self.size = size # in ''
        self.dist = dist # in pc

        self.coluna = np.linspace(0, size*dist*au2cm, num=250)
        self.cell = self.coluna[1]
        return

    def calcDensity(self):
        self.densityprofile = self.denProfile(self.sampling)
        return self.densityprofile

    def denProfile(self, dist):
        ### From Tafalla et al. 2004
        return self.centraldens/(1.0+(dist/self.plateau)**self.alpha)+self.densext

    def compute_av(self,dist,radius,density,dr,extav=1.0):
        #ext_radius = np.linspace(dist,self.size*self.dist*au2cm,100)
        #print np.sum(self.denProfile(ext_radius)),(ext_radius[1]-ext_radius[0])
        #ext_column = np.sum(self.denProfile(ext_radius))*(ext_radius[1]-ext_radius[0])
        ext_column = np.sum(density[radius >= dist])*dr
        av = ext_column/1.9e21 + extav
        return av

    def tempProfile_tanh(self,dist):
        ### simple temperature profile based on an hyperbolic tangent function
        output = (1. + np.tanh((dist - self.rtjump) / self.deltartjump))
        output *= (self.exttemp - self.centraltemp) / 2.
        output += self.centraltemp
        return output

    def tempProfile_arctan(self,dist):
        ### simple temperature profile based on an hyperbolic tangent function
        output = (np.pi/2.0 + np.arctan((dist - self.rtjump) / self.deltartjump))
        output *= (self.exttemp - self.centraltemp) / np.pi
        output += self.centraltemp
        return output

    def calcTemperature(self,dist):
        if self.templaw == "tanh":
            return self.tempProfile_tanh(dist)
        elif self.templaw == "power":
            return self.tempProfile_power(dist)
        elif self.templaw == "hocuk":
            return self.tempProfile_hocuk(dist)
        elif self.templaw == "arctan":
            return self.tempProfile_arctan(dist)
        else:
            return -np.inf

    def tempProfile_power(self,dist):
        return self.centraltemp+self.tempgain*(dist/(self.size*self.dist*au2cm))**self.tempexpo

    def tempProfile_hocuk(self,dist,draine=1):
        """
        Computes dust temperature based on the equation in Hocuk 2017
        :param dist: positions to compute the dust temperature profile
        :param draine: UV field intensity in draine 1978 units
        :return: dust temperature profile
        """
        radius = np.linspace(0,self.size * self.dist * au2cm,100)
        density = self.denProfile(radius)
        dr = radius[1] - radius[0]
        tdust = np.zeros(len(dist))
        for i in range(len(dist)):
            av = self.compute_av(dist[i],radius,density,dr)
            tdust[i] =(11 + 5.7*np.tanh(0.61 - np.log10(av)))*draine**(1/5.9)
        return tdust

    def emissionProfile(self,freq,beta=2,nu0=1.2e12,knu0=5.58505):
        # output in erg/cm^2/hz/strradian
        ### based on eq. 2 in Schuller 2009
        for i in range(len(self.sampling)):
            xpos = self.sampling[i]
            dist = np.sqrt(xpos**2+self.coluna**2)  # definite positive!
            temps = self.calcTemperature(dist)
            bnus = bnu(temps,freq)
            opacity = opac(freq,beta=beta,nu0=nu0,knu0=knu0)
            unitemiss = bnus*opacity*gas2dust*mh*mu
            emisspercell = 2.0 * self.denProfile(dist)*self.cell*unitemiss
            self.emission[i] = np.sum(emisspercell)
        return self.emission

    def emissionProfiling(self,freq,beta=2,nu0=1.2e12,knu0=5.58505,xpos=0,cum=True):
        # output in erg/cm^2/hz/strradian
        dist = np.sqrt(xpos**2+self.coluna**2)  # definite positive!
        temps = self.calcTemperature(dist)
        bnus = bnu(temps,freq)
        opacity = opac(freq,beta=beta,nu0=nu0,knu0=knu0)
        unitemiss = bnus*opacity*gas2dust*mh*mu
        emisspercell =  self.denProfile(dist)*self.cell*unitemiss
        if cum:
            return np.cumsum(emisspercell)
        else:
            return emisspercell

    def calcColDens(self):
        for i in range(len(self.sampling)):
            xpos = self.sampling[i]
            dist = np.sqrt(xpos**2+self.coluna**2)
            den = 2.0 * self.denProfile(dist)*self.cell
            self.coldens[i] = np.sum(den)
        return self.coldens

    def plot_emission_with_obs(self,obs,name,beta=2,nu0=1.2e12,knu0=5.58505,conversion=1e17,unit="MJy/sterradian",presentation=False):
        for ob in obs:
            figure = plt.figure()

            plt.clf()
            ax = plt.subplot(111)
            plt.yscale('log')
            plt.xscale('log')

            emisprof = conversion * self.emissionProfile(ob.freq,beta=beta,nu0=nu0,knu0=knu0)
            convemission = convolution_gauss(self.sampling, emisprof, ob.radii*self.dist*au2cm, ob.beam*self.dist*au2cm)
            plt.plot(ob.radii, convemission, color='black', label='Model')
            ### include obs
            label = ob.name
            plt.errorbar(ob.radii, ob.radial, yerr=ob.eradial, color="red", label=label, fmt=".", barsabove=False)

            if presentation:
                fsize =16
                plt.xlabel("Radial distance (arc seconds)",fontsize=fsize,weight='bold')
                plt.ylabel("Continuum intensity (" + unit + ")",fontsize=fsize,weight='bold')

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                plt.yticks(fontsize=fsize, weight='bold')
                plt.xticks(fontsize=fsize, weight='bold')

                plt.legend(loc='lower left', numpoints=1, fancybox=True, shadow=True,fontsize=fsize)
            else:
                plt.xlabel("Radial distance (arc seconds)")
                plt.ylabel("Continuum intensity ("+unit+")")
                plt.legend(loc='lower left', numpoints=1, fancybox=True,shadow=True)


            figure.set_tight_layout(True)
            savename = name + '_' + ob.name + '.pdf'
            replacelist = ['\\','$',' ']
            for item in replacelist:
                savename = savename.replace(item,'')
            plt.savefig(savename, format='pdf')
            plt.close(figure)

class fits_obs:

    def __init__(self,file,freq=1.2e12,noise=0.0,name="",beam = 0.0):
        hdulist = fits.open(file)
        if name == "":
            self.name = file
        else:
            self.name = name
        self.header = hdulist[0].header
        if self.header["NAXIS"] == 4:
            self.image = np.nan_to_num(hdulist[0].data[0][0])
            self.image = hdulist[0].data[0][0]
        elif self.header["NAXIS"] == 3:
            self.image = np.nan_to_num(hdulist[0].data[0])
            self.image = hdulist[0].data[0]
        else:
            self.image = np.nan_to_num(hdulist[0].data)
            self.image = hdulist[0].data

        nans = self.image == np.nan
        self.image[nans] = 1e6
        self.max = np.max(np.nan_to_num(self.image))
        self.min = np.min(np.nan_to_num(self.image))
        if noise == 0.0:
            self.noise = 0.1*self.max
        else:
            self.noise = noise

        self.beam = beam
        self.wcs = wcs.WCS(self.header)

        self.xscale = self.header["CDELT1"]
        self.yscale = self.header["CDELT2"]

        self.xyscale = (abs(self.xscale) / 2. + abs(self.yscale) / 2.) * 3600.
        self.size = self.header["NAXIS2"], self.header["NAXIS1"]
        #hdulist.close()
        self.filename = file
        self.freq=freq

        return

    def radial_profile(self,center,binsize,totalsize,pa=0,ratio=1):
        """
        Computes the radial profile of the image given bin size and the totalsize
        :param center: in pixels
        :param binsize: in arcseconds
        :param totsize: in arcseconds
        :return: the radial profile of the image
        """
        nsteps = int(totalsize/binsize)
        radii = np.zeros(nsteps)
        radial = np.zeros(nsteps)
        eradial = np.zeros(nsteps)
        binpix = binsize/self.xyscale
        megamask = np.zeros(self.size)
        for i in range(nsteps):
            inner,outer = i*binpix,(i+1)*binpix
            if i > 0:
                mask = anullarmask(self.image,center,self.size,inner,outer,pa=pa,ratio=ratio)
            else:
                mask = circmask(self.image,center,self.size,outer,pa=pa,ratio=ratio)
            megamask += mask
            avg = np.average(self.image,weights=mask)
            err = rms_masked(self.image,mask)
            radial[i] = avg
            eradial[i] = err
            radii[i] = (outer+inner)/2.

        self.radii = radii*self.xyscale
        self.radial = radial
        self.eradial = np.sqrt(eradial**2+self.noise**2)
        self.megamask = megamask
        return radii*self.xyscale,radial,eradial

    def cut_profile(self,center,sense,limit):
        """
        Creates cuts spaced by half a beam in size from zero to limit
        :param center: position to start the cut from (RA DEC in degrees)
        :param sense: How the cut grows Ex: (1,-2) -> 1 step in positive x 2 steps in negative y
        :param limit: ultimate size of the cut in arc seconds
        :return: Cut values
        """
        if not (sense[0] !=  1 or sense[0] != -1):
            print sense[0]
            raise Exception("sense[0] has to be 1 or -1")

        megamask = np.zeros(self.size)
        interval  = self.beam/2.0
        #print self.beam
        nsteps = int(limit/interval)
        interval /= 3600.
        sense = np.array(sense)
        center = np.array(center)
        radii = np.zeros(nsteps)
        radial = np.zeros(nsteps)
        eradial = np.zeros(nsteps)
        pos = np.array((0.0,0.0))
        for i in range(nsteps):
            alongl = i*interval
            pos[0] = sense[0]*alongl*np.sqrt(1./(1+sense[1]**2))
            pos[1] = pos[0]*sense[1]/sense[0]
            pos += center
            mask = self.beammask(pos)
            avg = np.average(np.nan_to_num(self.image), weights=mask)
            err = rms_masked(np.nan_to_num(self.image), mask)
            radii[i] = alongl
            radial[i] = avg
            eradial[i] = err
            megamask += mask
        self.megamask = megamask
        self.radii = radii *3600
        self.radial = radial
        self.eradial = np.sqrt(eradial ** 2 + self.noise ** 2)
        return  radii *3600, radial, eradial

    def beammask(self,pos):
        pixpos = self.wcs.wcs_world2pix(pos[0],pos[1],1)
        return circmask(self.image,pixpos,self.size,self.beam/2.0/self.xyscale)

    def normalize_radius(self):
        maxrprof = np.max(self.radial)
        self.radial/=maxrprof
        self.eradial /= maxrprof

    def visu_obs_mask(self):
        larger = self.megamask>1
        self.megamask[larger] =1
        masked = self.megamask*self.image
        plt.imshow(masked)
        plt.show()

    def histogram(self, name='', binsize=None, range=None):
        plt.clf()
        if name =='':
            name = self.name
        if binsize is None:
            binsize = 1e-2*self.max

        if range is None:
            range = (self.min,self.max)
        nbins = int((range[1]-range[0])/binsize)

        histo = np.histogram(self.image,bins=nbins,range=range)
        histoy = histo[0]
        histox = histo[1][:-1]+(histo[1][1]-histo[1][0])/2.
        plt.step(histox,histoy)
        plt.xlabel("Intensity")
        plt.ylabel("Counts")
        plt.savefig("histo_"+name+".png")
        return histox, histoy

def scatter_plot(obsx,obsy, name=''):
    if name == '':
        name = "dummy"

    plt.clf()
    hdux = fits.open(obsx.filename)[0]
    hduy = fits.open(obsy.filename)[0]

    if obsx.xyscale > obsy.xyscale:
        reprox = hdux.data
        reproy,dummy = reproject_exact(hduy,hdux.header)
    else:
        reproy = hduy.data
        reprox, dummy = reproject_exact(hdux, hduy.header)

    plt.scatter(reprox,reproy,s=1)
    plt.savefig("scatter_"+name+".png")

def circmask(image,center,size,radius,pa=0,ratio=1):
    rmask = np.zeros(size)
    y, x = np.ogrid[0:size[0], 0:size[1]]
    pa-=90
    pa *= np.pi/180. #convert to radians
    ### circular mask works
    #mask = (x-center[0])**2 + (y-center[1])**2 <= radius**2
    ### Eliptical mask testing

    mask = ((np.cos(pa)*(x-center[0])+np.sin(pa)*(y-center[1]))**2)/(ratio**2) + \
           ((np.sin(pa) * (x - center[0]) - np.cos(pa)*(y - center[1])) ** 2)  <= radius**2

    rmask[mask] = 1
    nans = image == np.nan
    rmask[nans] = 0
    return rmask

def anullarmask(image,center,size,inner,outer,pa=0,ratio=1):
    innermask = circmask(image,center,size,inner,pa=pa,ratio=ratio)
    outermask = circmask(image,center,size,outer,pa=pa,ratio=ratio)
    return outermask-innermask

def rms_masked(array,mask,mean=None):
    if mean is None:
        mean = np.average(array,weights=mask)
    narray = mask*(array-mean)**2
    nsamples = np.sum(mask)
    stdsum = np.sum(narray)
    return np.sqrt(stdsum/(nsamples-1))

def lnprob_norm_notkin(theta,*obs):
    """ln prob gives the ln of the posterior, calculated from the
    bayesian inference (ie) with the likelihood and the prior

    """
    lp = lnprior_norm_notkin(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_norm_notkin(theta,obs)

def lnprior_norm_notkin(theta):
    """
    Makes probability infinitilly small in the case that theta goes out of bounds.

    """
    alpha,plateau,ratio = theta
    if alpha < 0 or alpha > 6:
        return -np.inf
    if plateau < 40 or plateau > 150:
        return -np.inf
    if ratio > 1 or ratio < 0:
        return -np.inf
    return 0.0

def lnlike_norm_notkin(theta,obs):
    """
    likelihood function : probability of the observations knowing the model
    """
    sampling = obs[0].radii
    mod1 = model(theta[0],theta[1],theta[2], sampling) ## all free
    
    mod1.calcColDens()
    dist = 0
    for ob in obs:
        chivec = (mod1.coldens-ob.radial)**2/ob.eradial**2
        dist += np.sum(chivec)
    return -dist

def create_model_range(mostprobable,inflims,suplims,obs,sampling=100,type="relative"):
    if type == "absolute":
        ### NOT pratical!
        parranges = [np.linspace(mostprobable[i] - inflims[i], mostprobable[i] + suplims[i], sampling) for i in
                     range(len(mostprobable))]
        radii = obs.radii
        models = []

        for par1 in parranges[0]:
            for par2 in parranges[1]:
                for par3 in parranges[2]:
                    lemodel = model(par1, par2, par3, radii)
                    models.append(lemodel)
        return models
    else:
        parranges = [np.linspace(mostprobable[i]-inflims[i],mostprobable[i]+suplims[i],sampling) for i in range(3)]
        radii = obs.radii
        models = []
        for par1 in parranges[0]:
            for par2 in parranges[1]:
                for par3 in parranges[2]:
                    lemodel = model(par1,par2,par3,radii)
                    lemodel.calcColDens()
                    models.append(lemodel)
        return models

def plot_model_range_wobs(name,mostprobable,inflims,suplims,obs):
    figure = plt.figure()
    plt.clf()
    plt.yscale('log')
    plt.xscale('log')

    mrange = create_model_range(mostprobable,inflims,suplims,obs[0],sampling=20)

    i=0
    for lemodel in mrange:
        plt.plot(lemodel.sampling, lemodel.coldens, color = 'lightgrey', label='models within 1 $\\sigma$' if i == 0 else "")
        i+=1



    colors = ["red","blue","green"]
    i = 0
    for ob in obs:
        label = ob.name
        plt.errorbar(ob.radii, ob.radial, yerr=ob.eradial, color=colors[i], label=label,fmt = ".",barsabove=False)
        i+=1

    bmodel = model(mostprobable[0], mostprobable[1], mostprobable[2], obs[0].radii)
    bmodel.calcColDens()
    plt.plot(bmodel.sampling, bmodel.coldens, color='black', label='Best fit')

    plt.xlabel("Radial distance (arcseconds)")
    plt.ylabel("Normalized column density")

    plt.legend(loc='lower left', numpoints=1)

    figure.set_tight_layout(True)

    plt.savefig(name+'.png')

def bnu(temp, freq):
    flux = 2.*hplanck*freq**3/cc**2*1.0/(np.exp((hplanck*freq)/(kboltz*temp))-1)
    return flux

def opac(freq,beta=2,nu0=1.2e12,knu0=5.58505):
    ### values from Juvela 2015 Galactic cold cores V. Dust opacity nu =250 microns
    return knu0*(freq/nu0)**beta

def lnprob_abso(theta,priorinf,priorsup,fixed,fix,log,obs):
    """ln prob gives the ln of the posterior, calculated from the
    bayesian inference (ie) with the likelihood and the prior

    """
    lp = lnprior_abso(theta,fixed,fix,log,priorinf,priorsup)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_abso(theta,fixed,fix,log,obs)

def lnprior_abso(theta,fixed,fix,log,priorinf,priorsup):
    """
    Makes probability infinitilly small in the case that theta goes out of bounds.

    """
    for i in range(len(theta)):
        if not (priorinf[i] < theta[i] < priorsup[i]):
            return -np.inf
    return 0.0

def lnlike_abso(theta,fixed,fix,log,obs):
    """
    likelihood function : probability of the observations knowing the model
    """
    # Getting the complete set of model parameters mixed with the fixed parameters
    comppset = mcu.completepset(theta, fixed, fix, log)

    denspars = comppset[0:4]
    tkinpars = comppset[4:8]
    size = comppset[8]
    distance = comppset[9]
    beta = comppset[10]


    #from erg/cm^2/hz/strradian to Mjy/sterradian:
    erg2Mjy = 1.0e17
    conversionfac = [erg2Mjy,erg2Mjy,erg2Mjy,erg2Mjy]

    #juvela + tafalla
    #knu = [5.58505,5.58505,5.58505,0.5]
    #nu0 = [obs[0].freq,obs[0].freq,obs[0].freq,obs[3].freq]
    # juvela only
    knu = [5.58505, 5.58505, 5.58505, 5.8505]
    nu0 = [obs[0].freq, obs[0].freq, obs[0].freq, obs[0].freq]

    mid = mcu.random_str(10)
    dist = 0.0
    for i in range(len(obs)):

        beam = obs[i].beam
        # max radius for sampling of the model emission
        uncolsize = obs[0].radii[-1] + 6 * beam
        # minimal and maximal radii for sampling model emission
        uncollims = (-6 * beam, uncolsize)
        # Number of points in sampling
        nuncolspacing = int((uncollims[1] - uncollims[0]) / (beam / 2.0)) + 1
        # Array containing the sampling
        unconvol_sampling = np.linspace(uncollims[0], uncollims[1], nuncolspacing)
        # Creates model
        model = absolute_model(unconvol_sampling, denspars, tkinpars, size, distance, templaw="tanh")
        # Computes emission profile
        emissionprofile = conversionfac[i]*model.emissionProfile(obs[i].freq,beta=beta,knu0=knu[i],nu0=nu0[i])
        # Convolves emission profile with beam
        convemission = convolution_gauss(unconvol_sampling*distance*au2cm,emissionprofile,obs[i].radii*distance*au2cm,
                                         obs[i].beam*distance*au2cm)

        lock = mp.Lock()

        with lock:
            try:
                foutput = open('flux_id_val.dat','a')
                try:
                    foutput.write("{0:s}\t{1:.3e}".format(mid,obs[i].freq))
                    for fval in convemission:
                        foutput.write("\t{0:.3e}".format(fval))
                    foutput.write("\n")
                finally:
                    foutput.close()
            except:
                pass

        ## computes chi square from the obs
        chivec = (convemission - obs[i].radial) ** 2 / obs[i].eradial** 2
        # Sums up the chi squares
        dist += np.sum(chivec)

    return -dist

def cgs2jyperbeam(beamsize):
    """
    Computes conversion factors from CGS units to Jy/beam
    Assumes a gaussian beam
    :param beamsize: beam HPBW in arcseconds
    :return: conversion factor in Jy*hz*cm^2*strradian/beam/erg
    """
    constant = 1e23#cgs 2 MJy/str
    beamsize *= np.pi/180./3600. # Arcseconds to radians
    beamsolid = beamsize**2*np.pi/(4*np.log(2))
    return constant*beamsolid

def convolution_gauss(xemission,yemission,sampling,beam,beamefficiency = 1.0):
    nsamps = len(sampling)
    convolved = np.zeros(nsamps)
    step = xemission[1]-xemission[0]
    for i in range(nsamps):
        gauss = gaussian(xemission,beamefficiency,sampling[i],beam/2.355,step=step)
        conv = np.array(yemission*gauss)
        convolved[i] = np.sum(conv)
    return convolved

def gaussian(x,area,center,sigma,step=1.0):
    return area*np.exp(-(x-center)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)*step
