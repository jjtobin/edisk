from matplotlib.backends.backend_pdf import PdfPages
#from casahelper.utils import get_line_info
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import astropy.io.fits as fits
import numpy
import glob
import os

pdf = PdfPages('data.pdf')

# Get a list of sources.

source = "Ced110IRS4"

# A few definitions.

arcsec = 4.84813681e-06
c_l = 29979245800.0

class Transform:
    def __init__(self, xmin, xmax, dx, fmt):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.fmt = fmt

    def __call__(self, x, p):
        return self.fmt% ((x-(self.xmax-self.xmin)/2)*self.dx)

################################################################################
### Plot the continuum observations.
################################################################################

# The list of robust parameters we used.

robust_list = [2.0,1.0,0.5,0.0,-0.5,-1.0,-2.0]

# Generate a figure to put the plots in.

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8.25))

# Now, loop through and plot.

for i, ax in enumerate(axes.flatten()):
    if i >= len(robust_list):
        ax.set_axis_off()
        continue

    # Get the appropriate robust value

    robust = robust_list[i]

    # Load the continuum image.

    try:
        data, header = fits.getdata("{0:s}_SB_continuum_robust_{1:3.1f}.image."
                "tt0.fits".format(source, robust), header=True)
    except:
        ax.set_axis_off()
        continue

    image = data[0,0,:,:]

    # Get the center of the source(s).

    x0, y0 = 0., 0.

    # Plot the image.

    ticks = numpy.array([-2.0,-1.0,0.,1.0,2.0])

    N = image.shape[0]
    pixelsize = numpy.abs(header["CDELT2"])*numpy.pi/180. / \
            arcsec

    xmin, xmax = int(round(N/2-x0/pixelsize+ticks[0]/pixelsize)), \
            int(round(N/2-x0/pixelsize+ticks[-1]/pixelsize))
    ymin, ymax = int(round(N/2+y0/pixelsize+ticks[0]/pixelsize)), \
            int(round(N/2+y0/pixelsize+ticks[-1]/pixelsize))

    npix = min(xmax - xmin, N)
    if xmin < 0:
        xmin, xmax = 0, npix
    if xmax > N:
        xmin, xmax = N - npix, N
    if ymin < 0:
        ymin, ymax = 0, npix
    if ymax > N:
        ymin, ymax = N - npix, N

    # Plot the data.

    ax.imshow(image[ymin:ymax,xmin:xmax], origin="lower",\
            interpolation="nearest", cmap="inferno")

    # Add labels to the x-axis.

    transform = ticker.FuncFormatter(Transform(xmin, xmax, pixelsize, '%.1f"'))

    ax.set_xticks((ticks[1:-1]-ticks[0])/pixelsize)
    ax.set_yticks((ticks[1:-1]-ticks[0])/pixelsize)
    ax.get_xaxis().set_major_formatter(transform)
    ax.get_yaxis().set_major_formatter(transform)

    if i >= axes.size - axes.shape[1]:
        ax.set_xlabel('$\Delta$R.A. ["]', fontsize=20, labelpad=8)
    else:
        ax.set_xticklabels([])

    if i%axes.shape[1] == 0:
        ax.set_ylabel('$\Delta$Dec. ["]', fontsize=20, labelpad=12)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='both', direction='in', labelsize=20, color="white")

    ax.annotate("Robust = {0:3.1f}".format(robust), fontsize=20, color="white",\
            xy=(0.05,0.9), xycoords="axes fraction")

    # Make the axes white as well.

    for side in ["left","right","top","bottom"]:
        ax.spines[side].set_color("white")

    # Show the beam.

    bmaj = header["BMAJ"]/abs(header["CDELT1"])
    bmin = header["BMIN"]/abs(header["CDELT1"])
    bpa = header["BPA"]

    xy = ((xmax - xmin)*0.1, (ymax - ymin)*0.1)

    ax.add_artist(patches.Ellipse(xy=xy, width=bmaj, \
            height=bmin, angle=(bpa+90), facecolor="white", \
            edgecolor="black"))

# Adjust the spacing.

plt.tight_layout(pad=1.08, rect=(0.01,0.01,0.99,0.99))

# Save the figure.

pdf.savefig(fig)

# Clear the figure.

plt.close(fig)

################################################################################
### Plot the spectral line observations.
################################################################################

# Get a list of datasets to look for.

datasets = ["C18O","13CO","12CO","SO","H2CO_3_21-2_20_218.76GHz", \
        "H2CO_3_03-2_02_218.22GHz","H2CO_3_03-2_02_218.47GHz", \
        "c-C3H2_217.82","cC3H2_217.94","cC3H2_218.16","DCN","CH3OH", \
        "SiO"]

line_centers = [219.56035410,220.39868420,230.538,219.94944200,218.76006600,\
        218.22219200,218.47563200,217.82215,217.94005,218.16044,217.2386, \
        218.44006300,217.10498000]

# Loop through the sources and plot all the datasets for that source.

for dataset, line_center in zip(datasets, line_centers):
    # Check for which robust parameters are available.

    robust_list = [f.split("_")[-1].split("image")[0][0:-1] for f in \
            glob.glob("{0:s}_SB_{1:s}_robust_*.image.fits".format(source, \
            dataset))]
    
    # Now plot each of the images.

    for robust in robust_list:
        # Load in the image.

        image, header = fits.getdata("{0:s}_SB_{1:s}_robust_{2:s}.image."
                "fits".format(source, dataset, robust), header=True)

        freq = numpy.arange(image.shape[0])*header["CDELT3"] + header["CRVAL3"]

        # Load the continuum image.

        cont = fits.getdata("{0:s}_SB_continuum_robust_{1:s}.image."
                "tt0.fits".format(source, robust))[0,0]

        # Get the center of the source(s).

        x0, y0 = 0., 0.

        # Check the number of rows and number of columns.

        nrows, ncols = round(image.shape[0]**0.5), round(image.shape[0]**0.5)
        fontsize = 20

        # Loop over field of view.

        ticks_list = [numpy.array([-10.0,-5.0,0.0,5.0,10.0]), \
                numpy.array([-2.0,-1.0,0.0,1.0,2.0])]

        for ticks in ticks_list:
            # Plot the image.

            N = image.shape[1]
            pixelsize = numpy.abs(header["CDELT2"])*numpy.pi/180. / arcsec

            xmin, xmax = int(round(N/2-x0/pixelsize+ticks[0]/pixelsize)), \
                    int(round(N/2-x0/pixelsize+ticks[-1]/pixelsize))
            ymin, ymax = int(round(N/2+y0/pixelsize+ticks[0]/pixelsize)), \
                    int(round(N/2+y0/pixelsize+ticks[-1]/pixelsize))

            npix = min(xmax - xmin, N)
            if xmin < 0:
                xmin, xmax = 0, npix
            if xmax > N:
                xmin, xmax = N - npix, N
            if ymin < 0:
                ymin, ymax = 0, npix
            if ymax > N:
                ymin, ymax = N - npix, N

            # Make a figure to put it in.

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, \
                    figsize=(2*ncols,2*nrows+0.5))

            # Plot all of the channels.

            vmin = -2*numpy.nanstd(image)
            vmax = max(10*numpy.nanstd(image), numpy.nanmax(image))

            velocity = (line_center - freq/1e9) / line_center * c_l

            for i in range(nrows):
                for j in range(ncols):
                    ind = i*ncols + j

                    # If we are beyond the limits of the frequency range, turn
                    # off the axis.

                    if ind >= image.shape[0]:
                        ax[i,j].set_axis_off()
                        continue

                    # Plot the data.

                    ax[i,j].imshow(image[ind,ymin:ymax,xmin:xmax], \
                            origin="lower", interpolation="bilinear", \
                            vmin=vmin, vmax=vmax, cmap="inferno")

                    # Contour the continuum data.

                    ax[i,j].contour(cont[ymin:ymax,xmin:xmax], \
                            colors="white", levels=numpy.nanmax(cont)*\
                            (numpy.arange(5)+0.5)/5., linewidths=0.5, \
                            alpha=0.25)

                    # Add labels to the x-axis.

                    transform = ticker.FuncFormatter(Transform(xmin, xmax, \
                            pixelsize, '%.1f"'))

                    ax[i,j].set_xticks((ticks[1:-1]-ticks[0])/pixelsize)
                    ax[i,j].set_yticks((ticks[1:-1]-ticks[0])/pixelsize)
                    ax[i,j].get_xaxis().set_major_formatter(transform)
                    ax[i,j].get_yaxis().set_major_formatter(transform)

                    if i == nrows-1:
                        ax[i,j].set_xlabel('$\Delta$R.A. ["]', \
                                fontsize=fontsize, labelpad=8)
                    else:
                        ax[i,j].set_xticklabels([])
                    if j == 0:
                        ax[i,j].set_ylabel('$\Delta$Dec. ["]', \
                                fontsize=fontsize, labelpad=12)
                    else:
                        ax[i,j].set_yticklabels([])

                    ax[i,j].tick_params(axis='both', direction='in', \
                            labelsize=fontsize, color="white")

                    # Make the axes white as well.

                    for side in ["left","right","top","bottom"]:
                        ax[i,j].spines[side].set_color("white")

                    # Add the velocity to the image.

                    txt = ax[i,j].annotate(r"$v={0:3.1f}$".\
                            format(velocity[ind]/1e5), xy=(0.01,0.85), \
                            xycoords='axes fraction', fontsize=fontsize, \
                            color="white")

                    # Show the beam.

                    bmaj = header["BMAJ"]/abs(header["CDELT1"])
                    bmin = header["BMIN"]/abs(header["CDELT1"])
                    bpa = header["BPA"]

                    xy = ((xmax - xmin)*0.1, (ymax - ymin)*0.1)

                    ax[i,j].add_artist(patches.Ellipse(xy=xy, width=bmaj, \
                            height=bmin, angle=(bpa+90), facecolor="white", \
                            edgecolor="black"))

            # Add a title to the figure.

            fig.suptitle("{0:s}, Robust = {1:s}".format(dataset.replace("_",\
                    ""), robust), fontsize=2*fontsize)

            # Adjust the spacing.

            plt.tight_layout(pad=0, rect=(0.01,0.01,0.99,0.95))

            # Save the figure.

            pdf.savefig(fig)

            # Clear the figure.

            plt.close(fig)

# Close the pdf.

pdf.close()