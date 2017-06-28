"""Visualization functions."""
import numpy as np


__all__ = ['PacPlot']


class PacPlot(object):
    """Main PAC plotting class."""

    def pacplot(self, pac, xvec, yvec, xlabel='', ylabel='', cblabel='',
                title='', cmap='viridis', vmin=None, vmax=None, under=None,
                over=None, bad=None, pvalues=None, p=0.05, interp=None,
                rmaxis=False, dpaxis=False, plotas='imshow', ncontours=5,
                levels=None, levelcmap='Reds', polar=False, y=1.02,
                subplot=111):
        """Main plotting pac function.

        This method can be used to plot any 2D array.

        Args:
            pac: np.ndarray
                A 2D array.

            xvec: np.ndarray
                The vector to use for the x-axis.

            yvec: np.ndarray
                The vector to use for the y-axis.

        Kargs:
            xlabel: string, optional, (def: '')
                Label for the x-axis.

            ylabel: string, optional, (def: '')
                Label for the y-axis.

            cblabel: string, optional, (def: '')
                Label for the colorbar.

            title: string, optional, (def: '')
                Title of the plot.

            y: float, optional, (def: 1.02)
                Title location.

            cmap: string, optional, (def: 'viridis')
                Name of one Matplotlib's colomap.

            vmin: float, optional, (def: None)
                Threshold under which set the color to the uner parameter.

            vmax: float, optional, (def: None)
                Threshold over which set the color in the over parameter.

            under: string, optional, (def: 'gray')
                Color for values under the vmin parameter.

            over: string, optional, (def: 'red')
                Color for values over the vmax parameter.

            bad: string, optional, (def: None)
                Color for non-significant values.

            pvalues: np.ndarray, optional, (def: None)
                P-values to use for masking PAC values. The shape of this
                parameter must be the same as the shape as pac.

            p: float, optional, (def: .05)
                If pvalues is pass, use this threshold for masking
                non-significant PAC.

            interp: tuple, optional, (def: None)
                Tuple for controlling the 2D interpolation. For example,
                (.1, .1) will multiply the number of row and columns by 10.

            rmaxis: bool, optional, (def: False)
                Remove unecessary axis.

            dpaxis: bool, optional, (def: False)
                Despine axis.

            plotas: string, optional, (def: 'imshow')
                Choose how to display the comodulogram, either using imshow
                ('imshow') or contours ('contour'). If you choose 'contour',
                use the ncontours parameter for controlling the number of
                contours.

            ncontours: int, optional, (def: 5)
                Number of contours if plotas is 'contour'.

            levels: list, optional, (def: None)
                Add significency levels. This parameter must be a sorted list
                of p-values to use as levels.

            levelcmap: string, optional, (def: Reds)
                Colormap of signifiency levels.

        Returns:
            gca: axes
                The current matplotlib axes.
        """
        # Check if pac is 2 dimensions :
        if pac.ndim is not 2:
            raise ValueError("The PAC variable must have two dimensions.")
        # Try import matplotlib :
        try:
            import matplotlib.pyplot as plt
        except:
            raise ValueError("Matplotlib not installed.")
        # Define p-values (if needed) :
        if pvalues is None:
            pvalues = np.zeros_like(pac)
        # 2D interpolation (if needed)
        if interp is not None:
            pac, yvec, xvec = mapinterpolation(pac, xvec, yvec,
                                               interp[0], interp[1])
            pvalues = mapinterpolation(pvalues, self.xvec, self.yvec,
                                       interp[0], interp[1])[0]
        pac = np.ma.masked_array(pac, mask=pvalues >= p)
        # Polar plot :
        if polar:
            plotas = 'pcolor'
            plt.subplot(subplot, projection='polar')
        # Plot type :
        toplot = pac.data if levels is not None else pac
        if plotas is 'imshow':
            im = plt.imshow(toplot, aspect='auto', cmap=cmap, origin='upper',
                            vmin=vmin, vmax=vmax, interpolation='none',
                            extent=[xvec[0], xvec[-1], yvec[-1], yvec[0]])
            plt.gca().invert_yaxis()
        elif plotas is 'contour':
            im = plt.contourf(xvec, yvec, toplot, ncontours, cmap=cmap,
                              vmin=vmin, vmax=vmax)
        elif plotas is 'pcolor':
            im = plt.pcolormesh(xvec, yvec, toplot, cmap=cmap, vmin=vmin,
                                vmax=vmax)
        else:
            raise ValueError("The plotas parameter must either be 'imshow' or "
                             "'contour'")

        # Add levels :
        if levels is not None:
            plt.contour(pvalues, extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
                        levels=levels, cmap=levelcmap)

        # Under/Over/Bad :
        if under is not None:
            im.cmap.set_under(color=under)
        if over is not None:
            im.cmap.set_over(color=over)
        if bad is not None:
            im.cmap.set_bad(color=bad)

        # Title/Xlabel/Ylabel :
        plt.axis('tight')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, y=y)
        plt.clim(vmin=vmin, vmax=vmax)

        # Colorbar
        cb = plt.colorbar(im, shrink=0.7, pad=0.01, aspect=10)
        cb.set_label(cblabel)
        cb.outline.set_visible(False)
        ax = plt.gca()

        # Remove axis :
        if rmaxis:
            for loc, spine in ax.spines.items():
                if loc in ['top', 'right']:
                    spine.set_color('none')
                    ax.tick_params(**{loc: 'off'})

        # Despine axis :
        if dpaxis:
            for loc, spine in ax.spines.items():
                if loc in ['left', 'bottom']:
                    spine.set_position(('outward', 10))
                    spine.set_smart_bounds(True)

        return plt.gca()

    def comodulogram(self, pac, xlabel='Frequency for phase (hz)',
                     ylabel='Frequency for amplitude (hz)',
                     cblabel='PAC values', **kwargs):
        """Plot PAC using comodulogram.

        Args:
            pac: np.ndarray
                PAC array of shape (namp, pha)

        Kargs:
            xlabel: string, optional, (def: 'Frequency for phase (hz)')
                Label for the phase axis.

            ylabel: string, optional, (def: 'Frequency for amplitude (hz)')
                Label for the amplitude axis.

            cblabel: string, optional, (def: 'PAC values')
                Colorbar.

            kwargs:
                Further arguments are passed to the pacplot() method.

        Returns:
            gca: axes
                The current matplotlib axes.
        """
        xvec, yvec = self.xvec, self.yvec
        return self.pacplot(pac, xvec, yvec, xlabel, ylabel, cblabel,
                            **kwargs)

    def triplot(self, pac, fvec, tridx, xlabel='Starting frequency (hz)',
                ylabel='Ending frequency (hz)', cblabel='PAC values',
                bad='lightgray', **kwargs):
        """Triangular plot.

        The triplot method is used to find the [starting, ending] frequency
        either for the phase or for the amplitude.

        Args:
            pac: np.ndarray
                Pac array of shape (namp, npha)

            fvec: np.ndarray
                The frequency vector returned by the PacTriVec function.

            tridx: np.ndarray
                The index vector used to build the triangle. This argument is
                also returned by the PacTriVec function.

        Kargs:
            xlabel: string, optional, (def: 'Starting frequency (hz)')
                Label for the phase axis.

            ylabel: string, optional, (def: 'Ending frequency (hz)')
                Label for the amplitude axis.

            cblabel: string, optional, (def: 'PAC values')
                Colorbar.

            bad: string, optional, (def: 'lightgray')
                Color for non-significant values.

            kwargs:
                Further arguments are passed to the pacplot() method.

        Returns:
            gca: axes
                The current matplotlib axes.
        """
        pac, tridx = np.squeeze(pac), np.squeeze(tridx)
        # ___________________ CHECKING ___________________
        # Check if pac is a raw vector :
        if pac.ndim is not 1:
            raise ValueError("The PAC variable must be a row vector.")
        if len(pac) != tridx.shape[0]:
            raise ValueError("PAC and tridx variables must have the same "
                             "length.")

        # ___________________ RECONSTRUCT ___________________
        npac = tridx.max() + 2
        rpac = np.zeros((npac, npac), dtype=float)
        for num, k in enumerate(tridx):
            rpac[k[0], k[1]] = pac[num]
        # Build mask :
        mask = np.zeros_like(rpac, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        # Mask the lower triangle :
        rpac = np.ma.masked_array(np.flipud(rpac), mask=mask)

        # ___________________ PLOT ___________________
        # Define frequency vector :
        vector = fvec[tridx[:, 0] == 0, 0]
        xvec = yvec = np.append(vector, [fvec.max()])
        return self.pacplot(rpac, xvec, yvec, xlabel, ylabel, cblabel,
                            bad=bad, **kwargs)

    def polar(self, amp, xvec, yvec, interp=None, **kwargs):
        """Polar representation.

        This method is used to visualize amplitude as a function of phase using
        a polar (circle) representation.

        Args:
            amp: np.ndarray
                2D array.

            xvec: np.ndarray
                Vector for the x-axis.

            yvec: np.ndarray
                Vector for the y-axis (phases).

        Kargs:
            interp: float, optional, (def: None)
                Interplation factor.

            kwargs:
                Further arguments are passed to the pacplot() method.

        Returns:
            gca: axes
                The current matplotlib axes.
        """
        # Interpolation :
        if interp is not None:
            amp, yvec, xvec = mapinterpolation(amp, yvec, xvec, interp, 1)
        return self.pacplot(amp, xvec, yvec, polar=True, **kwargs)

    def show(self):
        """Display the figure."""
        import matplotlib.pyplot as plt
        plt.show()

    def savefig(self, filename, dpi=600):
        """Save the figure.

        Args:
            filename: string
                The name of the figure to save.

        Kargs:
            dpi: int, optional, (def: 600)
                DPI of the figure.
        """
        import matplotlib.pyplot as plt
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')


def mapinterpolation(data, x=None, y=None, interpx=1, interpy=1):
    """Interpolate a 2D map."""
    # Get data size :
    dim2, dim1 = data.shape
    # Define xticklabel and yticklabel :
    if x is None:
        x = np.arange(0, dim1, interpx)
    if y is None:
        y = np.arange(0, dim2, interpy)
    # Define the meshgrid :
    Xi, Yi = np.meshgrid(
        np.arange(0, dim1-1, interpx), np.arange(0, dim2-1, interpy))
    # 2D interpolation :
    datainterp = interp2(data, Xi, Yi)
    # Linearly interpolate vectors :
    xvecI = np.linspace(x[0], x[-1], datainterp.shape[0])
    yvecI = np.linspace(y[0], y[-1], datainterp.shape[1])
    return datainterp, xvecI, yvecI


def interp2(z, xi, yi, extrapval=0):
    """Linear interpolation.

    This function is equivalent to interp2(z, xi, yi,'linear') in Matlab.

    Args:
        z: np.ndarray
            Array to interpolate.

        xi: np.ndarray
            Array of x coordinates where interpolation is required.

        yi: np.ndarray
            Array of y coordinates where interpolation is required.

    Kargs:
        extrapval: float, optional, (def: 0.)
            Value for out of range positions.

    Returns:
        f: np.ndarray
            Extrapolated data.
    """
    x = xi.copy()
    y = yi.copy()
    nrows, ncols = z.shape

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")

    # find x values out of range
    x_bad = ((x < 0) | (x > ncols-1))
    if x_bad.any():
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows-1))
    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    # interpolate
    one_minus_t = 1 - y
    z = z.ravel()
    f = (z[ndx] * one_minus_t + z[ndx + ncols] * y) * (1 - x) + (
        z[ndx + 1] * one_minus_t + z[ndx + ncols + 1] * y) * x

    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f
