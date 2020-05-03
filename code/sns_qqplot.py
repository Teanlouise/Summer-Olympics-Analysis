from seaborn import PairGrid, color_palette, utils

from scipy.stats import rv_continuous, t

from seaborn_qqplot.utils import probability_plot

from matplotlib.pyplot import legend

import matplotlib.patches as patches

from pandas import DataFrame, Series


def qqplot(data, x=None, y=None, hue = None, hue_order=None, palette = None, kind="quantile",
            height = 2.5, aspect = 1, dropna=True, display_kws=None, plot_kws=None):
    """Draw a quantile-quantile plot of one variable against a probability distribution or
    two variables.
    Parameters
    ----------
    data : DataFrame
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    x : string, optional
        name of variables in ``data``.
    y : string or `scipy.rv_continuous` instance, optional
        name of variables in ``data`` or a probability distribution
    hue : string (variable name), optional
        Variable in ``data`` to map plot aspects to different colors.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    kind : { "quantile" | "probability"}, optional
        Kind of plot to draw. probability plots are not yet available
    height : numeric, optional
        Size of the figure (it will be square).
    aspect : numeric, optional
        Ratio of joint axes height to marginal axes height.
    dropna : bool, optional
        If True, remove observations that are missing from ``x`` and ``y``.
    {dispay, plot}_kws : dicts, optional
        Additional keyword arguments for the plot components.
    kwargs : key, value pairings
        Additional keyword arguments are passed to the function used to
        draw the plot on the joint Axes.
    Returns
    -------
    grid : :class:`PairGrid`
        :class:`PairGrid` object with the plot on it.
    """
    # only support pandas DataFrame for data
    if not isinstance(data, DataFrame):
        raise TypeError(
            "'data' must be pandas DataFrame object, not: {typefound}".format(
                typefound=type(data)))

    # x and y cannot be null values
    if x is None or y is None:
        raise TypeError("x and y cannot be of Nonetype")

    x_vars = [x]

    data_ = data.copy()

    if isinstance(y, rv_continuous):
        y_ = y(*y.fit(data_[x])).rvs(len(data_[x]))
        name = "{}_dist".format(y.name)
        data_[name] = Series(y_, index=data_.index)
        y_vars=[name]
    else:
        y_vars = [y]

    if plot_kws is None:
        plot_kws = {}
    if display_kws is None:
        display_kws = {}

    kws = {"plot_kws":plot_kws, "display_kws":display_kws}

    grid = PairGrid(data_, x_vars=x_vars, y_vars=y_vars, height=height, aspect=aspect, hue=hue, palette=palette)

    if kind == "quantile":
        f = qqplot
    #elif kind == "probability":
    #    f = ppplot
    else:
        msg = "kind must be 'quantile'" # need to add pp-plot functionality ('probability kind')
        raise ValueError(msg)

    grid.map(probability_plot, **kws)

    # Add a legend
    if hue is not None:
        labels = data[hue].unique()[::-1]
        n_colors = len(labels)
        # 'seabornic' handling of palettes
        current_palette = utils.get_color_cycle()
        if n_colors > len(current_palette):
            colors = color_palette("husl", n_colors)
        else:
            colors = color_palette(n_colors=n_colors)
        colors = colors.as_hex()[:len(labels)]
        handles = [patches.Patch(color=color, label=label) for color, label in zip(colors,labels)]
        legend(handles=handles)

    
    return grid
