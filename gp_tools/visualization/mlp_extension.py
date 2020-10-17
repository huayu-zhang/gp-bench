import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Size, Divider


class FigLayoutFixed:

    def __init__(self, nrows=1, ncols=1, n_axes=1,
                 penal_width=2.5, penal_height=2, p=0.6,
                 figsize=(8.27, 11.69),
                 mar_left=1, mar_bottom=1):
        self.nrows = nrows
        self.ncols = ncols
        self.n_axes = n_axes
        self.penal_width = penal_width
        self.penal_height = penal_height
        self.p = p
        self.figsize = figsize
        self.fig_width, self.fig_height = figsize
        self.mar_left = mar_left
        self.mar_bottom = mar_bottom

        if self.nrows > 1:
            self.v = divide_penal(penal_height, self.nrows, self.mar_bottom, self.p)
        else:
            self.v = [Size.Fixed(mar_bottom), Size.Fixed(penal_height)]

        if self.ncols > 1:
            self.h = divide_penal(penal_width, self.ncols, self.mar_left, self.p)
        else:
            self.h = [Size.Fixed(mar_left), Size.Fixed(penal_width)]

    def fig(self):

        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)

        divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), self.h, self.v, aspect=False)

        if (self.nrows == 1) & (self.ncols == 1):
            axes = [axes]
            axes[0].set_axes_locator(divider.new_locator(nx=1, ny=1))
            if self.n_axes > 1:
                for i in range(self.n_axes-1):
                    axes.append(fig.add_subplot(111))
                    axes[i+1].set_axes_locator(divider.new_locator(nx=1, ny=1))

        if (self.nrows > 1) & (self.ncols == 1):
            for j in range(self.nrows):
                axes[j].set_axes_locator(divider.new_locator(nx=1, ny=j*2+1))

        if (self.nrows == 1) & (self.ncols > 1):
            for i in range(self.ncols):
                axes[i].set_axes_locator(divider.new_locator(nx=i*2+1, ny=1))

        if (self.nrows > 1) & (self.ncols > 1):
            for j in range(self.nrows):
                for i in range(self.ncols):
                    axes[j, i].set_axes_locator(divider.new_locator(nx=i*2+1, ny=j*2+1))

        return fig, axes

    def fig_xlabel_xy(self):
        return (self.penal_width * 0.5 + self.mar_left)/self.fig_width, self.mar_bottom * 0.5 / self.fig_height

    def fig_ylabel_xy(self):
        return self.mar_left * 0.5 / self.fig_width, (self.penal_height * 0.5 + self.mar_bottom)/self.fig_height


def divide_penal(penal_size, n, mar=1.0, p=0.6):
    """
    Get horizontal/vertical term for mpl_toolkits.axes_grid1.Divider
    :param penal_size: total size of penal in inch
    :param n: number of subplots on this direction (horizontal or vertical)
    :param mar: the extra margin on left or bottom
    :param p: proportion of figure rect
    :return: h or v, [list of Size.Fixed()]
    """

    ax_size = penal_size/n * p
    interval = penal_size/n * (1 - p)

    divided = [ax_size, interval] * n
    divided[0] += mar

    divided = [Size.Fixed(i) for i in divided]

    return divided
