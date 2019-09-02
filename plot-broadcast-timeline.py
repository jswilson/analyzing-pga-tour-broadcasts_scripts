import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_video_timeline(title, commercials, regular_broadcast, pt, segments):
    """ Generates a matplotlib eventplot for the given set of inputs

    This does generate a particularly large image.  This is so small sections
    of frames of captured in the output.  If you a more typically sized image,
    the lower resolution might prevent small sections from appearing at all
    in the output image.

    Parameters:
        title (string): Title of the plot
        commercials (list): List of frame numbers which are labeled "commercial"
        regular_broadcast (list): List of frame numbers which are labeled "regular-broadcast"
        pt (list): List of frame numbers which are labeled "pt"
        segments (list): List of frame numbers which are labeled "segment"
    """
    fig = plt.figure(figsize=(124,84))
    fig.patch.set_facecolor('white')
    fig, axs = plt.subplots(1,1, figsize=(124,84))
    axs.eventplot([commercials, regular_broadcast, pt, segments],
                  orientation='horizontal',
                  colors=['b', 'g', '#fcba03', 'black'],
                  lineoffsets=[1, 3, 5, 7])

    _configure_labels_and_ticks(axs)
    _configure_legend(axs)

    plt.title(title, fontdict={'fontsize': '190'}, pad=100)
    
    plt.show()

def _configure_labels_and_ticks(axs):
    """Configure the axis labels and tickmarks for the plot

    Typically, you don't need to manually set fontsizes and pads when using
    matplotlib.  In this case, we're creating a large image, so we do need
    to manually adjust the sizes.
    """
    axs.set_xlabel('Frame number', fontsize=140, labelpad=100)
    for tick in axs.xaxis.get_major_ticks():
        tick.label.set_fontsize(100)
        tick.set_pad(100)
    axs.tick_params('x', length=50, width=2, which='major')
    axs.set_xlim(left=0)
    axs.set_xlim(right=500000)

    axs.set_yticks([1,3,5,7])
    axs.set_yticklabels(['commercial', 'regular broadcast', 'playing through', 'segment'])
    for tick in axs.yaxis.get_major_ticks():
        tick.label.set_fontsize(100)
        tick.set_pad(100)

def _configure_legend(axs):
    """Configure the legend for the plot"""
    blue = mpatches.Patch(color='blue', label='commercial')
    green = mpatches.Patch(color='green', label='regular broadcast')
    yellow = mpatches.Patch(color='#fcba03', label='playing through')
    black = mpatches.Patch(color='black', label='segment')

    axs.legend( handles=[black, yellow, green, blue], prop={'size': 96}, loc="best", bbox_to_anchor=(1.20, 1.0, 0., 0.))