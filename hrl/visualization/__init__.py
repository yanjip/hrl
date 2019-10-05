from functools import partial

import plotly.subplots as tools
import plotly.graph_objs as go
import plotly.figure_factory as ff

from hrl.visualization.plotter_one_hot import PlotterOneHot

tools.make_subplots = partial(
    tools.make_subplots,
    horizontal_spacing=0.005,
    vertical_spacing=0.005,
    print_grid=False
)
