import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets
import numpy
import sys, os
import array
import random

import transform
import model
import algo_strategy as AL



def printHeat(inV: array):
    """
    Input: Board-size array
    Prints it.
    """
    elems = [[' ' for x in range(transform.ARENA_SIZE)] for y in
            range(transform.ARENA_SIZE)]
    for p in range(transform.ARENA_VOL):
        x,y = transform.pos2_decode(p)
        v = inV[p]
        if v > 0:
            elems[y][x] = str(v)
        else:
            elems[y][x] = "0"
    grid = ["[" + " ".join(line) + "]" for line in elems]
    grid = "\n".join(grid)
    print(grid)

def plotHeat(inV):
    grid = numpy.zeros((transform.ARENA_SIZE, transform.ARENA_SIZE))
    for p in range(transform.ARENA_VOL):
        x,y = transform.pos2_decode(p)
        grid[x][y] = inV[p]
    pyplot.imshow(grid.T, origin='lower')
    pyplot.colorbar()

def printFloatReplay(path: str, turns: int):
    """
    Print something like a pressure map.
    """
    f = open(path, "rb")
    s = f.read()
    f.close()
    for i in range(turns + 1):
        a, s = transform.array_from_string(s, 'f')
    plotHeat(a)

def _load_field_from_file(path: str, ty='f'):
    if not os.path.isfile(path):
        raise ValueError('File does not exist: ' + path)
    f = open(path, "rb")
    s = f.read()
    f.close()
    result = []
    while len(s) > 0:
        a, s = transform.array_from_string(s, ty)
        result.append(a)
    return result

def _load_paths_from_file(f: str):
    f = open(f, "rb")
    s = f.read()
    f.close()
    result = []
    while len(s) > 0:
        result_turn, s = model.Path.group_fromBytes(s)
        result.append(result_turn)
    return result
            

class DiscreteSlider(widgets.Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 1)
        widgets.Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.items():
            func(discrete_val)

def interactive_ui(base_path: str):
    # Load all array file
    print("Loading data")
    stability_F     = _load_field_from_file(base_path + '/' + AL.FILE_STABILITY_F)
    stability_E     = _load_field_from_file(base_path + '/' + AL.FILE_STABILITY_E)
    stability_D     = _load_field_from_file(base_path + '/' + AL.FILE_STABILITY_D)
    pressure_self   = _load_field_from_file(base_path + '/' + AL.FILE_PRESSURE_SELF)
    pressure_enemy  = _load_field_from_file(base_path + '/' + AL.FILE_PRESSURE_ENEMY)
    barrage_self    = _load_field_from_file(base_path + '/' + AL.FILE_BARRAGE_SELF)
    barrage_enemy   = _load_field_from_file(base_path + '/' + AL.FILE_BARRAGE_ENEMY)
    proximity_self  = _load_field_from_file(base_path + '/' + AL.FILE_PROXIMITY_SELF)
    proximity_enemy = _load_field_from_file(base_path + '/' + AL.FILE_PROXIMITY_ENEMY)

    path1_self = _load_paths_from_file(base_path + '/' + AL.FILE_PATH1_SELF)
    path1_enemy = _load_paths_from_file(base_path + '/' + AL.FILE_PATH1_ENEMY)

    nTurns = len(stability_F)
    print("Turns: {}".format(nTurns))


    fig = pyplot.figure(figsize=(12,6))
    fig.subplots_adjust(hspace=0.05)
    nRows = 1
    nCols = 2
    heat_axis = fig.add_subplot(nRows, nCols, 1)

    grid = numpy.zeros((transform.ARENA_SIZE, transform.ARENA_SIZE))
    heat_im = heat_axis.imshow(grid.T,
            origin='lower', aspect='equal')
    path_axis = fig.add_subplot(nRows, nCols, 2)
    path_im = path_axis.imshow(grid.T,
            origin='lower', aspect='equal')

    def update_heat(a):
        #print("update_heat Called")
        grid = numpy.zeros((transform.ARENA_SIZE, transform.ARENA_SIZE))
        v_min = 0
        v_max = 1
        for p in range(transform.ARENA_VOL):
            x,y = transform.pos2_decode(p)
            grid[x][y] = a[p]
            v = a[p]
            if v == float('-inf') or v == float('inf'):
                v = -1
            if v < v_min: v_min = v
            if v > v_max: v_max = v
        heat_im.set_clim(vmin=v_min, vmax=v_max)
        heat_im.set_data(grid.T)
        #heat_im.set_clim()
        #print("Min: {}, Max: {}".format(v_min,v_max))

    update_heat(stability_F[0])

    def update_path(path, ty):
        grid = -numpy.ones((transform.ARENA_SIZE, transform.ARENA_SIZE))
        v_min = -1
        v_max = 1

        # Note: Enemy hazard is not computed.
        if path and not (ty == 2 and 'Enemy'== player_radio.value_selected):
            n = len(path)
            for i in range(n):
                x = path.px[i]
                y = path.py[i]
                if ty == 0:
                    grid[x][y] = path.damage_dp[i]
                elif ty == 1:
                    grid[x][y] = path.shield_dp[i]
                else:
                    grid[x][y] = path.hazard_dp[i]
                v = grid[x][y]
                if v < v_min: v_min = v
                if v > v_max: v_max = v
        
        path_im.set_clim(vmin=v_min, vmax=v_max)
        path_im.set_data(grid.T)



    fig.subplots_adjust(left=0.3,bottom=0.5,right=1.0)

    path_axis_bbox = path_axis.get_position()
    feasibility_plot_axis = fig.add_axes(
            [path_axis_bbox.xmin,0.3,path_axis_bbox.width,0.05])
            
    feasibility_plot_axis.set_autoscale_on(True)
    feasibility_plot_axis.get_xaxis().set_visible(False)
    feasibility_plot_axis.get_yaxis().set_visible(False)
    feasibility_plot, = feasibility_plot_axis.plot( \
        range(transform.ARENA_SIZE), [0] * transform.ARENA_SIZE)


    player_radio_axis = fig.add_axes([0,0.85,0.25,0.15])
    player_radio = widgets.RadioButtons(player_radio_axis,
            ('Self', 'Enemy'))

    fields_radio_axis = fig.add_axes([0,0.4,0.25,0.45])
    fields_radio = widgets.RadioButtons(fields_radio_axis,
            ('stability_F', 'stability_E', 'stability_D',
             'pressure', 'barrage', 'proximity'))
    path_radio_axis = fig.add_axes([0,0.2,0.25,0.2])
    path_radio = widgets.RadioButtons(path_radio_axis,
            ('Damage', 'Shield', 'Hazard'))

    turn_slider_axis = fig.add_axes([0.3,0.2,0.3,0.03])
    turn_slider = DiscreteSlider(turn_slider_axis, 'Turns', 0, nTurns-1,
            valinit=0, valfmt='%0.0f')
    colorbar_axis = fig.add_axes([0.3,0.3,0.3,0.03])
    fig.colorbar(heat_im, cax=colorbar_axis, orientation='horizontal')

    abscissa_slider_axis = fig.add_axes(
            [path_axis_bbox.xmin,0.2,path_axis_bbox.width,0.03])
    abscissa_slider = DiscreteSlider(abscissa_slider_axis, 'Ab.',
            0, transform.ARENA_SIZE - 1,
            valinit=0, valfmt='%0.0f')
    
    primal_button_axis = fig.add_axes([0.65,0.23, 0.05, 0.05])
    primal_button = widgets.Button(primal_button_axis, 'Primal')


    def onChange_field(val):
        turn = int(turn_slider.val)
        pname = player_radio.value_selected
        aname = fields_radio.value_selected
        #print("Event! aname: {}".format(aname))
        try:
            if   aname == 'stability_F': update_heat(stability_F[turn])
            elif aname == 'stability_E': update_heat(stability_E[turn])
            elif aname == 'stability_D': update_heat(stability_D[turn])
            elif aname == 'pressure':
                if pname == 'Self': update_heat(pressure_self[turn])
                else:               update_heat(pressure_enemy[turn])
            elif aname == 'barrage':
                if pname == 'Self': update_heat(barrage_self[turn])
                else:               update_heat(barrage_enemy[turn])
            elif aname == 'proximity':
                if pname == 'Self': update_heat(proximity_self[turn])
                else:               update_heat(proximity_enemy[turn])
            else: print("Unknown aname: {}".format(aname))
        except Exception as e:
            print(e)
        fig.canvas.draw_idle()
    def onChange_path(val):
        turns = int(turn_slider.val)
        ab = int(abscissa_slider.val)
        pname = player_radio.value_selected
        aname = path_radio.value_selected
        try:
            if   aname == 'Damage':
                if pname == 'Self': update_path(path1_self[turns][ab], ty=0)
                else:               update_path(path1_enemy[turns][ab], ty=0)
            elif aname == 'Shield':
                if pname == 'Self': update_path(path1_self[turns][ab], ty=1)
                else:               update_path(path1_enemy[turns][ab], ty=1)
            elif aname == 'Hazard':
                if pname == 'Self': update_path(path1_self[turns][ab], ty=2)
                else:               update_path(path1_enemy[turns][ab], ty=2)
            else: print("Unknown aname: {}".format(aname))
        except Exception as e:
            print(e)
        fig.canvas.draw_idle()
    def onClick_primal(event):
        turns = int(turn_slider.val)
        pname = player_radio.value_selected
        aname = path_radio.value_selected

        try:
            assert len(path1_self[turns]) == transform.ARENA_SIZE
            max_i = 0
            max_feasibility = float('-inf')
            for i in range(transform.ARENA_SIZE):
                if pname == 'Self':
                    p = path1_self[turns][i]
                else:
                    p = path1_enemy[turns][i]
                if not p:
                    continue
                assert p.evaluated
                if p.feasibility > max_feasibility:
                    max_i = i
                    max_feasibility = p.feasibility
            print("Primal at: {}".format(max_i))
            abscissa_slider.set_val(float(max_i))
        except Exception as e:
            raise
        onChange_path(None)


    def onChange(val):
        pname = player_radio.value_selected
        turns = int(turn_slider.val)
        if pname == 'Self': path_array = path1_self[turns]
        else:               path_array = path1_enemy[turns]

        feas = [(i,p.feasibility) for i,p in enumerate(path_array) if p]
        feas_x, feas_y = zip(*feas)
        feasibility_plot.set_data(feas_x, feas_y)
        feasibility_plot_axis.relim()
        feasibility_plot_axis.autoscale_view(True, True, True)

        onChange_field(val)
        onChange_path(val)

    player_radio.on_clicked(onChange)
    turn_slider.on_changed(onChange)
    fields_radio.on_clicked(onChange_field)
    abscissa_slider.on_changed(onChange_path)
    path_radio.on_clicked(onChange_path)
    primal_button.on_clicked(onClick_primal)




    pyplot.show()
    print('Plot closing')



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Sampo Visualiser')
    subparsers = parser.add_subparsers(dest='mode')
    parser_test = subparsers.add_parser('test')
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('-f', dest='path', action='store',
            help='Data File')
    parser_plot.add_argument('-i', dest='turns', action='store',
            default=1, help='Turns', type=int)

    parser_analyse = subparsers.add_parser('analyse')
    parser_analyse.add_argument('-f', dest='path', action='store',
            help='Data Path', default='.', type=str)

    args = parser.parse_args()
    print('Mode: {}'.format(args.mode))
    if args.mode == 'test':
        # Generate random ARENA-sized array.
        a = [random.randint(0,2) for x in range(transform.ARENA_VOL)]
        a = array.array('i', a)
        # Set the bottom line to 1
        for p in range(transform.ARENA_SIZE):
            x,y = transform.pos2_edge_decode(p)
            p2 = transform.pos2_encode((x,y))
            a[p2] = 5

        printHeat(a)
        plotHeat(a)
        pyplot.show()
    elif args.mode == 'plot':
        printFloatReplay(args.path, args.turns)
        pyplot.show()
    elif args.mode == 'analyse':
        print('Starting Interactive UI')
        try:
            interactive_ui(args.path)
        except Exception as e:
            print(e)
            print("Caught error ... Exiting")
    else:
        print("Unknown mode : %s" % args.mode)
