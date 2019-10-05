import os
from pathlib import Path
from typing import Union

import numpy as np
import plotly
from gym_minigrid.minigrid import MiniGridEnv

from hrl.visualization import tools, go, ff
from hrl.visualization.utilities import plotlyfig2json


class PlotterOneHot:
    
    def __init__(self,
                 env: MiniGridEnv,
                 ):
        self.env = env
        self.action_names = [r"$\uparrow$", r"$\downarrow$", r"$\leftarrow$",
                             r"$\rightarrow$", r"$\cdot$", r"$\cross$"]
        self.unicode_actions = [u'\u2191', u'\u2193', u'\u2190', u'\u2192',
                                u'\u00B7', u'\u2717']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'gray',
                       'gray', 'gray', 'gray']
        
        grid_size = self.env.unwrapped.width
        self.valid_cells = {(i, j) for i in range(grid_size) for j in
                            range(grid_size)}
        self.grid_size = grid_size
        self.dim = (grid_size, grid_size)
    
    def plot_option_value_function(self,
                                   option_name: str,
                                   option_state_values: np.ndarray,
                                   save_path: Union[Path, str]):
        """ Plots V of an agent associated with a particular mini-grid.
        
        Creates a figure with 4 subplots c
        """
        directions = option_state_values.shape[0]
        fig = tools.make_subplots(cols=directions, rows=1)
        
        # # Create annotations for the heatmap that correspond to policy
        # symbols = [self.unicode_actions[a] for a in option_policy.tolist()]
        # symbols = np.array(symbols).reshape(self.dim).tolist()
        
        # Create a heatmap object
        for direction, state_values in enumerate(option_state_values):
            heatmap = go.Heatmap(
                z=np.flip(state_values, axis=0),
                showscale=False,
                reversescale=False,
            )
            fig.append_trace(heatmap, 1, direction + 1)
        fig['layout'].update(title=f'{option_name}', height=400, width=800)
        plotly.offline.plot(fig, filename=f'{save_path}.html')
        plotlyfig2json(fig=fig, fpath=f'{save_path}.json')
    
    def plot_dynamics_model(self,
                            option_name: str,
                            model: np.ndarray,
                            save_path: Union[Path, str]):
        """ Plots probability transition matrices for each state as a subplots.
        
        Averages over direction dimension for simpler visualization.
        """
        
        subplot_idx = [(i, j) for i in range(self.grid_size) for j in
                       range(self.grid_size)]
        fig = tools.make_subplots(cols=self.grid_size, rows=self.grid_size)
        fig = self._remove_tick_labels(fig)
        full_grid = 0.99 * self._normalize_matrix(self._get_env_grid()).T
        full_grid = np.flip(full_grid, axis=0)
        
        # Turn (4, 19, 19, 4, 19, 19) into (19*19, 19, 19)
        dim = model.shape[-2:]
        model = np.mean(model, axis=(0, 3)).reshape(np.prod(dim), *dim)
        
        for i, p_map in enumerate(model):
            if full_grid[subplot_idx[i]] != 0:
                continue
            
            heatmap = go.Heatmap(
                z=np.flip(p_map, axis=0),
                showscale=False,
            )
            fig.append_trace(heatmap, subplot_idx[i][0], subplot_idx[i][1])
        
        fig['layout'].update(title=f'{option_name} dynamics model', height=1600,
                             width=1600)
        plotly.offline.plot(fig, filename=f'{save_path}.html')
        plotlyfig2json(fig=fig, fpath=f'{save_path}.json')
        
        return fig
    
    def plot_models_sf_plotly(self, config, current_obs, successor_features,
                              option_name, option_id):
        subplot_idx = [(i, j) for i in range(self.grid_size) for j in
                       range(self.grid_size)]
        fig = tools.make_subplots(cols=self.grid_size, rows=self.grid_size)
        fig = self._remove_tick_labels(fig)
        full_grid = 0.99 * self._normalize_matrix(self._get_env_grid())
        
        for i, (current_state, sfeat) in enumerate(
            zip(current_obs, successor_features)):
            
            if np.flip(np.flip(full_grid, axis=1), axis=0)[subplot_idx[i]] != 0:
                continue
            
            sfeat = np.reshape(sfeat, config.expanded_obs_size)
            sfeat = 0.99 * self._normalize_matrix(sfeat)
            
            grid = full_grid.copy()
            for j, k in np.array(np.where(full_grid == 0)).T:
                grid[j, k] = sfeat[j, k]
            grid[self.wrapper.obs_to_state(current_state)] = 1
            
            heatmap = go.Heatmap(
                z=np.flip(grid, axis=0), showscale=False,
                colorscale=[[0., 'white'], [0.0999, self.colors[option_id]],
                            [0.1, 'grey'], [0.999, 'black'], [1., 'red']],
            )
            fig.append_trace(heatmap, subplot_idx[i][0], subplot_idx[i][1])
        
        fig['layout'].update(title=f'{option_name} SF', height=800, width=800)
        
        plotlyfig2json(fig,
                       f'{config.images_path}/learn_models/option_{option_id}_sf.json')
        plotly.offline.plot(fig,
                            filename=f'{config.images_path}/learn_models/option_{option_id}_sf.html')
        
        return fig
    
    def plot_goals_plotly(self, config):
        fig = tools.make_subplots(cols=config.nb_options, rows=1)
        full_grid = 0.99 * self._normalize_matrix(self._get_env_grid())
        axis_template = dict(showgrid=False, zeroline=False,
                             showticklabels=False, ticks='')
        
        for o in range(config.nb_options):
            
            # Normalize the goal matrix
            goals = np.reshape(config.intrinsic_motivation.goal_reward_maps[o],
                               config.expanded_obs_size)
            for j, k in np.array(np.where(full_grid == 0)).T:
                if (j, k) not in self.valid_cells:
                    goals[
                        j, k] = 0  # Ensure that invalid cells do not affect normalization
            goals = 0.099 * self._normalize_matrix(goals)
            
            # Merge the full grid with the goals heatmap
            grid = full_grid.copy()
            for j, k in np.array(np.where(full_grid == 0)).T:
                grid[j, k] = goals[j, k]
            
            heatmap = go.Heatmap(
                z=np.flip(grid, axis=0), showscale=False,
                colorscale=[[0., 'white'], [0.1, self.colors[o]],
                            [0.10001, 'grey'], [1., 'black']],
            )
            fig.append_trace(heatmap, 1, o + 1)
            fig['layout'][f'xaxis{o + 1}'].update(axis_template)
            fig['layout'][f'yaxis{o + 1}'].update(axis_template)
        
        fig['layout'].update(title='Goals', width=300 * config.nb_options,
                             height=500)
        
        return fig
    
    def plot_qs_plotly(self, action_values_options, current_states_flat, config,
                       ep):
        idx = [(i, j) for i in range(self.grid_size) for j in
               range(self.grid_size)]
        full_grid = 0.99 * self._normalize_matrix(self._get_env_grid())
        
        for option_id, (option_instance, q) in enumerate(
            zip(config.options[:-config.action_size], action_values_options)):
            
            q = np.reshape(q,
                           (*config.expanded_obs_size, config.action_size + 1))
            v = np.max(q, axis=-1)
            
            # Since the walls have high q-values, make sure to zero-out their values before normalization
            for j, k in np.array(np.where(full_grid != 0)).T:
                v[j, k] = 0
            v = 0.099 * self._normalize_matrix(v)
            
            # Create annotations for the heatmap that correspond to policy
            symbols = [
                self.unicode_actions[option_instance.target_policy(q[i, j])[0]]
                for i, j in idx]
            symbols = np.array(symbols).reshape(self.dim).tolist()
            
            # Merge the heatmap of the grid with policy values
            grid = full_grid.copy()
            for j, k in np.array(np.where(full_grid == 0)).T:
                grid[j, k] = v[j, k]
            
            # Create a heatmap object
            heatmap = ff.create_annotated_heatmap(
                np.flip(grid, axis=0), showscale=False,
                annotation_text=symbols[::-1], font_colors=['black'],
                colorscale=[[0., 'white'], [0.1, self.colors[option_id]],
                            [0.10001, 'grey'], [1., 'black']],
            )
            
            # Increase the font size so that the policy is visible
            for i in range(len(heatmap.layout.annotations)):
                heatmap.layout.annotations[i].font.size = 25
            
            # Save the figure as a json(for adding to the dashboard) and html (for quick view)
            path = f'{config.images_path}/learn_options'
            os.makedirs(f'{path}/json', exist_ok=True)
            os.makedirs(f'{path}/html', exist_ok=True)
            plotlyfig2json(heatmap,
                           fpath=f'{path}/json/qs_{option_id}_{ep}.json')
            plotly.offline.plot(heatmap,
                                filename=f'{path}/html/qs_{option_id}_{ep}.html')
        
        return []
    
    def plot_reward_model(self,
                          option_name: str,
                          model: np.ndarray,
                          save_path: Union[Path, str]):
        fig = go.Figure(
            data=go.Heatmap(
                z=np.flip(np.mean(model, axis=0), axis=0),
                showscale=False,
            ),
            layout=dict(
                title=f'reward model of "{option_name}"',
                height=1600, width=1600,
            )
        )
        plotly.offline.plot(fig, filename=f'{save_path}.html')
        plotlyfig2json(fig=fig, fpath=f'{save_path}.json')
        
        return fig
    
    @staticmethod
    def _normalize_matrix(m):
        vmin, vmax = np.min(m), np.max(m)
        m = (m - vmin) / (vmax - vmin)
        return m
    
    def _remove_tick_labels(self, fig):
        """ Removes all the tick labels from heatmaps """
        axis_template = dict(showgrid=False, zeroline=False,
                             showticklabels=False, ticks='')
        for i in range(1, self.grid_size ** 2):
            fig['layout'][f'xaxis{i}'].update(axis_template)
            fig['layout'][f'yaxis{i}'].update(axis_template)
        return fig
    
    def _get_env_grid(self):
        """ """
        full_grid = self.env.grid.encode()[:, :, 0]
        full_grid[full_grid == 4] = 1  # remove the doors
        full_grid[full_grid == 5] = 1  # remove the keys
        full_grid[full_grid == 6] = 1  # remove the balls
        full_grid[full_grid == 7] = 1  # remove the boxes
        full_grid[full_grid == 8] = 1  # remove the goal state
        full_grid[full_grid == 9] = 1  # remove the lava
        return full_grid
