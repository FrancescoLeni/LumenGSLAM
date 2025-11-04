
import torch


class GaussiansOptimizer:
    def __init__(self, lrs_dict, params, device, light=None):
        self.opt = self._init_optimizer(params, lrs_dict, light=light)
        self.device = device

    def do_step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def cat_params(self, new_params, curr_params, variables):
        """
        adds new gaussians to the optimizer's parameter lists
        """
        for k, v in new_params.items():
            group = [g for g in self.opt.param_groups if g['name'] == k][0]
            stored_state = self.opt.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
                del self.opt.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                self.opt.state[group['params'][0]] = stored_state
                curr_params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                curr_params[k] = group["params"][0]

        num_pts = new_params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.cat((variables['means2D_gradient_accum'], torch.zeros(num_pts, device=self.device).float()), dim=0)
        variables['denom'] = torch.cat((variables['denom'], torch.zeros(num_pts, device=self.device).float()), dim=0)
        variables['max_2D_radius'] = torch.cat((variables['max_2D_radius'], torch.zeros(num_pts, device=self.device).float()), dim=0)
        variables['param_update_count'] = torch.cat((variables['param_update_count'], torch.zeros(num_pts, device=self.device).float()), dim=0)
        variables['param_update_count_opacity'] = torch.cat((variables['param_update_count_opacity'], torch.zeros(num_pts, device=self.device).float()), dim=0)

        return curr_params

    def reset_opacity(self, curr_params, variables):
        """
        resets the opacity of gaussians
        """

        def inverse_sigmoid(x):
            return torch.log(x / (1 - x))

        new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(curr_params['logit_opacities']) * 0.01)}
        num_pts = curr_params['means3D'].shape[0]

        for k, v in new_params.items():
            group = [x for x in self.opt.param_groups if x["name"] == k][0]
            stored_state = self.opt.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(v)
                stored_state["exp_avg_sq"] = torch.zeros_like(v)
                del self.opt.state[group['params'][0]]

                group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
                self.opt.state[group['params'][0]] = stored_state
                curr_params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
                curr_params[k] = group["params"][0]

        # resetting everything
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=self.device).float()
        variables['denom'] = torch.zeros(num_pts, device=self.device).float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device=self.device).float()
        variables['param_update_count_opacity'] = torch.zeros(num_pts, device=self.device).float()

        return curr_params

    def prune(self, to_remove, params, variables):
        """
        removes gaussians from optimizer's parameter lists'
        """
        to_keep = ~to_remove
        for k in params.keys():
            group = [g for g in self.opt.param_groups if g['name'] == k][0]
            stored_state = self.opt.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
                del self.opt.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
                self.opt.state[group['params'][0]] = stored_state
                params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
                params[k] = group["params"][0]
        variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
        variables['denom'] = variables['denom'][to_keep]
        variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
        variables['param_update_count'] = variables['param_update_count'][to_keep]
        variables['param_update_count_opacity'] = variables['param_update_count_opacity'][to_keep]

        return params, variables

    @staticmethod
    def _init_optimizer(params, lrs, light=None):
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k.split('__')[0]]} for k, v in params.items() if k != 'feature_rest']
        if 'feature_rest' in params:
            param_groups.append({'params': [params['feature_rest']], 'name': 'feature_rest', 'lr': lrs['rgb_colors'] / 20.0})

        # pbr
        if isinstance(light, torch.Tensor):
            param_groups.append({'params': [light], 'name': 'light', 'lr': lrs['light']})


        # if 'cam_trans_delta' in list(params.keys()):
        #     return torch.optim.AdamW(param_groups, lr=0.0, eps=1e-15, weight_decay=0.1)
        # else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

