import torch
import pdb

from cldm.cldm import ControlLDM, ControlNet
from ldm.util import instantiate_from_config, default, log_txt_as_img
from omegaconf import OmegaConf

from ldm.modules.diffusionmodules.util import (
    timestep_embedding,
)

from seq_mask_query import Mask


class SeqControlNet(ControlNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, hint, timesteps, context, seq_t, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False) # b 320
        emb = self.time_embed(t_emb) # b 1280

        # add seq_t_emb
        seq_t_emb = timestep_embedding(seq_t, self.model_channels, repeat_only=False) # b 320
        seq_t_emb = self.time_embed(seq_t_emb) # b 1280

        emb = emb + seq_t_emb
        
        # end add seq_t_emb

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class PLPModel(ControlLDM):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        
        self.mask_query = Mask()
        self.loss_mask_weights = 1.0

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # c is dict with c_crossattn (txt) and c_concat (hint)

        # add seq_t and mask to cond
        c['seq_t'] = batch['seq_t']
        c['mask'] = batch['mask']
        
        return x, c


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        seq_t = cond['seq_t'] # tensor (b, )
        mask_t = cond['mask'] # ground truth mask_t, (b, 512, 512, 1)

        mask_pred = None

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, seq_t=seq_t)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            last_hidden_state = control[-1].clone() # (bs, 1280, 8, 8)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            
            mask_pred = self.mask_query(last_hidden_state) # real img-sized mask

        return eps, mask_pred.squeeze(1), mask_t.squeeze(3)
    

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)) # b 4 64 64
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # sample x_t from x_0 using eps
        model_output, mask_pred, mask_t = self.apply_model(x_noisy, t, cond) # eps, mask_pred, mask_gt

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar: # False
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        

        # add mask query loss
        loss_mask = self.get_loss(mask_pred, mask_t, mean=False).mean([1, 2])
        loss_mask = (self.loss_mask_weights * loss_mask).mean()
        loss_dict.update({f'{prefix}/loss_mask': loss_mask})
        loss += loss_mask


        loss_dict.update({f'{prefix}/loss': loss})


        return loss, loss_dict
    

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked: # sd_locked = True
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        # add mask query params
        params += list(self.mask_query.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        return opt


    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, seq_t, mask = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c['seq_t'][:N], c['mask'][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows: # False
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample: # False
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0: # True
            # uc_cross = self.get_unconditional_conditioning(N)
            # uc_cat = c_cat  # torch.zeros_like(c_cat)
            # uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            uc_full = None

            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "seq_t": seq_t, "mask": mask},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log