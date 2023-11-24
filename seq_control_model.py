from cldm.cldm import ControlLDM, ControlNet
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class SeqControlNet(ControlNet):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)

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
    def __init__(self):
        config = OmegaConf.load('models/cldm_v15.yaml')
        config = config.model.params.seq_control_stage_config
        self.control_model = instantiate_from_config(config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # c is dict with c_crossattn (txt) and c_concat (hint)
        c['seq_t'] = batch['seq_t']
        
        return x, c

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        seq_t = cond['seq_t']

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, seq_t=seq_t)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps
