"""
self.action_dims = [4,
                    4, 5, 4, 4, 4, 5,
                    4, 4, 4, 4, 4, 5,
                    4, 4, 6, 4, 4, 5,
                    4, 4, 5, 4, 4, 4]
self.action_dims_Omni = [4, 5, 4, 4, 4, 5]
self.action_dims_mtad = [4, 4, 4, 4, 4, 5]
self.action_dims_InterFusion = [4, 4, 6, 4, 4, 5]
self.action_dims_SDFVAE = [4, 4, 5, 4, 4, 4]
"""
model_Omni_mask_Y = [True]
model_Omni_mask_N = [False]
"""self.action_dims_Omni = [4, 5, 4, 4, 4, 5]"""  # 26
Omni_mask_Y = [True, True, True, False,  # 4
               True, True, True, True, False,  # 5
               True, True, True, False,  # 4
               True, True, True, False,  # 4
               True, True, True, False,  # 4
               True, True, True, True, False]  # 5
Omni_mask_N = [False, False, False, True,
               False, False, False, False, True,
               False, False, False, True,
               False, False, False, True,
               False, False, False, True,
               False, False, False, False, True]

model_mtad_mask_Y = [True]
model_mtad_mask_N = [False]
"""self.action_dims_mtad = [4, 4, 4, 4, 4, 5]"""  # 25
mtad_mask_Y = [True, True, True, False,  # 4
               True, True, True, False,  # 4
               True, True, True, False,  # 4
               True, True, True, False,  # 4
               True, True, True, False,  # 4
               True, True, True, True, False]  # 5
mtad_mask_N = [False, False, False, True,
               False, False, False, True,
               False, False, False, True,
               False, False, False, True,
               False, False, False, True,
               False, False, False, False, True]

model_InterFusion_mask_Y = [True]
model_InterFusion_mask_N = [False]
"""self.action_dims_InterFusion = [4, 4, 6, 4, 4, 5]"""  # 27
InterFusion_mask_Y = [True, True, True, False,  # 4
                      True, True, True, False,  # 4
                      True, True, True, True, True, False,  # 6
                      True, True, True, False,  # 4
                      True, True, True, False,  # 4
                      True, True, True, True, False]  # 5
InterFusion_mask_N = [False, False, False, True,
                      False, False, False, True,
                      False, False, False, False, False, True,
                      False, False, False, True,
                      False, False, False, True,
                      False, False, False, False, True]

model_SDFVAE_mask_Y = [True]
model_SDFVAE_mask_N = [False]
"""self.action_dims_SDFVAE = [4, 4, 5, 4, 4, 4]"""  # 25
SDFVAE_mask_Y = [True, True, True, False,  # 4
                 True, True, True, False,  # 4
                 True, True, True, True, False,  # 5
                 True, True, True, False,  # 4
                 True, True, True, False,  # 4
                 True, True, True, False]  # 4
SDFVAE_mask_N = [False, False, False, True,
                 False, False, False, True,
                 False, False, False, False, True,
                 False, False, False, True,
                 False, False, False, True,
                 False, False, False, True]
