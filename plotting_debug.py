from experiments_qc import Experiments

exp = Experiments()
exp.load('results/results_0707.csv')
exp.results_df.loc[(exp.results_df['precision'] == '32_32_32'), 'method'] = 'float'

# a little oopsie in reading comprehension
# repqvit and iasvit attention bitwidths should be the same as the other activation (layernorm)
# it's actually a bit of a pain to re-generate the csv bc all the logit file names are tied to it
# so we'll just do a little bit of pd processing xD
import pandas as pd

# get df at repqvit and iasvit method
# for all precisions not 32_32_32, change prec to '_'.join(prec.split('_')[:-1]+prec.split('_')[-2])
mask = (((exp.results_df['method'] == 'repqvit') | (exp.results_df['method'] == 'iasvit')) 
        & (exp.results_df['precision'] != '32_32_32'))
valid_prec_df = exp.results_df[mask]

exp.results_df['precision_2'] = exp.results_df['precision'].astype("string")
prec_str = valid_prec_df['precision'].str
w_a_bit = prec_str.rsplit('_', expand=True, n=1)[0].str
att_bit = prec_str.split('_', expand=True)[1]
exp.results_df.loc[mask, 'precision_2'] = w_a_bit.cat(att_bit, join='left', sep='_')

exp.results_df['precision'] = exp.results_df['precision_2']
exp.results_df = exp.results_df.drop(columns='precision_2')

ALL_MODELS = [
                'vit_small_patch16_224', 
                'swin_small_patch4_window7_224',
                'swin_tiny_patch4_window7_224',
                'deit_small_patch16_224', 
                'deit_tiny_patch16_224', 
            ]

ALL_PRECISIONS_REPQ = ['32_32_32', '8_8_8', '6_6_6', '4_8_8', '8_4_4', '4_4_4']

# ************************************************************************
print(f"Plotting acc vs confidence thr curves")
settings = {
    'model': ALL_MODELS,
    'dataset': ['ImageNet-A'],
    'precision': ALL_PRECISIONS_REPQ, #, '8_4_32'], #, '4_4_32'],
    'method': ['float', 'repqvit'] #, 'iasvit'] #, 'fqvit']
}

# exp.plot_acc_vs_confidence_thr(
#     settings, 
#     save_file='acc_vs_conf_MODEL_ALL_METHOD_repqvit', 
#     size=(16,3),
#     conf_lim=[0.0,0.90],
#     acc_lim=[0.0,0.60]
# )

# settings['method'] = ['float', 'fqvit']
# settings['precision'] = ['32_32_32', '8_8_32', '4_8_32', '8_8_8', '4_8_8']
# exp.plot_acc_vs_confidence_thr(
#     settings, 
#     save_file='acc_vs_conf_MODEL_ALL_METHOD_fqvit', 
#     size=(16,3),
#     conf_lim=[0.0,0.90],
#     acc_lim=[0.0,0.60]
# )

# settings['dataset'] = ['ImageNet-A', 'ImageNet1k']
# exp.plot_acc_vs_confidence_thr(
#     settings, 
#     save_file='acc_vs_conf_MODEL_ALL_METHOD_repqvit_combined_dataset', 
#     size=(16,3),
#     conf_lim=[0.0,0.90],
#     acc_lim=[0.30,0.90]
# )

# settings['method'] = ['float', 'iasvit']
# exp.plot_acc_vs_confidence_thr(
#     settings, 
#     save_file='acc_vs_conf_MODEL_small_METHOD_iasvit', 
#     size=(10,3),
#     conf_lim=[0.0,0.90],
#     acc_lim=[0.0,0.60]
# )



# ************************************************************************
# print(f"Plotting RRA curves")
# settings = {
#     'model': ALL_MODELS,
#     'precision': ALL_PRECISIONS_REPQ,
#     'method': ['float', 'repqvit'] # 'fqvit', 'repqvit', 'iasvit'] #, 'fqvit']
# }

# exp.plot_rra_curves(settings,
#                     dataset='ImageNet-A',
#                     save_file='ImageNet-A_rra_MODEL_all_METHOD_repqvit',
#                     size=(12,2.2),
#                     acc_lim=[0,60]
#                 )

# exp.plot_rra_curves(settings,
#                     dataset='ImageNet1k',
#                     save_file='debug_rra_ImageNet1k',
#                     size=(10,10))


# # ************************************************************************
# print(f"Plotting predictive entropy histogram curves")

# settings = {
#     'model': ['deit_small_patch16_224', 'vit_small_patch16_224', 'swin_small_patch4_window7_224'],
#     'precision': ['32_32_32', '8_8_32', '4_8_32', '8_8_8', '4_8_8'], #'6_6_32'], #, '4_4_32'],
#     'method': ['float', 'fqvit']
# }

# exp.plot_entropy_histograms(settings,
#                     col_setting='model',
#                     save_file=f'entropy_kde_MODELS_small_METHOD_fqvit_commonnorm_0_ALL_DATA',
#                     size=(9,5),
#                     y_lim_known=[0,1.6],
#                     y_lim_unknown=[0,1.6]
#                 )

# settings = {
#     'model': ['deit_small_patch16_224', 'vit_small_patch16_224', 'swin_tiny_patch4_window7_224'], # 'swin_small_patch4_window7_224'],
#     'precision': ALL_PRECISIONS_REPQ,
#     'method': ['float', 'repqvit']
# }

# exp.plot_entropy_histograms(settings,
#                     col_setting='model',
#                     save_file=f'entropy_kde_MODELS_small_complexity_METHOD_repqvit_commonnorm_0_ALL_DATA',
#                     size=(9,5),
#                     y_lim_known=[0,1.6],
#                     y_lim_unknown=[0,1.6]
#                 )

# settings['method'] = ['float', 'iasvit']
# exp.plot_entropy_histograms(settings,
#                     col_setting='model',
#                     save_file=f'entropy_kde_MODELS_small_METHOD_iasvit_commonnorm_0',
#                     size=(9,5),
#                     y_lim_known=[0,0.6],
#                     y_lim_unknown=[0,0.6]
#                )

# settings['model'] = ALL_MODELS

# exp.plot_entropy_histograms(settings,
#                     col_setting='model',
#                     save_file=f'entropy_kde_MODELS_small_METHOD_fqvit_commonnorm_0',
#                     size=(9,5),
#                     y_lim_known=[0,0.6],
#                     y_lim_unknown=[0,0.6]
#                 )


# ************************************************************************
# print(f"Plotting AUPR comparisons")

# settings = {
#     'model': ALL_MODELS,
#     'precision': ALL_PRECISIONS_REPQ,
#     'method': ['float', 'repqvit'] #, 'iasvit'] 
# }

# exp.plot_AUPR_comparisons(settings,
#                     save_file=f"aupr_compare_MODEL_all_PRECISION_all",
#                     size=(3.2,4) #(2.2,2.9)
#                 )

# settings['precision'] = ['32_32_32', '8_8_32', '8_8_8', '4_8_32', '4_8_8']
# settings['method'] = ['float', 'fqvit']
# exp.plot_AUPR_comparisons(settings,
#                     save_file=f"aupr_compare_MODEL_all_PRECISION_qatt_vs_fpatt",
#                     size=(3.2,4)
#                 )

# ************************************************************************
# print(f"Plotting reliability curves")
# settings = {
#     'model': [
#                 'deit_small_patch16_224', 
#                 'vit_small_patch16_224', 
#                 'swin_small_patch4_window7_224'
#             ],
#     'precision': ['32_32_32', '8_8_32', '6_6_32', '4_8_32', '8_8_8', '4_8_8'],
#     'method': ['float', 'fqvit'],
#     'logits_RC': 0
# }

# print("...for small variants quantized on FQ-ViT")
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_fqvit_reliability_diagrams_unscaled',
#                            size=(10, 9))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_fqvit_reliability_diagrams_RC',
#                            size=(10, 9))

# print("...for small variants quantized on RepQ-ViT")
# settings['method'] = ['float', 'repqvit']
# settings['precision'] = ['32_32_32', '8_8_8', '6_6_6', '4_4_4']
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_repqvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_repqvit_reliability_diagrams_scaled',
#                            size=(7.2, 6))

# print("...on ImageNet-A")
# settings['dataset'] = ['ImageNet-A']
# settings['logits_RC'] = 1
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_imgnet-a_repqvit_reliability_diagrams_scaled',
#                            size=(7.2, 6)
#                         )

# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_imgnet-a_repqvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6)
#                         )


# print("...for small variants quantized on IaS-ViT")
# settings['method'] = ['float', 'iasvit']
# settings['precision'] = ['32_32_32', '8_8_8', '6_6_6', '4_4_4']
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_iasvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='small_arch_iasvit_reliability_diagrams_RC',
#                            size=(7.2, 6))

# print("...for deit variants quantized on RepQ-ViT")
# settings['model'] = [
#                 'deit_tiny_patch16_224', 
#                 'deit_small_patch16_224'
#             ]
# settings['method'] = ['float', 'repqvit']
# settings['precision'] = ['32_32_32', '8_8_8', '6_6_6', '4_4_4']
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='deit_repqvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='deit_repqvit_reliability_diagrams_RC',
#                            size=(7.2, 6))

# settings['dataset'] = ['ImageNet-A']
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='deit_imgnet-a_repqvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='deit_imgnet-a_repqvit_reliability_diagrams_RC',
#                            size=(7.2, 6))

# print("...for deit variants quantized on IaS-ViT")
# settings['method'] = ['float', 'iasvit']
# settings['precision'] = ALL_PRECISIONS_REPQ
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='deit_imgnet-a_iasvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='deit_imgnet-a_iasvit_reliability_diagrams_RC',
#                            size=(7.2, 6))

# print("...for swin variants quantized on RepQ-ViT")
# settings['model'] = [
#                 'swin_tiny_patch4_window7_224',
#                 'swin_small_patch4_window7_224'
#             ]
# settings['method'] = ['float', 'repqvit']
# settings['precision'] = ['32_32_32', '8_8_32', '6_6_32', '4_4_32']
# settings['dataset'] = ['ImageNet-A']
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='swin_imgnet-a_repqvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='swin_imgnet-a_repqvit_reliability_diagrams_RC',
#                            size=(7.2, 6))

# print("...for swin variants quantized on IaS-ViT")
# settings['method'] = ['float', 'iasvit']
# settings['precision'] = ['32_32_32', '8_8_32', '6_6_32', '4_4_32']
# settings['logits_RC'] = 0
# exp.plot_reliability_curve(settings, 
#                            save_file='swin_imgnet-a_iasvit_reliability_diagrams_unscaled',
#                            size=(7.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='swin_imgnet-a_iasvit_reliability_diagrams_RC',
#                            size=(7.2, 6))

# print("...for tiny variants quantized on FQ-ViT")
# settings = {
#     'model': [
#                 'swin_tiny_patch4_window7_224',
#                 'deit_tiny_patch16_224' 
#             ],
#     'precision': ['32_32_32', '8_8_32', '6_6_32', '4_4_32', '8_8_8', '4_8_8'],
#     'method': ['float', 'fqvit'],
#     'logits_RC': 0
# }
# exp.plot_reliability_curve(settings, 
#                            save_file='tiny_arch_fqvit_reliability_diagrams_unscaled',
#                            size=(10.2, 6))
# settings['logits_RC'] = 1 
# exp.plot_reliability_curve(settings, 
#                            save_file='tiny_all_arch_fqvit_reliability_diagrams_RC',
#                            size=(10.2, 6))

# ************************************************************************
print(f"Plotting pareto")
settings = {
    'model': ALL_MODELS,
    'precision': ['32_32_32', '8_8_32', '4_8_32', '8_8_8', '4_8_8'],
    'dataset': ['ImageNet1k'],
    'method': ['float', 'fqvit'] #, 'repq-vit'],
}
exp.plot_pareto_acc_vs_uncertainty(settings, save_file='pareto_acc_vs_uncertainty_METHOD_fqvit_test', 
                                    x_lim=[0.15,0.60], ece_lim=[0,0.40], rbs_lim=[0.45,0.8])
# settings = {
#     'model': ALL_MODELS,
#     'precision': ['4_4_32', '4_4_8'],
#     'method': ['fq-vit'] #, 'repq-vit'],
# }
# exp.plot_pareto_acc_vs_uncertainty(settings, save_file='pareto_fqvit_lessbit',
#                                     x_lim=[0.98,1.02], ece_lim=[-0.1,0.1], rbs_lim=[0.9,1.1])

# settings = {
#     'model': ALL_MODELS,
#     'precision': ALL_PRECISIONS_REPQ,
#     'dataset': ['ImageNet1k'],
#     'method': ['float', 'repqvit'],
# }

# exp.plot_pareto_acc_vs_uncertainty(settings, save_file='pareto_acc_vs_uncertainty_METHOD_repqvit', 
#                                     x_lim=[0.15,0.45], ece_lim=[0,0.25], rbs_lim=[0.45,0.8])

# settings['dataset'] = ['ImageNet-A']
# exp.plot_pareto_acc_vs_uncertainty(settings, save_file='pareto_acc_vs_uncertainty_DATASET_imgneta_METHOD_repqvit', 
#                                     x_lim=[0.6,1.0], ece_lim=[0.15,0.5], rbs_lim=[0.9,1.2])


# settings = {
#     'model': ALL_MODELS,
#     'precision': ALL_PRECISIONS_REPQ,
#     'method': ['float', 'iasvit'],
# }
# exp.plot_pareto_acc_vs_uncertainty(settings, save_file='pareto_acc_vs_uncertainty_METHOD_iasvit', 
#                                     x_lim=[0.15,0.45], ece_lim=[0,0.25], rbs_lim=[0.45,0.8])


# ************************************************************************
# settings = {
#     'model': [
#                 'swin_tiny_patch4_window7_224',
#                 'swin_small_patch4_window7_224' 
#                 ],
#     'precision': ['32_32_32', '8_8_32', '4_8_32', '8_4_32', '6_6_32', '4_4_32'], # '4_4_8'],
#     'method': ['float', 'iasvit', 'repqvit'],
# }
# exp.plot_error_and_uncertainty_vs_precision(settings, save_file="top1_nll_by_precision_short",
#                                             top1_lim=[0.1, 0.45], nll_lim=[0.6,1.8])
# exp.plot_error_and_uncertainty_vs_precision(settings, save_file="top1_nll_by_precision_full")

# ************************************************************************
# exp.plot_RC_deltas_by_precision(settings, save_file="debug_RC_deltas_by_precision_swins", size=(12,3), rc_lim=[-0.05,0.20])


# ************************************************************************
# settings = {
#     'model': ALL_MODELS,
#     'precision': ['32_32_32', '8_8_32', '6_6_32', '4_8_32', '4_4_32'], # '4_4_8'],
#     'method': ['float', 'repqvit']
# }

# Okay this is like crazy messy...
# may need to just keep it at one method at a time. Might not even need both metrics tbh
# The trends track 
# exp.plot_RC_deltas_by_error_deltas(settings, save_file="debug_RC_deltas_by_error_deltas_repqvit_only_CE_delta_proportional", top1_lim=[-0.025, 0.18])
# exp.plot_RC_deltas_by_error_deltas(settings, save_file="debug_RC_deltas_by_error_deltas_small_variants_closer", top1_lim=[-0.025, 0.05])

# settings['method'] = ['float', 'iasvit']
# exp.plot_RC_deltas_by_error_deltas(settings, save_file="debug_RC_deltas_by_error_deltas_iasvit_only", top1_lim=[-0.025, 0.3])
