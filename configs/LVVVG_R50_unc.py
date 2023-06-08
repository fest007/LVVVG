dataset='unc'

# output_dir='work_dirs/VLTVG_R50_unc/'
# output_dir='work_dirs_原版无改query1/VLTVG_R50_unc/'
# output_dir='work_dirs_原版改hs的query5/VLTVG_R50_unc/'
# output_dir='work_dirs_原版改outputs_coord的query5/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上tgt自注意query1/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上tgt自注意query5改hs/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上视觉注意query1现文本后视觉/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上视觉注意query1现视觉后文本/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证query1改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证query1在语言注意之前加的VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证query5hs改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证和视觉注意先视觉query1改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证和tgt自注意query1改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证和tgt自注意query5改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证加上残差query1改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证加上残差query5改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证加上残差之后加text_info的残差query1改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证加上残差之后加text_info的残差query5改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证加上残差之后在视觉后都加上的残差query1改语言注意的/VLTVG_R50_unc/'
# output_dir='work_dirs_原版加上语言验证加上残差之后在视觉后都加上的残差query5改语言注意的/VLTVG_R50_unc/'
output_dir='work_best/'



# batch_size=16
batch_size=8
epochs=90
lr_drop=60
freeze_epochs=10
freeze_modules=['backbone', 'input_proj', 'trans_encoder', 'bert']
load_weights_path='pretrained_checkpoints/detr-r50-unc.pth'

model_config = dict(
    decoder=dict(
        type='DecoderWithExtraEncoder',
        # num_queries=1,
        num_queries=5,

        query_dim=256,
        norm_dim = 256,
        return_intermediate=True,
        num_layers=6,
        layer=dict(
            type='MultiStageDecoderLayer',
            d_model=256,
            dim_feedforward=2048,
            dropout=0.,
            word_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1,
            ),
            img_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1,
            ),
            img_feat_chunk_num = 2,
        ),
        num_extra_layers=1,
        extra_layer=dict(
            type='DiscriminativeFeatEncLayer',
            d_model=256,
            img_query_with_pos=False,
            img2text_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            img2textcond_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            img2img_attn_args=dict(
                type='MHAttentionRPE',
                d_model=256, h=8, dropout=0.1,
                pos_x_range=[-20, 20],
                pos_y_range=[-20, 20],
                pos_index_offset=20
            ),
            vl_verify=dict(
                text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                scale=1.0,
                sigma=0.5,
                pow=2.0,
            ),
        )
    )
)