# python3 benchmark_new.py augment.active=false model.transition="marginal" augment.start_aug_epoch=50 augment.aug_interval=25 augment.NumSubgraphs=100 augment.diffusion_steps=10 augment.maxNode=35
# python3 benchmark_new.py model.transition="marginal" augment.start_aug_epoch=50 augment.aug_interval=25 augment.NumSubgraphs=100 augment.diffusion_steps=10 augment.maxNode=35

# 1-hop
python3 benchmark_new.py model.transition="marginal" augment.start_aug_epoch=30 augment.aug_interval=25 augment.NumSubgraphs=100 augment.diffusion_steps=10 augment.maxNode=35

# 2-hop: 250
python3 benchmark_new.py model.transition="marginal" augment.start_aug_epoch=30 augment.aug_interval=25 augment.NumSubgraphs=100 augment.diffusion_steps=10 augment.maxNode=250 augment.lggm_variant="adj_2hop"
