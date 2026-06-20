python -m flux.demo --device cuda:6
python -m flux.demo --device cuda:6 --speed_up 6x
python -m flux.demo_2stg --device cuda:6
python -m flux.demo_2stg --device cuda:6 --mask_type "base"

python -m wan.demo --device cuda:7
python -m wan.demo --device cuda:7 --mask_type "base"

auto download:
sam2.1_hiera_tiny.pt
yoloe-26s-seg-pf.pt