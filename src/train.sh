# base model
python launch.py -g 8 -t train_s1.py -c configs/s1.yml
# base model + GAN
python launch.py -g 8 -t train_s1_gan.py -c configs/s1_gan.yml --load-checkpoint logs/s1/checkpoint695000.ckpt
# super-resolution
python launch.py -g 8 -t train_s2.py -c configs/s2.yml --load-checkpoint logs/s1_gan/checkpoint750000.ckpt
