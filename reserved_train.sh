# batch 32 / Adam / lr 1e-4 / scheduler gamma 0.1
make train-conv

sed -i 's/gamma .1/gamma .2/g' conf/conv/training/training.yml
sed -i 's/1.xnor_conv(deterministic)/2.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .2/gamma .4/g' conf/conv/training/training.yml
sed -i 's/2.xnor_conv(deterministic)/3.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .4/gamma .6/g' conf/conv/training/training.yml
sed -i 's/3.xnor_conv(deterministic)/4.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .6/gamma .8/g' conf/conv/training/training.yml
sed -i 's/4.xnor_conv(deterministic)/5.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/g' train.py
make train-conv

sed -i 's/gamma .8/gamma .9/g' conf/conv/training/training.yml
sed -i 's/5.xnor_conv(deterministic)/6.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/g' train.py
make train-conv

sed -i 's/gamma .9/gamma .1/g' conf/conv/training/training.yml
sed -i 's/6.xnor_conv(deterministic)/7.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/g' train.py

sed -i 's/lr 1e-4/lr 1e-3/g' conf/conv/training/training.yml
sed -i 's/lr : [1e-4(ing), 1e-3, 1e-2, 1e-1]/lr : [1e-4, 1e-3(ing), 1e-2, 1e-1]/g' train.py

# batch 32 / Adam / lr 1e-3 / scheduler gamma 0.1
make train-conv

sed -i 's/gamma .1/gamma .2/g' conf/conv/training/training.yml
sed -i 's/7.xnor_conv(deterministic)/8.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .2/gamma .4/g' conf/conv/training/training.yml
sed -i 's/8.xnor_conv(deterministic)/9.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .4/gamma .6/g' conf/conv/training/training.yml
sed -i 's/9.xnor_conv(deterministic)/10.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .6/gamma .8/g' conf/conv/training/training.yml
sed -i 's/10.xnor_conv(deterministic)/11.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/g' train.py
make train-conv

sed -i 's/gamma .8/gamma .9/g' conf/conv/training/training.yml
sed -i 's/11.xnor_conv(deterministic)/12.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/g' train.py
make train-conv

sed -i 's/gamma .9/gamma .1/g' conf/conv/training/training.yml
sed -i 's/12.xnor_conv(deterministic)/13.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/g' train.py

sed -i 's/lr 1e-3/lr 1e-2/g' conf/conv/training/training.yml
sed -i 's/lr : [1e-4, 1e-3(ing), 1e-2, 1e-1]/lr : [1e-4, 1e-3, 1e-2(ing), 1e-1]/g' train.py

# batch 32 / Adam / lr 1e-2 / scheduler gamma 0.1
make train-conv

sed -i 's/gamma .1/gamma .2/g' conf/conv/training/training.yml
sed -i 's/13.xnor_conv(deterministic)/14.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .2/gamma .4/g' conf/conv/training/training.yml
sed -i 's/14.xnor_conv(deterministic)/15.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .4/gamma .6/g' conf/conv/training/training.yml
sed -i 's/15.xnor_conv(deterministic)/16.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .6/gamma .8/g' conf/conv/training/training.yml
sed -i 's/16.xnor_conv(deterministic)/17.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/g' train.py
make train-conv

sed -i 's/gamma .8/gamma .9/g' conf/conv/training/training.yml
sed -i 's/17.xnor_conv(deterministic)/18.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/g' train.py
make train-conv

sed -i 's/gamma .9/gamma .1/g' conf/conv/training/training.yml
sed -i 's/18.xnor_conv(deterministic)/19.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/g' train.py

sed -i 's/lr 1e-2/lr 1e-1/g' conf/conv/training/training.yml
sed -i 's/lr : [1e-4, 1e-3, 1e-2(ing), 1e-1]/lr : [1e-4, 1e-3, 1e-2, 1e-1(ing)]/g' train.py

# batch 32 / Adam / lr 1e-1 / scheduler gamma 0.1
make train-conv

sed -i 's/gamma .1/gamma .2/g' conf/conv/training/training.yml
sed -i 's/19.xnor_conv(deterministic)/20.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .2/gamma .4/g' conf/conv/training/training.yml
sed -i 's/20.xnor_conv(deterministic)/21.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2(ing), 0.4, 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .4/gamma .6/g' conf/conv/training/training.yml
sed -i 's/21.xnor_conv(deterministic)/22.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4(ing), 0.6, 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/g' train.py
make train-conv

sed -i 's/gamma .6/gamma .8/g' conf/conv/training/training.yml
sed -i 's/22.xnor_conv(deterministic)/23.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6(ing), 0.8, 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/g' train.py
make train-conv

sed -i 's/gamma .8/gamma .9/g' conf/conv/training/training.yml
sed -i 's/23.xnor_conv(deterministic)/24.xnor_conv(deterministic)/g' conf/conv/training/training.yml
sed -i 's/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8(ing), 0.9]/scheduler gamma: [0.1, 0.2, 0.4, 0.6, 0.8, 0.9(ing)]/g' train.py
make train-conv
