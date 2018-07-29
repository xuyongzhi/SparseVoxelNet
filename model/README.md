# Key large items to be developed
- [ ] Add Voxception Res architecture
- [ ] Average multiple rotation views in test
- [x] Calculate the size of SVNET
- [ ] Check if the zero will lead to error in the net again!
- [ ] Block_bottom_center is incorrect in the piano.

# Key small items to be developed
- [ ] add learning rate warm up
- [ ] check batch size and batch norm decay
- [ ] data augment: random crops, more complex rotation, flips
- [ ] set as channel first
- [ ] check weight decay
- [ ] check use bias or not
- [ ] check optimizer: adam, momentum, RMSProp
- [ ] add dropout
- [ ] Use kernel>1 in shortcut may somewhat impede the identity forward, try optimize later
- [ ] Try low resolution firstly. Andrew has achieved 0.95 with 32*32*32 resolution.
- [ ] Use bottleneck in shortcut MC to reduce model size
- [ ] Make shortcut inception as well. And remove NoRes_InceptionReduction
- [ ] Remove USE_CHARLES
- [ ] Try remove self.feature_uncompress_block


# Training notes
- batch norm decay shoulded be decayed
