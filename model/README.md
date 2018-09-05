# Key large items to be developed
- [x] Add Voxception Res architecture
- [x] Average multiple rotation views in test
- [x] Calculate the size of SVNET
- [x] Check if the zero will lead to error in the net again!
- [x] Block_bottom_center is incorrect in the piano.
- [ ] In feature propagation, some points may get no flatting index
- [ ] In feature propagation, scale0 orginael feature is None
- [ ] Move the last global scope to include larger scope, instead of strictly more by stride.
- [ ] Select global blocks random is bot good, -> choose by num point
- [ ] Speed up sg for globla block, especially when only global scale
- [ ] Try replace grouped_center by mean

# Key small items to be developed
- [x] add learning rate warm up
- [x] check batch size and batch norm decay
- [ ] data augment: random crops, more complex rotation, flips
- [ ] set as channel first
- [ ] check weight decay
- [x] check use bias or not
- [ ] check optimizer: adam, momentum, RMSProp
- [x] add dropout
- [ ] Use kernel>1 in shortcut may somewhat impede the identity forward, try optimize later
- [ ] Try low resolution firstly. Andrew has achieved 0.95 with 32*32*32 resolution.
- [x] Use bottleneck in shortcut MC to reduce model size
- [ ] Make shortcut inception as well. And remove NoRes_InceptionReduction
- [x] Remove USE_CHARLES
- [x] Try remove self.feature_uncompress_block


# Training notes
- batch norm decay shoulded be decayed
