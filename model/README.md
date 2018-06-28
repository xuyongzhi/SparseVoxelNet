# Key large items to be developed
- [ ] Add Voxception Res architecture
- [ ] Average multiple rotation views in test
- [ ] Calculate the size of SVNET

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


# Training notes
- batch norm decay shoulded be decayed
