
**BUGS RECORD**
- [ ] train_0_nocolor_lxyz2.ply is not corret

**IMPORTANT factors to research**
- [ ] effect of padding
- [ ] effect of loss weight
- [ ] max_padding = aim_attrs['block_step'] in get_blockids_of_dif_stride_step
- [ ] Add calculate nearest aim_block index for missed base valid block index. in get_bidxmap() in block_data_prep_util.py
      Or just do no feature back propogation for missed base blocks. Currently, I use random aim_block index to test program. But this is wrong.
- [ ] In the obj generated from prh5, color is wrong

- [ ] In Qi's complementaion, 3 nearest balls are fused in back-propogated. I only use one. What is the right way.
- [ ] Add GlobalSubBaseBLOCK configuration check in get_data_larger_block
- [ ] get_blockids_of_dif_stride_step cost too much time
- [ ] Check the correctness of norm in get_block_data_of_new_stride_step_byid
- [ ] Check the performance and time cost fo probability sampling of:
	 sample_choice,_ = get_sample_choice( all_cur_blockids.shape[0],GlobalSubBaseBlock.nsubblock,GlobalSubBaseBlock.nsubblock,random_sampl_pro )
	 in get_data_larger_block
- [ ] Check the performance of randomly sampling of:
	sample_choice,reduced_num = get_sample_choice(datas.shape[0],sample_num) in
	get_block_data_of_new_stride_step_byid
- [ ] Candidate improvement:                                                 
           1) save all_cur_blockids and all_cur_block_size in h5f to save time.

**CONCLUSIONS**
- currently, center mask may be not right. While use center mask, the accuracy is up to 0.9, but achieved 0.97 after removing this.

**IMPORTANT OPERATIONS**
* (1) Generate ply to show block center of each cascade  in get_bidxmap:
- In class GlobalSubBaseBLOCK() in block_data_prep_util.py:
	IsCheck_gsbb['gen_ply_gsbb'] = True and ENABLECHECK
	ENABLECHECK  = True
- Note: break point is insrted after ply of each global block is generated. So multi process should be disabled in scannet_block_util.py etc.
- Be patient, it is very slow.
* (2) Generate bounding box of block and points inside the block while fixing bmap size:
- IsCheck_gsbb['IsCheckMissingAimb'] = True and ENABLECHECK
- On Line 1833 to set the global block num to show:
	if debug_meta['global_bidx'] < 5:
