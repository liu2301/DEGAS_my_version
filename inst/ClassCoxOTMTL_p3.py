predict_sc=add_layer(layerF,hidden_feats,Lsc,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
predict_pat=add_layer(layerF,hidden_feats,Lpat,activation_function=tf.sigmoid,dropout_function=False,lambda1=lambda1)
layerae_sc2pat=add_layer(es,Lsc,hidden_feats,activation_function=tf.sigmoid,dropout_function=False,lambda1=lambda1)
predictae_sc2pat = add_layer(layerae_sc2pat,hidden_feats,Lpat,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
#***********************************************************************
# Compute the squared L2 distance between two matrices (Updated 2022/06/29)
cost_t = tf.reshape(tf.math.reduce_sum(tf.math.square(tf.slice(layerF, [0, 0], [lsc, hidden_feats])), 1), [-1, 1])
cost_t += tf.reshape(tf.math.reduce_sum(tf.math.square(tf.slice(layerF, [lsc, 0], [lpat, hidden_feats])), 1), [1, -1])
cost_t -= 2.0 * tf.matmul(tf.slice(layerF, [0, 0], [lsc, hidden_feats]), tf.transpose(tf.slice(layerF, [lsc, 0], [lpat, hidden_feats])))
#***********************************************************************
# loss functions
lossLabel1 = tf.reduce_mean(tf.reduce_sum(tf.square(ys_sc-tf.slice(predict_sc,[0,0],[lsc,Lsc])),reduction_indices=[1]))				# FIX HERE
lossLabel2 = -tf.reduce_mean((tf.squeeze(tf.slice(predict_pat,[lsc,0],[lpat,Lpat])) - tf.log(tf.reduce_sum(tf.exp(tf.squeeze(tf.slice(predict_pat,[lsc,0],[lpat,Lpat]))) * r_pat, 1))) * c_pat)		# FIX HERE
# replace MMD with OT (updated 2022/07/06)
# lossMMD = mmd_loss(tf.slice(layerF,[0,0],[lsc,hidden_feats]),tf.slice(layerF,[lsc,0],[lpat,hidden_feats]))
lossOT = OT_loss(tf.slice(layerF, [0, 0], [lsc, hidden_feats]), tf.slice(layerF, [lsc, 0], [lpat, hidden_feats]), trans_P)
lossConstSCtoPT = tf.reduce_mean(tf.reduce_sum(tf.square(tf.slice(predict_pat,[0,0],[lsc,Lpat])-(1.0/2.0)),reduction_indices=[1]))  #TESTING
lossConstPTtoSC = tf.reduce_mean(tf.reduce_sum(tf.square(tf.slice(predict_sc,[lsc,0],[lpat,Lsc])-(1.0/Lsc)),reduction_indices=[1]))  #TESTING
#lossConstPTtoPT = tf.reduce_mean(tf.reduce_sum(tf.square(tf.squeeze(tf.slice(predict_pat,[lsc,0],[lpat,Lpat]))-(1.0/2.0)))) 
#*************Use below cost functions**********************
# use OT to replace MMD loss (2022/07/06)
# loss = 2*lossLabel1 + lambda2*lossLabel2 +lambda3*lossMMD + lossConstSCtoPT + lossConstPTtoSC
loss = 2*lossLabel1 + lambda2*lossLabel2 +lambda3*lossOT + lossConstSCtoPT + lossConstPTtoSC
lossae_sc2pat = tf.reduce_mean(tf.reduce_sum(tf.square(ps-predictae_sc2pat),reduction_indices=[1]))
train_step1 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-3).minimize(loss)
train_step2 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-3).minimize(lossae_sc2pat)
train_sc = resample(50,Ysc,idx_sc)			# CHANGED
train_pat = idx_pat							# CHANGED
np.random.shuffle(train_pat)				# CHANGED 20201217
np.random.shuffle(train_sc)					# CHANGED 20201217
train_sc2 = train_sc[0:scbatch_sz]			# CHANGED
train_pat2 = train_pat[0:patbatch_sz]		# CHANGED
resampleGammaXYsc = resample_mixGamma(np.squeeze(Xsc[train_sc2,:]),np.squeeze(Ysc[train_sc2,:]),list(range(scbatch_sz)),scbatch_sz,Lsc)       # CHANGED 20201217
tensor_train = {xs: np.concatenate([resampleGammaXYsc[0],np.squeeze(Xpat[train_pat2,:])]), ys_sc: resampleGammaXYsc[1], r_pat: Rmatrix(survtime[train_pat2]), c_pat: np.squeeze(censor[train_pat2]), lsc: resampleGammaXYsc[1].shape[0], lpat: len(train_pat2), kprob: do_prc}
init=tf.global_variables_initializer()
#***********************************************************************
# training model
#run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
sess=tf.Session()
sess.run(init)
for i in range(train_steps + 1):
    #sess.run(train_step1, feed_dict=tensor_train,options=run_options)
    # sess.run(train_step1, feed_dict=tensor_train)
    # BELOW IS UNTESTED
    # sess.run(train_step2, feed_dict={es: sess.run(predict_sc,feed_dict={xs:np.squeeze(Xpat[train_pat2,:]), kprob: do_prc}), ps: sess.run(predict_pat,feed_dict={xs:np.squeeze(Xpat[train_pat2,:]), kprob: do_prc}), kprob: do_prc})
    if(i % 50 == 0):
        # print(str(sess.run(loss, feed_dict=tensor_train))+' '+str(sess.run(lossLabel1,feed_dict=tensor_train))+'	'+str(sess.run(lossLabel2,feed_dict=tensor_train))+'	'+str(sess.run(lossMMD,feed_dict=tensor_train))+' '+str(sess.run(lossae_sc2pat, feed_dict={es: sess.run(predict_sc,feed_dict={xs:np.squeeze(Xpat[train_pat2,:]), kprob: do_prc}), ps: sess.run(predict_pat,feed_dict={xs:np.squeeze(Xpat[train_pat2,:]), kprob: do_prc}), kprob: do_prc})))
        train_sc = resample(50,Ysc,idx_sc)			# CHANGED
        train_pat = idx_pat							# CHANGED
        np.random.shuffle(train_sc)					# CHANGED
        np.random.shuffle(train_pat)				# CHANGED
        train_sc2 = train_sc[0:scbatch_sz]			# CHANGED
        train_pat2 = train_pat[0:patbatch_sz]		# CHANGED
        resampleGammaXYsc = resample_mixGamma(np.squeeze(Xsc[train_sc2,:]),np.squeeze(Ysc[train_sc2,:]),list(range(scbatch_sz)),scbatch_sz,Lsc)       # CHANGED 20201217
        tensor_train = {xs: np.concatenate([resampleGammaXYsc[0],np.squeeze(Xpat[train_pat2,:])]), ys_sc: resampleGammaXYsc[1], r_pat: Rmatrix(survtime[train_pat2]), c_pat: np.squeeze(censor[train_pat2]), lsc: resampleGammaXYsc[1].shape[0], lpat: len(train_pat2), kprob: do_prc}

        # calculate the transportation function (updated 2022/07/06)
        M = sess.run(cost_t, feed_dict=tensor_train)
        transportation_P = ot.emd(a=[], b=[], M=M)
        tensor_train[trans_P] = transportation_P
        # calculate the regularized transportation function (updated 2022/07/14)
        # featureF, N_lsc = sess.run([layerF, lsc], feed_dict=tensor_train)
        # ot_lpl1 = ot.da.SinkhornLpl1Transport(reg_e=1e-1, reg_cl=2e0)
        # ot_lpl1.fit(Xs = featureF[:int(N_lsc), :], ys = tensor_train[ys_sc].argmax(1), Xt = featureF[int(N_lsc):, :])
        # transportation_P = ot_lpl1.coupling_
        # tensor_train[trans_P] = transportation_P
        
        
    sess.run(train_step1, feed_dict=tensor_train)
    sess.run(train_step2, feed_dict={es: sess.run(predict_sc,feed_dict={xs:np.squeeze(Xpat[train_pat2,:]), kprob: do_prc}), ps: sess.run(predict_pat,feed_dict={xs:np.squeeze(Xpat[train_pat2,:]), kprob: do_prc}), kprob: do_prc})
