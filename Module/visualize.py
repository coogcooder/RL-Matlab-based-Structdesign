import matplotlib.pyplot as plt



#######学習時プロット
def rwrd_plot(rewards,name,plt_intrvl):
    epi=[plt_intrvl*i for i in range(len(rewards))]
    plt.figure(figsize=(20,5))
    # for i in range(10):
    #     plt.hlines(5*i+15, 0,np.max(epi),color='black',alpha=0.3 )
    plt.plot(epi,rewards,"b") 
    plt.xlabel("Episode",size=12)   
    plt.ylabel("Σ r",size=12)
    plt.savefig('./output_images/cumulative_reward'+name+'.jpg')
    plt.close()

# def plot(sa_time,member,rwrd_epi,name):

#     plt.figure()
#     if name=='out':
#      col1=member[:,0]
#      col2=member[:,2]
#      bm=member[:,1]
#     elif name=='in':
#      col1=member[:,2]
#      col2=member[:,4]
#      bm=member[:,3]          
#     for i in range(4):    
#         p_col1=envmnt.col_list[str(col1[i])]['D']
#         p_col2=envmnt.col_list[str(col2[i])]['D']
#         p_bm=envmnt.beam_list[str(bm[i])]['H']

#         plt.plot([0,0],[1.5*i,1.5*(i+1)],lw=int(p_col1)/100,color="green")
#         plt.plot([7,7],[1.5*i,1.5*(i+1)],lw=int(p_col1)/100,color="green")
#         plt.text(-1.2,1.5*i+1,str(col1[i]),color=anly.oi_check(col1[i]))
#         plt.plot([3.5,3.5],[1.5*i,1.5*(i+1)],lw=int(p_col2)/100,color="red")
#         plt.text(2.2,1.5*i+1,str(col2[i]),color=anly.oi_check(col2[i]))
#         plt.plot([0,3.5],[1.5*(i+1),1.5*(i+1)],lw=int(p_bm)/100,color="blue")
#         plt.plot([3.5,7],[1.5*(i+1),1.5*(i+1)],lw=int(p_bm)/100,color="blue")    
#         plt.text(7.4,1.5*(i+1),str(bm[i]),color=anly.oi_check(bm[i]))  
#     plt.title('Total Reward='+str(rwrd_epi)+' V='+str(anly.V(member)))          
#     plt.xlim(-1.5,8.5)
#     plt.ylim(-1,9)
#     plt.savefig('./output_images/frame_image'+str(sa_time)+name+'.jpg')