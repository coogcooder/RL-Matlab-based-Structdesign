from typing import Hashable
from matplotlib.pyplot import subplot2grid
import matlab.engine
eng=matlab.engine.start_matlab()  
import math 
import numpy as np 
import csv 
import copy
# import make_map
import itertools
import math

##鋼材リスト
list_H=[400,450,500, 550,600,650,700,750,800,850,900,950,1000]##13
list_B=[200,250,300,350,400]##5
list_tw=[9,12,14,16,19,22]##6
list_tf=[12,16,19,22,25,28,32,36,40]##9
list_D=[350,400,450,500,550,600,650,700,750,800,850,900,950,1000]##14
list_t=[12,16,19,22,25,28,32,36,38,40]##10


##鋼材リスト
def dctnry(filename,cb_switch):##cb_switch 'column 0' 'beam 1'
    reader = csv.reader(open(filename, 'r'))
    dic={}
    label=[]
    ddic=[]
    list_=[]
    length=0
    for id,i in enumerate(reader):
      if id ==0:
        label=i
        length=len(label)       
     
      elif id>=1:
       ddic=i ##数値データ
       if cb_switch==0:
         dic[str(ddic[0])]={label[1]:ddic[1]}##Dで作成
         dic[str(ddic[0])][label[2]]=ddic[2]
         for k in range(2,length):
            dic[str(ddic[0])][label[k]]=ddic[k] 

         list_.append([str(ddic[0]),str(ddic[3]),str(ddic[4])])        
       else:
          dic[str(ddic[0])]={label[1]:ddic[1]}##Hで作成
          for k in range(2,length):
            dic[str(ddic[0])][label[k]]=ddic[k]
          list_.append([str(ddic[0]),str(ddic[5]),str(ddic[6])])   
    return dic,  list_   

def init_dctnry(filename):
    reader = csv.reader(open(filename, 'r'))
    dic={}
    label=[]
    ddic=[]
    
    for id,i in enumerate(reader):
      
      if id ==0:
        label=i    
     
      elif id>=1:
       ddic=i ##数値データ
       if int(ddic[8])<=3:
         dic[str(-1*id)]={str(0):ddic[0]}
         for i in range(7):
           dic[str(-1*id)][str(i+1)]=ddic[i+1]
       else:
         dic[str(-1*id-8)]={str(0):ddic[0]}
         for i in range(7):
           dic[str(-1*id-8)][str(i+1)]=ddic[i+1]     
    return dic       


##初期部材ロード
# def initial_member(name,lx,zSpan,xSpan,mrgn):
#     reader = csv.reader(open('./csv/離散化完了.csv', 'r'))
#     z=zSpan+mrgn*2
#     lx=lx+mrgn*2
#     member=np.full(z*lx*lx,'aaa',dtype=object) 
#     for i in range(len(member.reshape((-1)))):
#       member[i]=str(0)
#     member=member.reshape((z,lx,lx))
    
#     for read in reader:
#       if name in read[0]:
        
#         beam_span=int((xSpan+1)/2)
#         beam_num=zSpan*beam_span
#         col_span=int((xSpan+2)/2)  
#         col_num=zSpan*col_span 
        
#         inframe_num=col_span - 1
#         if 'o' in read[0]:
#           for i in range(beam_num):
#              member[i//beam_span+mrgn,0+mrgn,2*(i%beam_span)+1+mrgn]='0x'+read[i+1]+read[i+1+beam_num]+read[i+1+2*beam_num]+read[i+1+3*beam_num]
#           for i in range(col_num):
#              member[i%zSpan+mrgn,0+mrgn,2*(i//zSpan)+mrgn]='0x'+read[i+1+4*beam_num]+read[i+1+col_num+4*beam_num]
#         elif 'i' in read[0]:
#           for j in range(inframe_num):
#              for i in range(beam_num):
#                  member[i//beam_span+mrgn,2*j+2+mrgn,2*(i%beam_span)+1+mrgn]='0x'+read[i+1]+read[i+1+beam_num]+read[i+1+2*beam_num]+read[i+1+3*beam_num]
#              for i in range(col_num):
#                  member[i%zSpan+mrgn,2*j+2+mrgn,2*(i//zSpan)+mrgn]='0x'+read[i+1+4*beam_num]+read[i+1+col_num+4*beam_num]                   
#     member=mirror(member,0) 
#     return member   

 


##選択鋼材リスト
def variable1(member,col_list,beam_list,arraymat):

   H,B,tw,tf=[],[],[],[]
   D,td=[],[]
   size=np.shape(arraymat)
   sizez,sizey,sizex=size[0],int((size[1]+1)/2),int((size[2]+1)/2)###切り捨て
   ####Gx
   for i in range(sizez):
     for j in range(sizey):
       for k in range(sizex):
         if arraymat[i,j,k]==2:
           if j%2==0:     
              id=member[i,j,k]
              h_=beam_list[str(id[2:])]['H']
              b_=beam_list[str(id[2:])]['B']
              tw_=beam_list[str(id[2:])]['tw']
              tf_=beam_list[str(id[2:])]['tf']
              H.append(int(h_))
              B.append(int(b_))
              tw.append(int(tw_)) 
              tf.append(int(tf_))              
   ####Gy
   for i in range(sizez):
     for k in range(sizex):
       for j in range(sizey):
         if arraymat[i,j,k]==2:
           if k%2==0:    
              id=member[i,j,k]
              h_=beam_list[str(id[2:])]['H']
              b_=beam_list[str(id[2:])]['B']
              tw_=beam_list[str(id[2:])]['tw']
              tf_=beam_list[str(id[2:])]['tf']
              H.append(int(h_))
              B.append(int(b_))
              tw.append(int(tw_)) 
              tf.append(int(tf_))                              
   ####C
   for j in range(sizey):
     for k in range(sizex):
       for i in range(sizez):
         if arraymat[i,j,k]==1:
            id=member[i,j,k]
            D_=col_list[str(id[2:])]['D']
            td_=col_list[str(id[2:])]['td'] 
            D.append(int(D_))
            td.append(int(td_))      
   
   var=np.concatenate([np.array(H),np.array(B),np.array(tw),np.array(tf)
                       ,np.array(D),np.array(td)])
   
   return var    






##選択鋼材リスト
def variable2(member,col_list,beam_list,arraymat):

   H,B,tw,tf=[],[],[],[]
   D,td=[],[]
   size=np.shape(arraymat)
   sizez,sizey,sizex=size[0],int((size[1]+1)/2),int((size[2]+1)/2)###切り捨て
   ####Gx
   for i in range(size[0]):
     for j in range(size[1]):
       for k in range(size[2]):
         if arraymat[i,j,k]==2:
           if j%2==0:     
              id=member[i,j,k]
              h_=beam_list[str(id[2:])]['H']
              b_=beam_list[str(id[2:])]['B']
              tw_=beam_list[str(id[2:])]['tw']
              tf_=beam_list[str(id[2:])]['tf']
              H.append(int(h_))
              B.append(int(b_))
              tw.append(int(tw_)) 
              tf.append(int(tf_))              
   ####Gy
   for i in range(size[0]):
     for k in range(size[2]):
       for j in range(size[1]):
         if arraymat[i,j,k]==2:
           if k%2==0:    
              id=member[i,j,k]
              h_=beam_list[str(id[2:])]['H']
              b_=beam_list[str(id[2:])]['B']
              tw_=beam_list[str(id[2:])]['tw']
              tf_=beam_list[str(id[2:])]['tf']
              H.append(int(h_))
              B.append(int(b_))
              tw.append(int(tw_)) 
              tf.append(int(tf_))                              
   ####C
   for j in range(size[1]):
     for k in range(size[2]):
       for i in range(size[0]):
         if arraymat[i,j,k]==1:
            id=member[i,j,k]
            D_=col_list[str(id[2:])]['D']
            td_=col_list[str(id[2:])]['td'] 
            D.append(int(D_))
            td.append(int(td_))      
   
   var=np.concatenate([np.array(H),np.array(B),np.array(tw),np.array(tf)
                       ,np.array(D),np.array(td)])
   
   return var    
  
def variable_ex(member,col_list,beam_list,arraymat):

   H,B,tw,tf=[],[],[],[]
   D,td=[],[]
   size=np.shape(arraymat)
   sizez,sizey,sizex=size[0],int((size[1]+1)/2),int((size[2]+1)/2)
   ####Gx
   for j in range(sizey):
    for k in range(sizex):
      for i in range(sizez):
         if arraymat[i,j,k]==2:
           if j%2==0:     
              id=member[i,j,k]
              h_=beam_list[str(id[2:])]['H']
              b_=beam_list[str(id[2:])]['B']
              tw_=beam_list[str(id[2:])]['tw']
              tf_=beam_list[str(id[2:])]['tf']
              H.append(int(h_))
              B.append(int(b_))
              tw.append(int(tw_)) 
              tf.append(int(tf_))              
   ####Gy
   
   for k in range(sizex):
    for j in range(sizey):
      for i in range(sizez):
         if arraymat[i,j,k]==2:
           if k%2==0:    
              id=member[i,j,k]
              h_=beam_list[str(id[2:])]['H']
              b_=beam_list[str(id[2:])]['B']
              tw_=beam_list[str(id[2:])]['tw']
              tf_=beam_list[str(id[2:])]['tf']
              H.append(int(h_))
              B.append(int(b_))
              tw.append(int(tw_)) 
              tf.append(int(tf_))                              
   ####C
   for j in range(sizey):
     for k in range(sizex):
       for i in range(sizez):
         if arraymat[i,j,k]==1:
            id=member[i,j,k]
            D_=col_list[str(id[2:])]['D']
            td_=col_list[str(id[2:])]['td'] 
            D.append(int(D_))
            td.append(int(td_))      
             
   ####Gy
   
   var=np.concatenate([np.array(H),np.array(B),np.array(tw),np.array(tf)
                       ,np.array(D),np.array(td)])
   
   return var    

def code2idz(code_num,c_b):##部材番号と柱梁サイン0/1  a154⇒ 10 0 5 4 0xを外したものがインプット
  
  if c_b==0:
    if code_num=='0':output=[0,0,0,0]
    else:  
      pos = str(code_num).zfill(2)
      pos=list(pos)
      D=int(pos[0],16)
      t=int(pos[1],16)
      output=[D,t,0,0]
  else:
    if code_num=='0':output=[0,0,0,0]
    else:
      pos=str(code_num).zfill(4)
      pos=list(pos)
      H=int(pos[0],16)
      B=int(pos[1],16)
      tw=int(pos[2],16)
      tf=int(pos[3],16)
      output=[H,B,tw,tf]

  return output

def idz2code(idz_num,c_b):##部材番号と柱梁サイン0/1  10 1 5 4⇒ a154
  if c_b==0:
    
    D=str(hex(idz_num[0]))
    t=str(hex(idz_num[1])[2:])
    output=D+t
    
  else:
    H=str(hex(idz_num[0]))
    B=str(hex(idz_num[1])[2:])
    tw=str(hex(idz_num[2])[2:])
    tf=str(hex(idz_num[3])[2:]) 
    output=H+B+tw+tf
  
  return output  



def tanh(x):
    y = (np.exp(4*x) - np.exp(-4*x)) / (np.exp(4*x) + np.exp(-4*x))
    return y

def state_update(stress_ratio,cofX,cofY,deflect,num_el,margin,arraymat,member):

  if num_el==[0,0,0]:S=np.ones((1,57)).flatten()
  else:
    code=member[num_el[0],num_el[1],num_el[2]]
    dim=code2idz(str(code)[2:],arraymat[num_el[0],num_el[1],num_el[2]]-1)
    if arraymat[num_el[0],num_el[1],num_el[2]]==1:ss5=[(dim[0]-1)/13,(dim[0]-1)/13,(dim[1]-1)/9]
    else:                                         ss5=[(dim[0]-1)/12,(dim[1]-1)/4,(dim[3]-1)/8]
    size1=np.shape(stress_ratio)
    deflect=deflect.reshape((-1))
    s3_=deflect ############s3 層間変形角比
    
    s1=np.zeros((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin),dtype=np.float32)
    s3=np.zeros((size1[0]+2*margin),dtype=np.float32)
    
    s1[margin:-1*margin,margin:-1*margin,margin:-1*margin]=s1[margin:-1*margin,margin:-1*margin,margin:-1*margin]+stress_ratio
    s3[margin:-1*margin]=s3[margin:-1*margin]+s3_
    
    s2_1=np.zeros((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin),dtype=np.float32)
    s2_2=np.zeros((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin),dtype=np.float32)
    s2_1[margin:-1*margin,margin:-1*margin,margin:-1*margin]=s2_1[margin:-1*margin,margin:-1*margin,margin:-1*margin]+cofX
    s2_2[margin:-1*margin,margin:-1*margin,margin:-1*margin]=s2_2[margin:-1*margin,margin:-1*margin,margin:-1*margin]+cofY   
    if arraymat[num_el[0],num_el[1],num_el[2]]==1:#柱
      c1,c2=s2_1[num_el[0]-1,num_el[1],num_el[2]],s2_1[num_el[0],num_el[1],num_el[2]]
      c3,c4=s2_2[num_el[0]-1,num_el[1],num_el[2]],s2_2[num_el[0],num_el[1],num_el[2]]
      ss2=np.array([max([c1,c3]),max([c2,c4])])    
    else:
      if num_el[2]%2==1:
        c1,c2=s2_1[num_el[0],num_el[1],num_el[2]-1],s2_1[num_el[0],num_el[1],num_el[2]+1]
        ss2=np.array([c1,c2])
      else:
        c3,c4=s2_2[num_el[0],num_el[1]-1,num_el[2]],s2_2[num_el[0],num_el[1]+1,num_el[2]]
        ss2=np.array([c3,c4])

    ss1=s1[num_el[0],num_el[1],num_el[2]]
    ss3=s3[num_el[0]]
    if ss2[0]==0:ss2[0]=ss2[1]
    if ss2[1]==0:ss2[1]=ss2[0]
    ss2=ss2-1
    if ss2[0]>ss2[1]:np.array([ss2[1],ss2[0]])
    ss1=s1[num_el[0],num_el[1],num_el[2]]-1
    ss3=s3[num_el[0]]    -1
    
    
    S=np.concatenate([ss1.flatten(),ss2.flatten(),ss3.flatten()])
    for  i in range(len(S)):
      if S[i]>=0:
       S[i]=tanh(S[i])
    S=np.concatenate([S.flatten(),np.array(ss5).flatten()])
  return  S

def state_updateDIM(stress_ratio,cofX,cofY,deflect,num_el,margin,arraymat,member):
  if num_el==[0,0,0]:S=np.ones((1,57)).flatten()
  else:
    code=member[num_el[0],num_el[1],num_el[2]]
    dim=code2idz(str(code)[2:],arraymat[num_el[0],num_el[1],num_el[2]]-1)
    if arraymat[num_el[0],num_el[1],num_el[2]]==1:ss5=[(dim[0]-1)/13,(dim[0]-1)/13,(dim[1]-1)/9]
    else:                                         ss5=[(dim[0]-1)/12,(dim[1]-1)/4,(dim[3]-1)/8]
    ss5=np.array(ss5)
    size1=np.shape(stress_ratio)
    deflect=deflect.reshape((-1))

    s3_=deflect ############s3 層間変形角比

    s1=np.zeros((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin),dtype=np.float32)
    s3=np.zeros((size1[0]+2*margin),dtype=np.float32)
    
    s1[margin:-1*margin,margin:-1*margin,margin:-1*margin]=s1[margin:-1*margin,margin:-1*margin,margin:-1*margin]+stress_ratio
    s3[margin:-1*margin]=s3[margin:-1*margin]+s3_



    s2_1=np.zeros((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin),dtype=np.float32)
    s2_2=np.zeros((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin),dtype=np.float32)
    s2_1[margin:-1*margin,margin:-1*margin,margin:-1*margin]=s2_1[margin:-1*margin,margin:-1*margin,margin:-1*margin]+cofX
    s2_2[margin:-1*margin,margin:-1*margin,margin:-1*margin]=s2_2[margin:-1*margin,margin:-1*margin,margin:-1*margin]+cofY   
    if arraymat[num_el[0],num_el[1],num_el[2]]==1:#柱
      ss4=np.array([0,1])
      c1,c2=s2_1[num_el[0]-1,num_el[1],num_el[2]],s2_1[num_el[0],num_el[1],num_el[2]]
      c3,c4=s2_2[num_el[0]-1,num_el[1],num_el[2]],s2_2[num_el[0],num_el[1],num_el[2]]
      ss2=np.array([max([c1,c3]),max([c2,c4])])    
    else:
      ss4=np.array([1,0])
      if num_el[2]%2==1:
        c1,c2=s2_1[num_el[0],num_el[1],num_el[2]-1],s2_1[num_el[0],num_el[1],num_el[2]+1]
        ss2=np.array([c1,c2])
      else:
        c3,c4=s2_2[num_el[0],num_el[1]-1,num_el[2]],s2_2[num_el[0],num_el[1]+1,num_el[2]]
        ss2=np.array([c3,c4])
   
   
    ss1=s1[num_el[0],num_el[1],num_el[2]]
    ss3=s3[num_el[0]]
    if ss2[0]==0:ss2[0]=ss2[1]
    if ss2[1]==0:ss2[1]=ss2[0]
    ss2=ss2-1
    if ss2[0]>ss2[1]:np.array([ss2[1],ss2[0]])
    ss1=s1[num_el[0],num_el[1],num_el[2]]-1
    ss3=s3[num_el[0]]    -1
    S=np.concatenate([ss1.flatten(),ss2.flatten(),ss3.flatten()])
    for  i in range(len(S)):
      if S[i]>=0:
       S[i]=tanh(S[i])
    S=np.concatenate([S.flatten(),ss4.flatten(),ss5.flatten() ])
  return  S


def violate(stress_ratio,cofX,cofY,deflect,mrgn,arraymat):
  
  size1=np.shape(stress_ratio)
  violation_total=np.zeros((size1[0],size1[1],size1[2]),dtype=np.float32)
  arraymat=arraymat[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn]
  cof=np.maximum(cofX,cofY)

  for i in range(size1[0]) :
    for j in range(size1[1]) :
      for k in range(size1[2]) :
        ##許容応力度比1.0 
        if stress_ratio[i,j,k]>1.0 :
         violation_total[i,j,k]=violation_total[i,j,k]+stress_ratio[i,j,k] -1
        ##柱梁耐力比1.5  
        if cof[i,j,k]>1.0 and cof[i,j,k]>0 :
         violation_total[i,j,k]=violation_total[i,j,k]+cof[i,j,k]-1
        ##層間変形角1/200 
        if  deflect[0,i]>1.0 and arraymat[i,j,k]>0 :
         violation_total[i,j,k]=violation_total[i,j,k]+deflect[0,i]-1

  out=np.sum(violation_total)   
  
  return out



def mirror(target,maxmin,symFlag):  
  size=np.shape(target)
  a1=target
  if symFlag==1:
      a2=np.rot90(a1, axes=(1,2))
      a3=np.rot90(a2, axes=(1,2))
      a4=np.rot90(a3, axes=(1,2))
      a5=a1[:,:,::-1]
      a6=np.rot90(a5, axes=(1,2))
      a7=np.rot90(a6, axes=(1,2))
      a8=np.rot90(a7, axes=(1,2))

      a1=a1.reshape(-1)
      a2=a2.reshape(-1)
      a3=a3.reshape(-1)
      a4=a4.reshape(-1)
      a5=a5.reshape(-1)
      a6=a6.reshape(-1)
      a7=a7.reshape(-1)
      a8=a8.reshape(-1)  
    
      for i in range(len(a1)):
        if type(a1[0]) is str:
          set_=[int(a1[i],16),int(a2[i],16),int(a3[i],16),int(a4[i],16),int(a5[i],16),int(a6[i],16),int(a7[i],16),int(a8[i],16)]
        else:
          set_=[a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i],a8[i]]
        set=[a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i],a8[i]] 
        if maxmin==0:
          a1[i]=set[np.argmax([set_])]
        else:
          a1[i]=set[np.argmin([set_])]
  else:
      a2=np.flip(a1,axis=1)
      a3=np.flip(a1,axis=2)
      a4=np.flip(a1,axis=(1,2))

      a1=a1.reshape(-1)
      a2=a2.reshape(-1)
      a3=a3.reshape(-1)
      a4=a4.reshape(-1)
    
      for i in range(len(a1)):
        if type(a1[0]) is str:
          set_=[int(a1[i],16),int(a2[i],16),int(a3[i],16),int(a4[i],16)]
        else:
          set_=[a1[i],a2[i],a3[i],a4[i]]
        set=[a1[i],a2[i],a3[i],a4[i]] 
        if maxmin==0:
          a1[i]=set[np.argmax([set_])]
        else:
          a1[i]=set[np.argmin([set_])]         
  return a1.reshape(size[0],size[1],size[2])


def V(member,lengthmat,margin):#◎


  Vt=0
  member=member[margin:-margin,margin:-margin,margin:-margin]
  size=np.shape(member)  

  for i in range(size[0]):
    for j in range(size[1]):
      for k in range(size[2]):
        code=member[i,j,k]
        if len(code)==4:
          code=str(code)[2:]
          Dtd=code2idz(code,0)
          D=list_D[Dtd[0]-1]
          td=list_t[Dtd[1]-1]

          Vc=D*D-(D-2*td)*(D-2*td)
    
          Vt=Vt+Vc/1000000*lengthmat[i,j,k]/1000
        elif len(code)==6:
          code=str(code)[2:]
          HBtwtf=code2idz(code,1)
          H=list_H[HBtwtf[0]-1]
          B=list_B[HBtwtf[1]-1]
          tw=list_tw[HBtwtf[2]-1]
          tf=list_tf[HBtwtf[3]-1]      
          Vb=H*B-(H-2*tf)*(B-tw)
          Vt=Vt+Vb/1000000*lengthmat[i,j,k]/1000    
  Vt=np.round(Vt,4)
  return  Vt   

def A(member,num_el):#◎


  code=member[num_el[0],num_el[1],num_el[2]]

  if len(code)==4:
    code=str(code)[2:]
    Dtd=code2idz(code,0)
    D=list_D[Dtd[0]-1]
    td=list_t[Dtd[1]-1]

    A=D*D-(D-2*td)*(D-2*td)


  elif len(code)==6:
    code=str(code)[2:]
    HBtwtf=code2idz(code,1)
    H=list_H[HBtwtf[0]-1]
    B=list_B[HBtwtf[1]-1]
    tw=list_tw[HBtwtf[2]-1]
    tf=list_tf[HBtwtf[3]-1]      
    A=H*B-(H-2*tf)*(B-tw)

  
  return  A   






def reward_gvn(dV,violation,dviolation) :
  #### G REWARD
  reward=0
  if violation==0  :reward=reward+dV
  else:
    reward=reward+dviolation

 
  
  return reward


def update_flag(dV,violation,dG) :
  if violation==0:
    if   dV>0 and dG>0:flag=True
    elif dV<0 and dG>0:flag=False
    elif dV>0 and dG<0:flag=False
    elif dV<0 and dG<0:flag=False
    elif dV>0 and dG==0:flag=True
    elif dV==0 and dG>0:flag=True
    elif dV<0 and dG==0:flag=False
    elif dV==0 and dG<0:flag=False
    elif dV==0 and dG==0:flag=True
  else:
    if   dV>0 and dG>0:flag=True
    elif dV<0 and dG>0:flag=True
    elif dV>0 and dG<0:flag=False
    elif dV<0 and dG<0:flag=False
    elif dV>0 and dG==0:flag=True
    elif dV==0 and dG>0:flag=True
    elif dV<0 and dG==0:flag=False
    elif dV==0 and dG<0:flag=False
    elif dV==0 and dG==0:flag=True

  
  return flag





# 数値→アルファベット
def num2alpha(num):
    if num<=26:
        return chr(64+num)
    elif num%26==0:
        return num2alpha(num//26-1)+chr(90)
    else:
        return num2alpha(num//26)+chr(64+num%26)




##構造解析
def r_cof(member,col_list,beam_list,mrgn,xSpan,ySpan,zSpan,arraymat,symFlag) :
  variable_=variable1(member,col_list,beam_list,arraymat)
  variable_ =matlab.double(variable_.tolist())
  stress_bx,stress_by,stress_c,cof,deflect= eng.execute_stkcv(variable_,nargout=5)
  stress_bx=list(np.array(stress_bx))[0]
  stress_by=list(np.array(stress_by))[0]
  stress_c=list(np.array(stress_c))[0]
  cof=list(np.array(cof))[0]
  cof=cof.reshape((2,-1))
  z=int(zSpan)
  x=int(xSpan)
  y=int(ySpan)
  lx=2*x+1
  ly=2*y+1
  id_bx,id_by,id_c,id_cof=0,0,0,0

  stress_ratio=np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))
  #cofX,Y
  cof_nodeX= np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))
  cof_mainX=np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))  
  cof_nodeY= np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))
  cof_mainY=np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn)) 

  ####stress_bx,cof
  for i in range(z+2*mrgn):
    for j in range(ly+2*mrgn):
      for k in range(lx+2*mrgn):
        if arraymat[i,j,k]==2:
          if j%2==0:
             stress_ratio[i,j,k]= stress_bx[id_bx]
             id_bx=id_bx+1
        elif arraymat[i,j,k]==1:  
             cof_nodeX[i,j,k]=cof[0,id_cof]
             cof_nodeY[i,j,k]=cof[1,id_cof]
             id_cof=id_cof+1       
            
  ####stress_by
  for i in range(z+2*mrgn):
    for k in range(lx+2*mrgn):
      for j in range(ly+2*mrgn):
        if arraymat[i,j,k]==2:
          if k%2==0:
             stress_ratio[i,j,k]= stress_by[id_by]
             id_by=id_by+1   
  ####stress_c
  for j in range(ly+2*mrgn):
    for k in range(lx+2*mrgn):
      for i in range(z+2*mrgn):
        if arraymat[i,j,k]==1:
             stress_ratio[i,j,k]= stress_c[id_c]
             id_c=id_c+1
 
  stress_ratio=np.round(stress_ratio,3)           
  stress_ratio=mirror(stress_ratio,0,symFlag)
  deflect=np.array(deflect)*200 

  #cofX,Y

  for i in range(z+2*mrgn) :
    for j in range(ly+2*mrgn) :
      for k in range(lx+2*mrgn) :
        if cof_nodeX[i,j,k]>0:
          cof_mainX[i,j,k]=1.5/cof_nodeX[i,j,k]
          cof_mainY[i,j,k]=1.5/cof_nodeY[i,j,k]
  cof_mainX=np.round(cof_mainX,3)
  cof_mainY=np.round(cof_mainY,3)
  # cof[cof!=0]=1.5/cof[cof!=0]
  # print(cof[0,:].reshape((-1,3,3)))
  # print(cof[1,:].reshape((-1,3,3)))
  # print(cof_mainX[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn])
  # print(cof_mainY[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn])
  return (stress_ratio[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn],cof_mainX[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn]
                      ,cof_mainY[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn],deflect)


##構造解析
def r_cof2(member,col_list,beam_list,mrgn,xSpan,ySpan,zSpan,arraymat,symFlag) :
  variable_=variable2(member,col_list,beam_list,arraymat)
  variable_ =matlab.double(variable_.tolist())
  stress_bx,stress_by,stress_c,cof,deflect= eng.execute_stkcv2(variable_,5,nargout=5)
  stress_bx=list(np.array(stress_bx))[0]
  stress_by=list(np.array(stress_by))[0]
  stress_c=list(np.array(stress_c))[0]
  cof=list(np.array(cof))[0]
  cof=cof.reshape((2,-1))
  z=int(zSpan)
  x=int(xSpan)
  y=int(ySpan)
  lx=2*x+1
  ly=2*y+1
  id_bx,id_by,id_c,id_cof=0,0,0,0

  stress_ratio=np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))
  #cofX,Y
  cof_nodeX= np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))
  cof_mainX=np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))  
  cof_nodeY= np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn))
  cof_mainY=np.zeros((z+2*mrgn,ly+2*mrgn,lx+2*mrgn)) 

  ####stress_bx,cof
  for i in range(z+2*mrgn):
    for j in range(ly+2*mrgn):
      for k in range(lx+2*mrgn):
        if arraymat[i,j,k]==2:
          if j%2==0:
             stress_ratio[i,j,k]= stress_bx[id_bx]
             id_bx=id_bx+1
        elif arraymat[i,j,k]==1:  
             cof_nodeX[i,j,k]=cof[0,id_cof]
             cof_nodeY[i,j,k]=cof[1,id_cof]
             id_cof=id_cof+1       
            
  ####stress_by
  for i in range(z+2*mrgn):
    for k in range(lx+2*mrgn):
      for j in range(ly+2*mrgn):
        if arraymat[i,j,k]==2:
          if k%2==0:
             stress_ratio[i,j,k]= stress_by[id_by]
             id_by=id_by+1   
  ####stress_c
  for j in range(ly+2*mrgn):
    for k in range(lx+2*mrgn):
      for i in range(z+2*mrgn):
        if arraymat[i,j,k]==1:
             stress_ratio[i,j,k]= stress_c[id_c]
             id_c=id_c+1
 
  stress_ratio=np.round(stress_ratio,3)           
  # stress_ratio=mirror(stress_ratio,0,symFlag)
  deflect=np.array(deflect)*200 

  #cofX,Y

  for i in range(z+2*mrgn) :
    for j in range(ly+2*mrgn) :
      for k in range(lx+2*mrgn) :
        if cof_nodeX[i,j,k]>0:
          cof_mainX[i,j,k]=1.5/cof_nodeX[i,j,k]
          cof_mainY[i,j,k]=1.5/cof_nodeY[i,j,k]
  cof_mainX=np.round(cof_mainX,3)
  cof_mainY=np.round(cof_mainY,3)
  # cof[cof!=0]=1.5/cof[cof!=0]
  # print(cof[0,:].reshape((-1,3,3)))
  # print(cof[1,:].reshape((-1,3,3)))
  # print(cof_mainX[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn])
  # print(cof_mainY[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn])
  return (stress_ratio[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn],cof_mainX[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn]
                      ,cof_mainY[mrgn:-1*mrgn,mrgn:-1*mrgn,mrgn:-1*mrgn],deflect)


def mirrorEL(target,maxmin,symFlag):  
  size=np.shape(target)
  a1=target
  if symFlag==1:
      a2=np.rot90(a1, axes=(1,2))
      a3=np.rot90(a2, axes=(1,2))
      a4=np.rot90(a3, axes=(1,2))
      a5=a1[:,:,::-1]
      a6=np.rot90(a5, axes=(1,2))
      a7=np.rot90(a6, axes=(1,2))
      a8=np.rot90(a7, axes=(1,2))

      a1=a1.reshape(-1)
      a2=a2.reshape(-1)
      a3=a3.reshape(-1)
      a4=a4.reshape(-1)
      a5=a5.reshape(-1)
      a6=a6.reshape(-1)
      a7=a7.reshape(-1)
      a8=a8.reshape(-1)  
    
      for i in range(len(a1)):
        if type(a1[0]) is str:
          set_=[int(a1[i],16),int(a2[i],16),int(a3[i],16),int(a4[i],16),int(a5[i],16),int(a6[i],16),int(a7[i],16),int(a8[i],16)]
        else:
          set_=[a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i],a8[i]]
        set=[a1[i],a2[i],a3[i],a4[i],a5[i],a6[i],a7[i],a8[i]]
        if maxmin==0:
          a1[i]=set[np.argmax([set_])]
        else:
          if np.max(set_)==0:a1[i]=set[np.argmin([set_])]
          else:
            p,pp=[],[]
            for k in range(len(set_)):
              if not set_[k]==0:
                p.append(set_[k])
                pp.append(set[k])
            a1[i]=pp[np.argmin(np.array(p))]  
 
          # a1[i]=set[np.argmin([set_])]
  else:
      a2=np.flip(a1,axis=1)
      a3=np.flip(a1,axis=2)
      a4=np.flip(a1,axis=(1,2))

      a1=a1.reshape(-1)
      a2=a2.reshape(-1)
      a3=a3.reshape(-1)
      a4=a4.reshape(-1)
    
      for i in range(len(a1)):
        if type(a1[0]) is str:
          set_=[int(a1[i],16),int(a2[i],16),int(a3[i],16),int(a4[i],16)]
        else:
          set_=[a1[i],a2[i],a3[i],a4[i]]
        set=[a1[i],a2[i],a3[i],a4[i]] 
        if maxmin==0:
          a1[i]=set[np.argmax([set_])]
        else:
          a1[i]=set[np.argmin([set_])]         
  return a1.reshape(size[0],size[1],size[2])
def FRAME_gen(Model,arraymat,margin,vecFrame,symFlag):
  size1=np.shape(arraymat)
  init_member=np.full(size1[0]*size1[1]*size1[2],'aaa',dtype=object)
  member=np.full((size1[0]+2*margin)*(size1[1]+2*margin)*(size1[2]+2*margin),'aaa',dtype=object)
  if vecFrame=='x':
    init_member=init_member.reshape((size1[0],size1[1],size1[2]))

    member=member.reshape((size1[0]+2*margin,size1[1]+2*margin,size1[2]+2*margin)) 
  else:
    init_member=init_member.reshape((size1[0],size1[2],size1[1]))
    member=member.reshape((size1[0]+2*margin,size1[2]+2*margin,size1[1]+2*margin)) 
    arraymat=arraymat.transpose((0,2,1))        
  size1= np.shape(init_member)
  for i in range(size1[0]):
   for j in range(size1[1]):
    for k in range(size1[2]):
      init_member[i,j,k]="0"

  ##連続最適解生成   
  G_IN,C_IN=eng.output(Model[2],Model[1],Model[4]*1000,Model[5]*1000,Model[0],'i',vecFrame,nargout=2)
  G_IN,C_IN=np.array(G_IN).reshape((4,-1)).T.tolist(),np.array(C_IN).reshape((-1,2)).tolist()
  G_OUT,C_OUT=eng.output(Model[2],Model[1],Model[4]*1000,Model[5]*1000,Model[0],'o',vecFrame,nargout=2)
  G_OUT,C_OUT=np.array(G_OUT).reshape((4,-1)).T.tolist(),np.array(C_OUT).reshape((-1,2)).tolist()
  # print('G_IN,C_IN')
  # print(G_IN,C_IN)
  # print('G_OUT,C_OUT')
  # print(G_OUT,C_OUT)

  ##行列の形⇒リスト型
  ##立体化＆離散化
  for j in range(size1[1]):
    idx1=0
    idx2=0 
    if j%2==0:
      if j==0 or j==size1[1]-1:
        G,C=G_OUT,C_OUT
      else:
        G,C=G_IN,C_IN
      for i in range(size1[0]):
        for k in range(size1[2]) :
         if k<=size1[2]/2:
          if arraymat[i,j,k]==2:
           H=np.argmin(abs(np.array(list_H)-G[idx1][0]))+1
           B=np.argmin(abs(np.array(list_B)-G[idx1][1]))+1
           tw=np.argmin(abs(np.array(list_tw)-G[idx1][2]))+1
           tf=np.argmin(abs(np.array(list_tf)-G[idx1][3]))+1           

           idx1=idx1+1
           init_member[i,j,k]=idz2code([H,B,tw,tf],1)  
 
                 
      for k in range(size1[2]):
        for i in range(size1[0]) : 
         if k<=size1[2]/2:
          if arraymat[i,j,k]==1:       
           D=np.argmin(abs(np.array(list_D)-C[idx2][0]))+1
           t=np.argmin(abs(np.array(list_t)-C[idx2][1]))+1
           idx2=idx2+1
           init_member[i,j,k]=idz2code([D,t],0) 

  for i in range(size1[0]+2*margin) :
    for j in range(int(size1[1]+2*margin)) :
      for k in range(int(size1[2]+2*margin)) :  
        member[i,j,k]="0"    
        
  member[margin:-margin,margin:-margin,margin:-margin]=init_member
  f=0
  if np.random.rand()>=0.5:f=0
  member=mirrorEL(member,f,symFlag) 
  if vecFrame=='y':
    member=member.transpose((0,2,1))
  return member
# zSpan=3
# xSpan=2
# ySpan=xSpan
# lx=2*xSpan+1
# ly=2*ySpan+1

# length=6000
# zlength=np.ones((zSpan))*4000
# margin=2
# arraymat=framemat(zSpan,lx,ly)
# Model=[zSpan,ySpan,xSpan,4,length/1000,length/1000]

# member=FRAME_gen(Model,arraymat,margin)
# print(member)




