# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:31:31 2014

@author: Huang,Zheng

Linear CRF in Python

License (BSD)
==============
Copyright (c) 2013, Huang,Zheng.  huang-zheng@sjtu.edu.cn
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy
from scipy.misc import logsumexp
import os
import codecs
import re
import datetime
import ctypes
import multiprocessing
from multiprocessing import Process, Queue
import sys

_gradient = None  #  global variable used to store the gradient calculated in Liklihood function.

def logdotexp_vec_mat(loga,logM):
        return logsumexp(loga+logM,axis=1)
        
def logdotexp_mat_vec(logM,logb):
    return logsumexp(logM+logb[:,numpy.newaxis],axis=0)

def validTemplateLine(strt):
   ifvalid=True
   if strt.count("[")!=strt.count("]"):
       ifvalid=False
   if "UuBb".find(strt[0])==-1:
       ifvalid=False
   if ifvalid==False:
       print "error in template file:", strt
   return ifvalid

def readData(dataFile):
    texts = []
    labels = []
    text = []
    label=[]
    obydic=dict()
    file = codecs.open(dataFile, 'r')  #  default encoding.
    obyid=0
    linecnt=0
    spacecnt=0
    for line in file:
        #print line
        line = line.strip()
        if len(line) == 0:
            if len(text)>0:
                texts.append(text)
                labels.append(label)
            text = []
            label = []
        else:
            linecnt+=1
            if linecnt % 10000 == 0 :
                print "read ",linecnt , " lines."
            chunk = line.split()
            if spacecnt==0:
                spacecnt=len(chunk)
            else:
                if len(chunk)!=spacecnt:
                    print "Error in input data:",line
            text.append(chunk[0:-1])
            ylabel=chunk[-1]
            if obydic.has_key(ylabel)==False:
                obydic[ylabel]=obyid
                label.append(obyid)
                obyid+=1
            else:
               label.append(obydic[ylabel])
              
    if len(text)>0:  # sometimes, there is no empty line at the end of file.
        texts.append(text)
        labels.append(label)
    
    #print texts,labels
    texts=texts
    oys=labels
    seqnum=len(oys)
    seqlens=[len(x) for x in texts]
    K = len(obydic)
    y2label = dict([(obydic[key],key) for key in obydic.keys()])
    print "number of labels:", K
    return texts,seqlens,oys,seqnum,K,obydic,y2label
   
def readTemplate(tmpFile):
    tlist=[]  # list of list(each template line)  
    file = codecs.open(tmpFile, 'r')  #  default encoding.
    #repat=r'\[\d+,\d+\]'
    repat=r'\[-?\d+,-?\d+\]'    #-?[0-9]*
    for line in file:
        #print line    
        line=line.strip()
        if len(line)==0:
            continue
        if line[0]=="#":  # not comment line
            continue
        fl=line.find("#")
        if fl!=-1:  # remove the comments in the same line.
            line=line[0:fl]
        if validTemplateLine(line)==False:
            continue
        fl=line.find(":")
        if fl!=-1:  # just a symbol
            eachlist=[line[0:fl]]
        else:
            eachlist=[line[0]]
            
        for a in list(re.finditer(repat, line)):
            locstr=line[a.start()+1:a.end()-1]
            loc = locstr.split(",")
            eachlist.append(loc) 
            #print a.start(),a.end()
        tlist.append(eachlist)
    print "Valid Template Line Number:",len(tlist)
    return tlist


def expandOBX(texts,seqid,locid,tp):  # expend the observation at locid for sequence(seqid)
    strt=tp[0]
    for li in tp[1::]:
        row=locid+int(li[0]); col=int(li[1])
        if row>=0 and row<len(texts[seqid]):
            if col>=0 and col<len(texts[seqid][row]):
                strt+= ":" + texts[seqid][row][col]
    #print strt
    return strt       

    
def processFeatures(tplist,texts,seqnum,K,fd=1):
    uobxs =  dict(); bobxs=dict() ; 
    '''add feature reduction here'''
    for ti,tp in enumerate(tplist):  # for each template line
        for sid in range(seqnum):  # for each traning sequence.
            for lid in range(len(texts[sid])):
                obx=expandOBX(texts,sid,lid,tp)
                if obx[0]=="B":
                    if bobxs.has_key(obx)==False:
                        bobxs[obx]=1
                    else:
                        tval= bobxs[obx]
                        bobxs[obx]= tval+1
                        
                if obx[0]=="U":
                    if uobxs.has_key(obx)==False:
                        uobxs[obx]=1
                    else:
                        tval= uobxs[obx]
                        uobxs[obx]= tval+1
                        
    if fd>=2:  # need to remove ub frequency less than fd
        uobxnew = { k : v for k,v in uobxs.iteritems() if v >= fd }
        bobxnew = { k : v for k,v in bobxs.iteritems() if v >= fd }
        del uobxs; del bobxs;
        uobxs,bobxs=uobxnew,bobxnew
    
    ufnum, bfnum = 0 , 0                    
    for obx in bobxs.keys():
        bobxs[obx]=bfnum
        bfnum+=K*K
    for obx in uobxs.keys():
        uobxs[obx]=ufnum
        ufnum+=K
    return uobxs,bobxs,ufnum,bfnum

def calObservexOn(tplist,texts,uobxs,bobxs,seqnum):
    '''speed up the feature calculation
      calculate the on feature functions ''' 
    uon=[]; bon=[]
    for sid in range(seqnum):  # for each traning sequence.
        sequon=[];seqbon=[]
        for lid in range(len(texts[sid])):
            luon=[];lbon=[]
            for ti,tp in enumerate(tplist):  # for each template line
                obx=expandOBX(texts,sid,lid,tp)
                #skey = str(ti)+":"+str(sid)+":"+str(lid)
                #obx=oball[skey]
                if tp[0][0]=="B":
                    fid=bobxs.get(obx)
                    #print fid
                    if fid!=None:
                        lbon.append(fid)
                if tp[0][0]=="U":
                    fid=uobxs.get(obx)
                    if fid!=None:
                        luon.append(fid)
            sequon.append(luon);seqbon.append(lbon)
        uon.append(sequon);bon.append(seqbon)
    return uon,bon

def calObservexOnLoc(uon,bon,seqnum,mp):
    '''speed up the feature calculation (Mulitprocessing)
      calculate the on feature list and location ''' 
    ulen=0 ; loclen=0
    for a in uon:
        loclen+=len(a)
        for b in a:
            ulen+=len(b)
    #ulen = sum[(len(b)) for b in a for a in uon]
    if sys.platform=="win32" and mp==1:   # windows system need shared memory to do multiprocessing
        uonarr=multiprocessing.Array('i', ulen)
        uonseqsta=multiprocessing.Array('i', seqnum)
        uonlocsta=multiprocessing.Array('i', loclen)
        uonlocend=multiprocessing.Array('i', loclen)     
    else:
        uonarr=numpy.zeros((ulen),dtype=numpy.int)
        uonseqsta=numpy.zeros((seqnum),dtype=numpy.int)
        uonlocsta=numpy.zeros((loclen),dtype=numpy.int)
        uonlocend=numpy.zeros((loclen),dtype=numpy.int)
        
    uid=0 ; seqi = 0 ; loci=0
    for seq in uon:  # for each traning sequence.
        uonseqsta[seqi]=loci
        for loco in seq:
            uonlocsta[loci]=uid
            for aon in loco:
                uonarr[uid]=aon
                uid+=1
            uonlocend[loci]=uid
            loci+=1
        seqi+=1
        
    blen=0 ; loclen=0
    for a in bon:
        loclen+=len(a)
        for b in a:
            blen+=len(b)
    #ulen = sum[(len(b)) for b in a for a in uon]
            
    if sys.platform=="win32" and mp==1:   # windows system need shared memory to do multiprocessing
        bonarr=multiprocessing.Array('i', ulen)
        bonseqsta=multiprocessing.Array('i', seqnum)
        bonlocsta=multiprocessing.Array('i', loclen)
        bonlocend=multiprocessing.Array('i', loclen)     
    else:
        bonarr=numpy.zeros((ulen),dtype=numpy.int)
        bonseqsta=numpy.zeros((seqnum),dtype=numpy.int)
        bonlocsta=numpy.zeros((loclen),dtype=numpy.int)
        bonlocend=numpy.zeros((loclen),dtype=numpy.int)
        
    bid=0 ; seqi = 0 ; loci=0
    for seq in bon:  # for each traning sequence.
        bonseqsta[seqi]=loci
        for loco in seq:
            bonlocsta[loci]=bid
            for aon in loco:
                bonarr[bid]=aon
                bid+=1
            bonlocend[loci]=bid
            loci+=1
        seqi+=1
    #print len(uonlocsta)
    return uonarr,uonseqsta,uonlocsta,uonlocend, bonarr,bonseqsta,bonlocsta,bonlocend

def calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0):
    fss=numpy.zeros((ufnum+bfnum))
    fssb=fss[0:bfnum]
    fssu=fss[bfnum:]
    for i in range(seqnum):
        for li in range(len(texts[i])):
            for ao in uon[i][li]:
                fssu[ao+oys[i][li]]+=1.0
            for ao in bon[i][li]:
                if li==0:  # the first , yt-1=y0
                    fssb[ao+oys[i][li]*K+y0]+=1.0
                else:
                    fssb[ao+oys[i][li]*K+oys[i][li-1]]+=1.0
    return fss

def random_param(ufnum,bfnum):
    #theta=numpy.random.randn(ufnum+bfnum)
    theta=numpy.ones(ufnum+bfnum)
    return theta

def regularity(theta,type=0,sigma=1.0):
    if type == 0:
        regularity = 0
    elif type == 1:
        regularity = numpy.sum(numpy.abs(theta)) / sigma
    else:
        v = sigma ** 2
        v2 = v * 2
        regularity = numpy.sum(numpy.dot(theta,theta) )/ v2
    return regularity

def regularity_deriv(theta,type=0,sigma=1.0):
    if type == 0:
        regularity_deriv = 0
    elif type == 1:
        regularity_deriv = numpy.sign(theta) / sigma
    else:
        v = sigma ** 2
        regularity_deriv = theta / v
    return regularity_deriv

#def logM_b(seqlen,seqid,uon,bon,K, thetau,thetab):
#    ''' logMlist (n, K, K ) --> (sequence length, Yt, Yt-1)'''
#    mlist=[]
#    for li in range(seqlen):
#        fv = numpy.zeros((K,K))
#        for ao in uon[seqid][li]:
#            fv+=thetau[ao:ao+K][:,numpy.newaxis]
#        for ao in bon[seqid][li]:
#            fv+=thetab[ao:ao+K*K].reshape((K,K))
#        mlist.append(fv)
#    
#    for i in range(0,K):  # set the energe function for ~y(0) to be -inf.
#        mlist[0][i][1:]= - float("inf")
#    #print "mlist:",mlist
#    return mlist      

def logMarray(seqlen,auon,abon,K, thetau,thetab):
    ''' logMlist (n, K, K ) --> (sequence length, Yt, Yt-1)'''
    mlist=[]
    for li in range(seqlen):
        fv = numpy.zeros((K,K))
        for ao in auon[li]:
            fv+=thetau[ao:ao+K][:,numpy.newaxis]
        for ao in abon[li]:
            fv+=thetab[ao:ao+K*K].reshape((K,K))
        mlist.append(fv)
    
    for i in range(0,K):  # set the energe function for ~y(0) to be -inf.
        mlist[0][i][1:]= - float("inf")
    #print "mlist:",mlist
    return mlist      

def logM_sa(seqlen,seqid,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,K, thetau,thetab):
    ''' logMlist (n, K, K ) --> (sequence length, Yt, Yt-1)'''
    mlist=[]
    iloc = uonseqsta[seqid]
    ilocb = bonseqsta[seqid]
    for li in range(seqlen):
        fv = numpy.zeros((K,K))
        for i in range(uonlocsta[iloc+li],uonlocend[iloc+li]):
            ao=uonarr[i]
            fv+=thetau[ao:ao+K][:,numpy.newaxis]
        for i in range(bonlocsta[ilocb+li],bonlocend[ilocb+li]):
            ao=bonarr[i]
            fv+=thetab[ao:ao+K*K].reshape((K,K))
        mlist.append(fv)
    
    for i in range(0,K):  # set the energe function for ~y(0) to be -inf.
        mlist[0][i][1:]= - float("inf")
    #print "mlist:",mlist
    return mlist      

'''numpy version of logM, it is even slower than list version'''
#def logMarray(seqlen,seqid,uon,bon,K, thetau,thetab):  
#    ''' logMlist (n, K, K ) --> (sequence length, Yt, Yt-1)'''
#    mlist=numpy.zeros((seqlen,K,K))
#    for li in range(seqlen):
#        fv = mlist[li]
#        for ao in uon[seqid][li]:
#            fv+=thetau[ao:ao+K][:,numpy.newaxis]
#        for ao in bon[seqid][li]:
#            fv+=thetab[ao:ao+K*K].reshape((K,K))
#        #mlist.append(fv)
#    
#    for i in range(0,K):  # set the energe function for ~y(0) to be -inf.
#        mlist[0][i][1:]= - float("inf")
#    #print "mlist:",mlist
#    return mlist      


def logAlphas(Mlist):
    logalpha = Mlist[0][:,0] # alpha(1)
    logalphas = [logalpha]
    for logM in Mlist[1:]:
        logalpha = logdotexp_vec_mat(logalpha, logM)
        logalphas.append(logalpha)
    #print "logalphas:",logalphas
    return logalphas
    
def logBetas(Mlist):
    logbeta = numpy.zeros_like(Mlist[-1][:, 0])
    logbetas = [logbeta]
    for logM in Mlist[-1:0:-1]:
        logbeta = logdotexp_mat_vec(logM, logbeta)
        logbetas.append(logbeta)
    #print "logbeta:",logbetas[::-1]
    return logbetas[::-1]


def likelihood_standalone(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    '''conditional log likelihood log p(Y|X)'''
    likelihood = numpy.dot(fss,theta)
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood=0.0
    for seqid in range(seqnum):
        logMlist = logMarray(seqlens[seqid],uon[seqid],bon[seqid],K, thetau,thetab)
        logZ = logsumexp(logAlphas(logMlist)[-1])
        #print "logz in likelihood:",logZ
        likelihood -= logZ
    return likelihood - regularity(theta,regtype,sigma)

def likelihoodthread_o(seqlens,uon,bon,thetau,thetab,seqnum,K,ufnum,bfnum,starti,endi,que):
    likelihood=0.0
    for seqid in range(starti,endi):
        logMlist = logMarray(seqlens[seqid],uon[seqid],bon[seqid],K, thetau,thetab)
        logZ = logsumexp(logAlphas(logMlist)[-1])
        likelihood -= logZ
    que.put(likelihood)
                       
#def likelihoodthread(seqlens,uon,bon,theta,seqnum,K,ufnum,bfnum,starti,endi,que1,que2):
def likelihoodthread(seqlens,theta,seqnum,K,ufnum,bfnum,starti,endi,que1,que2,ns):
    uon=ns.uon ; bon = ns.bon
    grad = numpy.zeros(ufnum+bfnum)  
    likelihood = 0
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    for si in range(starti,endi):
        #logMlist = logMarray(seqlens[si],si,uon,bon,K,thetau,thetab)
        logMlist = logMarray(seqlens[si],uon[si],bon[si],K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            for ao in uon[si][i]:
                gradu[ao:ao+K] -= p_yi
            for ao in bon[si][i]:
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    que1.put(likelihood)
    que2.put(grad)
        
        
        
        
#def likelihoodthread(seqlens,uon,bon,theta,seqnum,K,ufnum,bfnum,starti,endi,que1,que2):
def likelihoodthread_simple(seqlens,uon,bon,theta,K,ufnum,bfnum,que1,que2):
    grad = numpy.zeros(ufnum+bfnum)  
    likelihood = 0
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    for si in range(len(seqlens)):
        #logMlist = logMarray(seqlens[si],si,uon,bon,K,thetau,thetab)
        logMlist = logMarray(seqlens[si],uon[si],bon[si],K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            for ao in uon[si][i]:
                gradu[ao:ao+K] -= p_yi
            for ao in bon[si][i]:
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    que1.put(likelihood)
    que2.put(grad)        


def likelihoodthread_sa(seqlens,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,theta,starti,endi,seqnum,K,ufnum,bfnum,que1,que2):
    grad = numpy.zeros(ufnum+bfnum)  
    likelihood = 0
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    for si in range(starti,endi):
        #logMlist = logMarray(seqlens[si],si,uon,bon,K,thetau,thetab)
        logMlist = logM_sa(seqlens[si],si,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            iloc = uonseqsta[si]
            for it in range(uonlocsta[iloc+i],uonlocend[iloc+i]):            
                ao=uonarr[it]
                gradu[ao:ao+K] -= p_yi
                
            iloc = bonseqsta[si]    
            for it in range(bonlocsta[iloc+i],bonlocend[iloc+i]):            
                ao=bonarr[it]
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
                
    que1.put(likelihood)
    que2.put(grad)        


def likelihood_multithread_O(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum):   # multithread version of likelihood
    '''conditional log likelihood log p(Y|X)'''
    likelihood = numpy.dot(fss,theta)
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    que = Queue()
    np = 0
    subprocesses = []
    corenum=multiprocessing.cpu_count()
    #corenum=1
    if corenum>1:
        chunk=seqnum/corenum+1
    else:
        chunk=seqnum
    starti=0
    while starti < (seqnum):
        endi=starti+chunk
        if endi>seqnum:
            endi=seqnum    
        p = Process(target=likelihoodthread, 
           args=(seqlens,uon,bon,thetau,thetab,seqnum,K,ufnum,bfnum,starti,endi,que))
        p.start()
        np+=1
        #print 'delegated %s:%s to subprocess %s' % (starti, endi, np)
        subprocesses.append(p)
        starti += chunk
    for i in range(np):
        likelihood += que.get()
    while subprocesses:
        subprocesses.pop().join()
    return likelihood - regularity(theta)


#def likelihood_mp(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
def likelihood_mp(seqlens,fss,theta,seqnum,K,ufnum,bfnum,regtype,sigma,ns):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    likelihood = numpy.dot(fss,theta)
    que1 = Queue() # for the likihood output
    que2 = Queue() # for the gradient output
    np = 0
    subprocesses = []
    corenum=multiprocessing.cpu_count()
    #corenum=1
    if corenum>1:
        chunk=seqnum/corenum+1
    else:
        chunk=seqnum
    starti=0
    while starti < (seqnum):
        endi=starti+chunk
        if endi>seqnum:
            endi=seqnum    
        p = Process(target=likelihoodthread, 
            args=(seqlens,theta,seqnum,K,ufnum,bfnum,starti,endi,que1,que2,ns))
           #args=(seqlens,uon,bon,theta,seqnum,K,ufnum,bfnum,starti,endi,que1,que2))
        p.start()
        np+=1
        #print 'delegated %s:%s to subprocess %s' % (starti, endi, np)
        subprocesses.append(p)
        starti += chunk
    for i in range(np):
        likelihood += que1.get()
    for i in range(np):
        grad += que2.get()
    while subprocesses:
        subprocesses.pop().join()
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)


'''simply use windows multiprocess, pickle everything'''
def likelihood_mp_simple(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    likelihood = numpy.dot(fss,theta)
    que1 = Queue() # for the likihood output
    que2 = Queue() # for the gradient output
    np = 0
    subprocesses = []
    corenum=multiprocessing.cpu_count()
    #corenum=1
    if corenum>1:
        chunk=seqnum/corenum+1
    else:
        chunk=seqnum
    starti=0
    while starti < (seqnum):
        endi=starti+chunk
        if endi>seqnum:
            endi=seqnum    
        p = Process(target=likelihoodthread_simple, 
            args=(seqlens[starti:endi],uon[starti:endi],bon[starti:endi],theta,K,ufnum,bfnum,que1,que2))
        p.start()
        np+=1
        #print 'delegated %s:%s to subprocess %s' % (starti, endi, np)
        subprocesses.append(p)
        starti += chunk
    for i in range(np):
        likelihood += que1.get()
    for i in range(np):
        grad += que2.get()
    while subprocesses:
        subprocesses.pop().join()
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)


'''using shared array to do the multiprocessing in windows'''
def likelihood_mp_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    likelihood = numpy.dot(fss,theta)
    que1 = Queue() # for the likihood output
    que2 = Queue() # for the gradient output
    np = 0
    subprocesses = []
    corenum=multiprocessing.cpu_count()
    #corenum=1
    if corenum>1:
        chunk=seqnum/corenum+1
    else:
        chunk=seqnum
    starti=0
    while starti < (seqnum):
        endi=starti+chunk
        if endi>seqnum:
            endi=seqnum    
        p = Process(target=likelihoodthread_sa, 
            args=(seqlens,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,theta,starti,endi,seqnum,K,ufnum,bfnum,que1,que2))
        p.start()
        np+=1
        #print 'delegated %s:%s to subprocess %s' % (starti, endi, np)
        subprocesses.append(p)
        starti += chunk
    for i in range(np):
        likelihood += que1.get()
    for i in range(np):
        grad += que2.get()
    while subprocesses:
        subprocesses.pop().join()
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)


def likelihood_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    likelihood = numpy.dot(fss,theta)
    for si in range(seqnum):
        logMlist = logM_sa(seqlens[si],si,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            iloc = uonseqsta[si]
            for it in range(uonlocsta[iloc+i],uonlocend[iloc+i]):            
                ao=uonarr[it]
                gradu[ao:ao+K] -= p_yi
                
            iloc = bonseqsta[si]    
            for it in range(bonlocsta[iloc+i],bonlocend[iloc+i]):            
                ao=bonarr[it]
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)


def likelihood(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    likelihood = numpy.dot(fss,theta)
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    for si in range(seqnum):
        #logMlist = logMarray(seqlens[si],si,uon,bon,K,thetau,thetab)
        logMlist = logMarray(seqlens[si],uon[si],bon[si],K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            for ao in uon[si][i]:
                gradu[ao:ao+K] -= p_yi
            for ao in bon[si][i]:
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)


def gradient_likelihood_standalone(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    grad = numpy.array(fss,copy=True)  # data distribuition
    #print grad
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    for si in range(seqnum):
        logMlist = logMarray(seqlens[si],uon[si],bon[si],K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        #likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            for ao in uon[si][i]:
                gradu[ao:ao+K] -= p_yi
            for ao in bon[si][i]:
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    return grad - regularity_deriv(theta,regtype,sigma)

def gradient_likelihood(theta):    # this is a dummy function
    global _gradient
    return _gradient


def checkCrfDev(datafile,tpltfile):
    '''Check if the Derivative calculation is correct.
    Don't call this function if your model has millions of features. 
    Otherwise it will run forever...      '''
    if not os.path.isfile(tpltfile):
        print "Can't find the template file!"
        return -1
    tplist=readTemplate(tpltfile)    
    
    if not os.path.isfile(datafile):
        print "Data file doesn't exist!"
        return -1
    texts,seqlens,oys,seqnum,K,obydic,y2label=readData(datafile)
    
    uobxs,bobxs,ufnum,bfnum=processFeatures(tplist,texts,seqnum,K)
    fnum=ufnum+bfnum
   
    uon,bon = calObservexOn(tplist,texts,uobxs,bobxs,seqnum)
    
    y0=0
    regtype=2 ;  sigma=1.0
    fss=calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0)
    print "Linear CRF in Python.. ver 0.1 "
    print "B features:",bfnum,"U features:",ufnum, "total num:",fnum
    print "training sequence number:",seqnum

    theta=random_param(ufnum,bfnum)
    #theta = multiprocessing.Array(ctypes.c_double, ufnum+bfnum)
    #theta = numpy.ctypeslib.as_array(theta.get_obj())
    #theta = theta.reshape(ufnum+bfnum)
    #manager=multiprocessing.Manager()
    #ns=manager.Namespace()
    #ns.uon=uon
    #ns.bon=bon
    #ns.theta=theta
    #assert ns.theta is theta
    delta=0.0001
    for i in range(fnum):
        ta=likelihood_mp_simple(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        #ta=likelihood(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        #ta=likelihood_mp(seqlens,fss,theta,seqnum,K,ufnum,bfnum,regtype,sigma,ns)
        dev=gradient_likelihood(theta)
        #dev=gradient_likelihood_standalone(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        #dev = grad
        theta[i]=theta[i]+delta
        #tb=likelihood(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        tb=likelihood_mp_simple(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        devest=(tb-ta)/delta
        print "dev:",dev[i],"dev numeric~:",devest, str(datetime.datetime.now())[10:19]
        theta[i]=theta[i]-delta  # reverse to original    


def checkCrfDev_sa(datafile,tpltfile,mp=0):
    '''Check if the Derivative calculation is correct.
    Don't call this function if your model has millions of features. 
    Otherwise it will run forever...      '''
    if not os.path.isfile(tpltfile):
        print "Can't find the template file!"
        return -1
    tplist=readTemplate(tpltfile)    
    
    if not os.path.isfile(datafile):
        print "Data file doesn't exist!"
        return -1
    texts,seqlens,oys,seqnum,K,obydic,y2label=readData(datafile)
    
    uobxs,bobxs,ufnum,bfnum=processFeatures(tplist,texts,seqnum,K)
    fnum=ufnum+bfnum
   
    uon,bon = calObservexOn(tplist,texts,uobxs,bobxs,seqnum)
    uonarr,uonseqsta,uonlocsta,uonlocend,bonarr,bonseqsta,bonlocsta,bonlocend = calObservexOnLoc(uon,bon,seqnum,mp)
    
    y0=0
    regtype=2 ;  sigma=1.0
    fss=calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0)
    print "Linear CRF in Python.. ver 0.1 "
    print "B features:",bfnum,"U features:",ufnum, "total num:",fnum
    print "training sequence number:",seqnum

    if sys.platform=="win32" and mp==1:
        theta = multiprocessing.Array(ctypes.c_double, ufnum+bfnum)
        theta = numpy.ctypeslib.as_array(theta.get_obj())
        theta = theta.reshape(ufnum+bfnum)
        
        #theta=random_param(ufnum,bfnum)
    else:
        theta=random_param(ufnum,bfnum)
    #theta = multiprocessing.Array(ctypes.c_double, ufnum+bfnum)
    #theta = numpy.ctypeslib.as_array(theta.get_obj())
    #theta = theta.reshape(ufnum+bfnum)
    #assert ns.theta is theta
    delta=0.0001
    for i in range(fnum):
        ta=likelihood_mp_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                      bonarr,bonseqsta,bonlocsta,bonlocend,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        dev=gradient_likelihood(theta)
        #dev=gradient_likelihood_standalone(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        #dev = grad
        theta[i]=theta[i]+delta
        #tb=likelihood(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        tb=likelihood_mp_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                      bonarr,bonseqsta,bonlocsta,bonlocend,theta,seqnum,K,ufnum,bfnum,regtype,sigma)
        devest=(tb-ta)/delta
        print "dev:",dev[i],"dev numeric~:",devest, str(datetime.datetime.now())[10:19]
        theta[i]=theta[i]-delta  # reverse to original    


def saveModel(bfnum,ufnum,tlist,obydic,uobxs,bobxs,theta,modelfile):
    import cPickle as pickle
    with open(modelfile, 'wb') as f:
        pickle.dump([bfnum,ufnum,tlist,obydic,uobxs,bobxs,theta], f)


def outputFile(texts,oys,maxys,y2label,resfile):
    if resfile=="":   #  don't need to write to file.
        return 0  
    fo = codecs.open(resfile, "w")
    for si in range(len(oys)):
        for li in range(len(oys[si])):
            strt=""
            for x in texts[si][li]:
                strt += x +" "
            strt += y2label[oys[si][li]]+" "
            strt += y2label[maxys[si][li]]
            strt += "\n"
            fo.write(strt)
        fo.write(" \n")
    fo.close()
    return 0

def loadModel(modelFile):
    import cPickle as pickle
    if not os.path.isfile(modelFile):
        print "Error: model file does not Exist!"
        return -1
    with open(modelFile, 'rb') as f:
        bfnum,ufnum,tlist,obydic,uobxs,bobxs,theta = pickle.load(f)
    K=len(obydic)
    y2label = dict([(obydic[key],key) for key in obydic.keys()])
    return bfnum,ufnum,tlist,obydic,uobxs,bobxs,theta,K,y2label

def tagging(seqlens,uon,bon,theta,seqnum,K,ufnum,bfnum):
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    maxys=[]
    for si in range(seqnum):
        logMlist = logMarray(seqlens[si],uon[si],bon[si], K, thetau,thetab)
        #logalphas = logAlphas(logMlist)
        #logZ = logsumexp(logalphas[-1])
        maxalpha=numpy.zeros((len(logMlist),K))
        my=[]  ;  maxilist=[]
        seqlen=len(logMlist)
        for i in range(seqlen):
            if i == 0:
                maxalpha[i] = logMlist[0][:,0]
                #print maxalpha[0]
            elif i < seqlen :
                at = logMlist[i]+maxalpha[i-1]
                maxalpha[i] = at.max(axis=1)
                #print maxalpha[i]
                maxilist.append(at.argmax(axis=1))
        ty=maxalpha[-1].argmax()
        my.append(ty)
        for a in (reversed(maxilist)):
            my.append(a[ty])
            ty=a[ty]
        maxys.append(my[::-1])
    return maxys

def checkTagging(maxys,oys):
    tc=0 ; te=0;
    #print maxys
    #print oy
    for si in range(len(oys)):
        for li in range(len(oys[si])):
            if oys[si][li]==maxys[si][li]:
                tc += 1
            else:
                te +=1
    print "Note: If Y is useless, correct rate is also useless."            
    print "correct:",tc,"error:",te," correct rate:",float(tc)/(tc+te)
        
def crfpredict(datafile,modelfile,resfile=""):
    import time 
    start_time = time.time()
    '''read all the data'''
    bfnum,ufnum,tplist,obydic,uobxs,bobxs,theta,K,y2label=loadModel(modelfile)
    fnum=ufnum+bfnum
    if fnum==0:
        print "ERROR: Load the model file failed!"
        return -1
    texts,seqlens,oys,seqnum,t1,obydictmp,y2ltmp=readData(datafile)
    if seqnum==0 or len(obydic)==0:
        print "ERROR: Read data file failed!"
        return -1
    # change the oys to be concist with model
    for i in range(len(oys)):
        for j in range(len(oys[i])):
            slabel=y2ltmp[oys[i][j]]
            if obydic.has_key(slabel):  # some
                oys[i][j] = obydic[y2ltmp[oys[i][j]]]
            else:
                oys[i][j] = 0
    
    print "Linear CRF in Python.. ver 0.1 "
    print "B features:",bfnum,"U features:",ufnum, "total num:",fnum
    print "Prediction sequence number:",seqnum
    uon,bon = calObservexOn(tplist,texts,uobxs,bobxs,seqnum)
    maxys = tagging(seqlens,uon,bon,theta,seqnum,K,ufnum,bfnum)
    checkTagging(maxys,oys)
    print "Write max(y) to file:",resfile
    outputFile(texts,oys,maxys,y2label,resfile)
    print "Test finished in ", time.time() - start_time, "seconds. \n "

def train(datafile,tpltfile,modelfile,mp=1,regtype=2,sigma=1.0,fd=1.0):
    import time 
    import cPickle as pickle

    start_time = time.time()
    if not os.path.isfile(tpltfile):
        print "Can't find the template file!"
        return -1
    tplist=readTemplate(tpltfile)    
    #print tplist
    if not os.path.isfile(datafile):
        print "Data file doesn't exist!"
        return -1
    texts,seqlens,oys,seqnum,K,obydic,y2label=readData(datafile)
    #print seqlens 
    
    uobxs,bobxs,ufnum,bfnum=processFeatures(tplist,texts,seqnum,K, fd=fd)
    fnum=ufnum+bfnum
    print "Linear CRF in Python.. ver 0.1 "
    print "B features:",bfnum,"U features:",ufnum, "total num:",fnum
    print "training sequence number:",seqnum
    print "start to calculate ON feature.  elapsed time:", time.time() - start_time, "seconds. \n "
    if fnum==0:
        print "No Parameters to Learn. "
        return
    uon,bon = calObservexOn(tplist,texts,uobxs,bobxs,seqnum)
    with open("ubobx", 'wb') as f:
        pickle.dump([uobxs,bobxs], f)
    del uobxs
    del bobxs
    
    print "start to calculate data distribuition. elapsed time:", time.time() - start_time, "seconds. \n "    
    y0=0
    fss=calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0)
    del texts    
    del oys

    print "start to translate ON Feature list to Array.  elapsed time:", time.time() - start_time, "seconds. \n "
    uonarr,uonseqsta,uonlocsta,uonlocend,bonarr,bonseqsta,bonlocsta,bonlocend = calObservexOnLoc(uon,bon,seqnum,mp)
    #print  "ubon size:", sys.getsizeof(uon)
    #print  "uobxs size:", sys.getsizeof(uobxs)
    #print  "texts size:", sys.getsizeof(texts)
    del uon
    del bon
		
    print "start to learn distribuition. elapsed time:", time.time() - start_time, "seconds. \n " 

    #return
    
    from scipy import optimize
    if sys.platform=="win32" and mp==1:  # using shared memory
        theta = multiprocessing.Array(ctypes.c_double, ufnum+bfnum)
        print "allocate theta OK. elapsed time:", time.time() - start_time, "seconds. \n "
        theta = numpy.ctypeslib.as_array(theta.get_obj())
        theta = theta.reshape(ufnum+bfnum)
    else:
        theta=random_param(ufnum,bfnum)
    
    if mp==1:  # using multi processing
        likeli = lambda x:-likelihood_mp_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                      bonarr,bonseqsta,bonlocsta,bonlocend,x,seqnum,K,ufnum,bfnum,regtype,sigma)
    else:
        likeli = lambda x:-likelihood_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                      bonarr,bonseqsta,bonlocsta,bonlocend,x,seqnum,K,ufnum,bfnum,regtype,sigma)
    likelihood_deriv = lambda x:-gradient_likelihood(x)
    theta,fobj,dtemp = optimize.fmin_l_bfgs_b(likeli,theta, 
            fprime=likelihood_deriv , disp=1, factr=1e12)

    with open("ubobx", 'rb') as f:
        uobxs,bobxs = pickle.load(f)
    saveModel(bfnum,ufnum,tplist,obydic,uobxs,bobxs,theta,modelfile)
    print "Training finished in ", time.time() - start_time, "seconds. \n "

def train_simple(datafile,tpltfile,modelfile,mp=1,regtype=2,sigma=1.0):
    import time 
    start_time = time.time()
    if not os.path.isfile(tpltfile):
        print "Can't find the template file!"
        return -1
    tplist=readTemplate(tpltfile)    
    #print tplist
    
    if not os.path.isfile(datafile):
        print "Data file doesn't exist!"
        return -1
    texts,seqlens,oys,seqnum,K,obydic,y2label=readData(datafile)
    #print seqlens 
    
    uobxs,bobxs,ufnum,bfnum=processFeatures(tplist,texts,seqnum,K)
    fnum=ufnum+bfnum
    print "Linear CRF in Python.. ver 0.1 "
    print "B features:",bfnum,"U features:",ufnum, "total num:",fnum
    print "training sequence number:",seqnum
    print "start to calculate ON feature.  ", time.time() - start_time, "seconds. \n "
    uon,bon = calObservexOn(tplist,texts,uobxs,bobxs,seqnum)

    #print  "ubon size:", sys.getsizeof(uon)
    #print  "uobxs size:", sys.getsizeof(uobxs)
    #print  "texts size:", sys.getsizeof(texts)
    print "start to calculate data distribuition. ", time.time() - start_time, "seconds. \n "    
#    del uobxs
#    del bobxs
    
    y0=0
    fss=calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0)
    del texts    
    del oys
    
#    import cPickle as pickle
#    with open("dump", 'wb') as f:
#        pickle.dump([uon], f)
        
    print "start to learn distribuition. ", time.time() - start_time, "seconds. \n " 

    #return
    
    from scipy import optimize
    if mp==1:  # using multi processing
        theta=random_param(ufnum,bfnum)
        likeli = lambda x:-likelihood_mp_simple(seqlens,fss,uon,bon,x,seqnum,K,ufnum,bfnum,regtype,sigma)
    else:
        theta=random_param(ufnum,bfnum)
        likeli = lambda x:-likelihood(seqlens,fss,uon,bon,x,seqnum,K,ufnum,bfnum,regtype,sigma)
    likelihood_deriv = lambda x:-gradient_likelihood(x)
    theta,fobj,dtemp = optimize.fmin_l_bfgs_b(likeli,theta, 
            fprime=likelihood_deriv , disp=1, factr=1e12)

    saveModel(bfnum,ufnum,tplist,obydic,uobxs,bobxs,theta,modelfile)
    print "Training finished in ", time.time() - start_time, "seconds. \n "

def main():
    #checkCrfDev("train.txt","template.txt") 
    #checkCrfDev("train2.txt","template.txt") 
    #checkCrfDev_sa("..\\train1.txt","..\\template.txt",mp=1) 
    #checkCrfDev("trainexample2.txt","template.txt")
    #checkCrfDev_sa("trainsimple.data","templatesimple.txt",mp=1)
    #train("..\\train1.txt","..\\template.txt","model",mp=0,fd=1)
    #train("train1.txt","template.txt","model",mp=1)
    #train("datas\\4.msr_training.data","templatechunk","model")
    #train("..\\train2.txt","..\\template.txt","model",mp=1,fd=2)
    #train("datas\\train.txt","templatechunk","model")
    train("trainsimple.data","templatesimple.txt","model",mp=0)
    #train("tr1.utf8.txt","templatesimple.txt","model")
    
    #crfpredict("train2.txt","model","res.txt")
    #crfpredict("tr1.utf8.txt","model","res.txt")
            
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()