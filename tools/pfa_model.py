# Exact-layout model of the MIDDLE = 3*M2 (M2 = 1, 2, 4) NTT done as a Good-Thomas (prime-factor) transform, see src/cl/base.cl.
# Mirrors the kernels over GF(M61^2): buffer (x, line) layout, width -> fftMiddleIn -> height, the tail pairing with the
# real onePairSq formula and SWAP_XY outputs, and the inverse run as forward transforms on SWAP_XY'd data.  Checks the
# result against a plain cyclic convolution, reports the output scale and the lines whose carries are rotated.
# Usage: pfa_model.py WIDTH M2 SMALL_HEIGHT   (tiny sizes, e.g. 8 2 4)
import random, sys, functools
P=(1<<61)-1
def add(a,b): return ((a[0]+b[0])%P,(a[1]+b[1])%P)
def sub(a,b): return ((a[0]-b[0])%P,(a[1]-b[1])%P)
def mul(a,b): return ((a[0]*b[0]-a[1]*b[1])%P,(a[0]*b[1]+a[1]*b[0])%P)
def smul(a,s): return (a[0]*s%P,a[1]*s%P)
def pw(a,e):
    r=(1,0)
    while e:
        if e&1: r=mul(r,a)
        a=mul(a,a); e>>=1
    return r
conj=lambda a:(a[0],(-a[1])%P); swap=lambda a:(a[1],a[0])
H0=(264036120304204,4677669021635377); root=lambda n: pw(H0,(1<<62)//n)
J=(pow(37,(P-1)//3,P),0); Ji=mul(J,J); INV3=pow(3,-1,P)
W,M2,SH=[int(a) for a in sys.argv[1:4]]
BBH=M2*SH; L=W*BBH; ND=3*L; NW=2*ND; TW=W*M2
v=root(L); wW=root(W); wH=root(SH); wM=root(M2) if M2>1 else (1,0)
def crt(r,q): return q + L*(((r-q)%3)*(L%3)%3)
def pair_of(x,g): return crt(g%3, x*BBH + g%BBH)
def dft(seq,w): return [functools.reduce(add,[mul(seq[j],pw(w,j*k)) for j in range(len(seq))],(0,0)) for k in range(len(seq))]
def rows_of(vals3, t, s):          # vals3[j] is row (t + j*s) % 3; return in row order
    A=[None]*3
    for j in range(3): A[(t+j*s)%3]=vals3[j]
    return A
def unrows(A, t, s): return [A[(t+j*s)%3] for j in range(3)]
def dft3(a,w): return [add(add(a[0],mul(a[1],pw(w,k))),mul(a[2],pw(w,2*k))) for k in range(3)]
s=(M2*SH)%3
def middle_in(u,x,y):                       # u[m], m in [0,3*M2): line m*SH+y
    u=[mul(u[m],pw(v,x*(y+SH*(m%M2)))) for m in range(3*M2)]          # w_L^(x*b)
    out=[None]*(3*M2)
    Y={}
    for m2 in range(M2):
        t=(m2*SH+y)%3
        A=rows_of([u[m2+M2*j] for j in range(3)],t,s)
        Yk=dft3(A,J)
        for k3 in range(3): Y[(k3,m2)]=Yk[k3]
    for k3 in range(3):
        b=dft([Y[(k3,m2)] for m2 in range(M2)],wM) if M2>1 else [Y[(k3,0)]]
        for km in range(M2): out[km+M2*k3]=mul(b[km],pw(root(BBH),y*km))   # middleMul w_BBH^(y*km)
    return out
def middle_out(u,x,y):                      # swapped-domain inverse; u[i], i = km + M2*k3
    Y={}
    for k3 in range(3):
        a=[mul(u[km+M2*k3],pw(root(BBH),y*km)) for km in range(M2)]
        b=dft(a,wM) if M2>1 else a
        for m2 in range(M2): Y[(k3,m2)]=b[m2]
    out=[None]*(3*M2)
    for m2 in range(M2):
        t=(m2*SH+y)%3
        A=dft3([Y[(k3,m2)] for k3 in range(3)],Ji)
        vals=unrows(A,t,s)
        for j in range(3): out[m2+M2*j]=vals[j]
    return [smul(mul(out[m],pw(v,x*(y+SH*(m%M2)))),INV3) for m in range(3*M2)]
def forward(z):
    buf=[[z[pair_of(x,g)] for x in range(W)] for g in range(3*BBH)]
    buf=[dft(r,wW) for r in buf]
    tail={}
    cols={}
    for kx in range(W):
        for y in range(SH):
            o=middle_in([buf[m*SH+y][kx] for m in range(3*M2)],kx,y)
            for i in range(3*M2): cols[(kx+W*i,y)]=o[i]
    for line in range(W*3*M2): tail[line]=dft([cols[(line,y)] for y in range(SH)],wH)
    return tail
def inverse(tail):
    cols={}
    for line,vals in tail.items():
        h=dft(vals,wH)
        for y in range(SH): cols[(line,y)]=h[y]
    buf=[[None]*W for _ in range(3*BBH)]
    for kx in range(W):
        for y in range(SH):
            o=middle_out([cols[(kx+W*i,y)] for i in range(3*M2)],kx,y)
            for m in range(3*M2): buf[m*SH+y][kx]=o[m]
    buf=[dft(r,wW) for r in buf]
    z=[None]*ND
    for g in range(3*BBH):
        for x in range(W): z[pair_of(x,g)]=buf[g][x]
    return z
def onePairSq(a,b,t2):
    a1=add(a,conj(b)); b1=sub(a,conj(b))
    c=sub(mul(a1,a1),mul(mul(b1,b1),t2)); d=smul(mul(a1,b1),2)
    return swap(add(c,d)),swap(conj(sub(c,d)))
random.seed(2)
x=[random.randrange(1000) for _ in range(NW)]
z=[(x[2*p],x[2*p+1]) for p in range(ND)]
T=forward(z)
om=mul(J,v)
def K2of(line,ky): return line%TW + TW*ky
ok=all(T[line][ky]==functools.reduce(add,[mul(z[p],pw(om,p*crt(line//TW,K2of(line,ky)))) for p in range(ND)],(0,0)) for line in T for ky in range(SH))
print("forward == DFT at K=CRT(k3, line%TW + TW*ky), k3 = line//TW:",ok)
out={}
for line,vals in T.items():
    k3=line//TW; out[line]=[None]*SH
    for ky in range(SH):
        K2=K2of(line,ky); K2p=(-K2)%L
        pline=k3*TW + K2p%TW; pky=K2p//TW
        out[line][ky]=onePairSq(vals[ky],T[pline][pky],mul(pw(J,k3),pw(v,K2)))[0]
zq=inverse(out)
y=[]
for p in range(ND): y+=[zq[p][1],zq[p][0]]
ref=[sum(x[i]*x[(n-i)%NW] for i in range(NW))%P for n in range(NW)]
sc=y[0]*pow(ref[0],-1,P)%P
print("conv ok:",all(y[n]==ref[n]*sc%P for n in range(NW)),"scale 2^",[k for k in range(62) if pow(2,k,P)==sc],"expect",(2*NW//3).bit_length()-1)
# carry chain
rot=set()
loc={}
for g in range(3*BBH):
    for xx in range(W): loc[pair_of(xx,g)]=(xx,g)
bad=0
for p in range(ND):
    (x0,g0),(x1,g1)=loc[p],loc[(p+1)%ND]
    if g1!=(g0+1)%(3*BBH): bad+=1
    if x1!=x0: rot.add((g1,(x1-x0)%W))
print("carry line g->g+1:",bad==0,"rotated lines:",sorted(rot))
