# The FP side of a PFA hybrid FFT/NTT (see base.cl): the same exact-layout model as pfa_model.py, over the complex numbers.  The
# radix-R root w = e^(-2*pi*i/R) is complex, so the inverse radix-R is the forward one on SWAP_XY'd data (no output reversal),
# and the tail pairs row k3 with row R - k3: line kx + TW*k3 with (TW - kx) % TW + TW*((R - k3) % R), t^2 = w^k3 * v^(kx + TW*ky).
# Checks the result against a plain cyclic convolution and reports the output scale (2*NWORDS/R, removed by fftMiddleOut's factor).
# Usage: pfa_model_fp.py R WIDTH M2 SMALL_HEIGHT   (tiny sizes, e.g. 7 4 2 4)
import random, sys, functools, cmath
def add(a,b): return a+b
def sub(a,b): return a-b
def mul(a,b): return a*b
def smul(a,s): return a*s
def pw(a,e): return a**e if e>=0 else (1/a)**(-e)
conj=lambda a:a.conjugate(); swap=lambda a:complex(a.imag,a.real)
root=lambda n: cmath.exp(-2j*cmath.pi/n)
R,W,M2,SH=[int(a) for a in sys.argv[1:5]]
J=root(R); wr=J; INV3=1.0/R
C=[(J**m+J**(-m))/2 for m in range(R)]
S=[(J**m-J**(-m))/2 for m in range(R)]
WP=[J**m for m in range(R)]
BBH=M2*SH; L=W*BBH; ND=R*L; NW=2*ND; TW=W*M2
v=root(L); wW=root(W); wH=root(SH); wM=root(M2) if M2>1 else (1,0)
LINV=pow(L%R,-1,R)
def crt(r,q): return q + L*(((r-q)%R)*LINV%R)
def pair_of(x,g): return crt(g%R, x*BBH + g%BBH)
def dft(seq,w): return [functools.reduce(add,[mul(seq[j],pw(w,j*k)) for j in range(len(seq))],0j) for k in range(len(seq))]
def rows_of(b, t, s):              # b[j] is row (t + j*s) % R: permute (compile time) then rotate right by t (barrel)
    d=[None]*R
    for j in range(R): d[(j*s)%R]=b[j]
    bit=1
    while bit<R:
        if t & bit: d=[d[(i-bit)%R] for i in range(R)]
        bit<<=1
    return d
def unrows(a, t, s):
    e=list(a); bit=1
    while bit<R:
        if t & bit: e=[e[(i+bit)%R] for i in range(R)]
        bit<<=1
    return [e[(j*s)%R] for j in range(R)]
def dft3k(a,w):   # kernel formula: y1 = a0 - a2 + w(a1-a2), y2 = a0 - a1 - w(a1-a2)
    wd=smul(sub(a[1],a[2]),w)
    return [add(add(a[0],a[1]),a[2]), add(sub(a[0],a[2]),wd), sub(sub(a[0],a[1]),wd)]
def dftR(a):     # forward DFT with root wr, as the kernels compute it
    if R==3: return dft3k(a,wr)
    if R==9:
        B=[dft3k([a[n1],a[n1+3],a[n1+6]],WP[3]) for n1 in range(3)]          # over n2, output k2
        tw={(1,1):1,(1,2):2,(2,1):2,(2,2):4}
        for (n1,k2),e in tw.items(): B[n1][k2]=smul(B[n1][k2],WP[e])
        Y=[None]*9
        for k2 in range(3):
            o=dft3k([B[0][k2],B[1][k2],B[2][k2]],WP[3])
            for k1 in range(3): Y[k2+3*k1]=o[k1]
        return Y
    h=(R-1)//2
    Sk=[add(a[k],a[R-k]) for k in range(1,h+1)]; Dk=[sub(a[k],a[R-k]) for k in range(1,h+1)]
    y=[None]*R; y[0]=a[0]
    for x in Sk: y[0]=add(y[0],x)
    for j in range(1,h+1):
        Pj=0j; Qj=0j
        for k in range(1,h+1):
            Pj=add(Pj,smul(Sk[k-1],C[j*k%R])); Qj=add(Qj,smul(Dk[k-1],S[j*k%R]))
        y[j]=add(add(a[0],Pj),Qj); y[R-j]=sub(add(a[0],Pj),Qj)
    return y
def idftR(a): return dftR(a)
s=(M2*SH)%R
def middle_in(u,x,y):                       # u[m], m in [0,3*M2): line m*SH+y
    u=[mul(u[m],pw(v,x*(y+SH*(m%M2)))) for m in range(R*M2)]          # w_L^(x*b)
    out=[None]*(R*M2)
    Y={}
    for m2 in range(M2):
        t=(m2*SH+y)%R
        A=rows_of([u[m2+M2*j] for j in range(R)],t,s)
        Yk=dftR(A)
        for k3 in range(R): Y[(k3,m2)]=Yk[k3]
    for k3 in range(R):
        b=dft([Y[(k3,m2)] for m2 in range(M2)],wM) if M2>1 else [Y[(k3,0)]]
        for km in range(M2): out[km+M2*k3]=mul(b[km],pw(root(BBH),y*km))   # middleMul w_BBH^(y*km)
    return out
def middle_out(u,x,y):                      # swapped-domain inverse; u[i], i = km + M2*k3
    Y={}
    for k3 in range(R):
        a=[mul(u[km+M2*k3],pw(root(BBH),y*km)) for km in range(M2)]
        b=dft(a,wM) if M2>1 else a
        for m2 in range(M2): Y[(k3,m2)]=b[m2]
    out=[None]*(R*M2)
    for m2 in range(M2):
        t=(m2*SH+y)%R
        A=idftR([Y[(k3,m2)] for k3 in range(R)])
        vals=unrows(A,t,s)
        for j in range(R): out[m2+M2*j]=vals[j]
    return [smul(mul(out[m],pw(v,x*(y+SH*(m%M2)))),INV3) for m in range(R*M2)]
def forward(z):
    buf=[[z[pair_of(x,g)] for x in range(W)] for g in range(R*BBH)]
    buf=[dft(r,wW) for r in buf]
    tail={}
    cols={}
    for kx in range(W):
        for y in range(SH):
            o=middle_in([buf[m*SH+y][kx] for m in range(R*M2)],kx,y)
            for i in range(R*M2): cols[(kx+W*i,y)]=o[i]
    for line in range(W*R*M2): tail[line]=dft([cols[(line,y)] for y in range(SH)],wH)
    return tail
def inverse(tail):
    cols={}
    for line,vals in tail.items():
        h=dft(vals,wH)
        for y in range(SH): cols[(line,y)]=h[y]
    buf=[[None]*W for _ in range(R*BBH)]
    for kx in range(W):
        for y in range(SH):
            o=middle_out([cols[(kx+W*i,y)] for i in range(R*M2)],kx,y)
            for m in range(R*M2): buf[m*SH+y][kx]=o[m]
    buf=[dft(r,wW) for r in buf]
    z=[None]*ND
    for g in range(R*BBH):
        for x in range(W): z[pair_of(x,g)]=buf[g][x]
    return z
def onePairSq(a,b,t2):
    a1=add(a,conj(b)); b1=sub(a,conj(b))
    c=sub(mul(a1,a1),mul(mul(b1,b1),t2)); d=smul(mul(a1,b1),2)
    return swap(add(c,d)),swap(conj(sub(c,d)))
random.seed(2)
x=[random.randrange(1000) for _ in range(NW)]
z=[complex(x[2*p],x[2*p+1]) for p in range(ND)]
T=forward(z)
om=mul(J,v)
def K2of(line,ky): return line%TW + TW*ky
ok=all(abs(T[line][ky]-sum(z[p]*om**(p*crt(line//TW,K2of(line,ky))) for p in range(ND)))<1e-5 for line in T for ky in range(SH))
print("forward == DFT at K=CRT(k3, line%TW + TW*ky), k3 = line//TW:",ok)
out={}
for line,vals in T.items():
    k3=line//TW; out[line]=[None]*SH
    for ky in range(SH):
        K2=K2of(line,ky); K2p=(-K2)%L
        pline=((R-k3)%R)*TW + K2p%TW; pky=K2p//TW
        out[line][ky]=onePairSq(vals[ky],T[pline][pky],mul(pw(J,k3),pw(v,K2)))[0]
zq=inverse(out)
y=[]
for p in range(ND): y+=[zq[p].imag,zq[p].real]
ref=[sum(x[i]*x[(n-i)%NW] for i in range(NW)) for n in range(NW)]
sc=y[0]/ref[0]
print("conv ok:",all(abs(y[n]-ref[n]*sc)<1e-6*abs(sc)*max(ref) for n in range(NW)),"scale",sc,"expect",2*NW//R)
