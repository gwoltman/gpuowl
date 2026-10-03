# Exact-layout model of the MIDDLE=3 NTT done as a Good-Thomas (prime-factor) transform, see src/cl/base.cl.
# Mirrors the kernels over GF(M61^2): buffer (x, line) layout, width -> fftMiddleIn -> height, the tail pairing with the
# real onePairSq formula and SWAP_XY outputs, and the inverse run as forward transforms on SWAP_XY'd data.
# Checks the result against a plain cyclic convolution and reports the output scale.  Usage: pfa_model.py WIDTH SMALL_HEIGHT
import random, sys
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
conj=lambda a:(a[0],(-a[1])%P)
swap=lambda a:(a[1],a[0])
H0=(264036120304204,4677669021635377)          # PRPLL GF61 generator, order 2^62
root=lambda n: pw(H0,(1<<62)//n)                # power-of-two roots, as GF61::root_one
J=(pow(37,(P-1)//3,P),0)                        # cube root of unity in Z_p
Ji=mul(J,J)
INV3=pow(3,-1,P)

W,SH=int(sys.argv[1]) if len(sys.argv)>1 else 8, int(sys.argv[2]) if len(sys.argv)>2 else 4
L=W*SH; ND=3*L; NW=2*ND
v=root(L); wW=root(W); wH=root(SH)
def crt(r,q): return q + L*(((r-q)%3)*(L%3)%3)            # CRT: ==r mod 3, ==q mod L
def pair_of(x,g): return crt(g%3, x*SH + g%SH)              # logical pair stored at (x, line g)
def dft(seq,w): n=len(seq); return [ (lambda k: __import__('functools').reduce(add,[mul(seq[j],pw(w,j*k)) for j in range(n)],(0,0)))(k) for k in range(n)]

def forward(z):            # z: logical pairs -> tail lines dict[(kx,k3)] = list over ky
    buf=[[z[pair_of(x,g)] for x in range(W)] for g in range(3*SH)]
    buf=[dft(row,wW) for row in buf]                      # width FFT per line
    tail={}
    for kx in range(W):
        cols={}
        for y in range(SH):
            u=[buf[m*SH+y][kx] for m in range(3)]         # readMiddleInLine: u[m] = line m*SH+y
            u=[mul(t,pw(v,kx*y)) for t in u]              # binary twiddle, same for all m
            A=[None]*3
            for m in range(3): A[(m*SH+y)%3]=u[m]         # rows
            for k3 in range(3):
                s=(0,0)
                for r in range(3): s=add(s,mul(A[r],pw(J,r*k3)))
                cols[(k3,y)]=s
        for k3 in range(3):
            tail[(kx,k3)]=dft([cols[(k3,y)] for y in range(SH)],wH)   # height FFT: index ky
    return tail

def inverse_from_swapped(tail):  # tail values are SWAP_XY'd; returns logical pairs (still swapped-domain)
    cols={}
    for (kx,k3),line in tail.items():
        h=dft(line,wH)                                    # forward height FFT on swapped data
        for y in range(SH): cols[(kx,k3,y)]=h[y]
    buf=[[None]*W for _ in range(3*SH)]
    for kx in range(W):
        for y in range(SH):
            Y=[cols[(kx,k3,y)] for k3 in range(3)]
            A=[]
            for r in range(3):
                s=(0,0)
                for k3 in range(3): s=add(s,mul(Y[k3],pw(Ji,r*k3)))   # inverse radix-3 uses J^-1
                A.append(s)
            tw=smul(pw(v,kx*y),INV3)                       # same forward twiddle, 1/3 folded in
            for m in range(3): buf[m*SH+y][kx]=mul(A[(m*SH+y)%3],tw)
    buf=[dft(row,wW) for row in buf]                      # forward width FFT on swapped data
    z=[None]*ND
    for g in range(3*SH):
        for x in range(W): z[pair_of(x,g)]=buf[g][x]
    return z

def onePairSq(a,b,t2):
    a1=add(a,conj(b)); b1=sub(a,conj(b))
    c=sub(mul(a1,a1),mul(mul(b1,b1),t2)); d=smul(mul(a1,b1),2)
    cn=add(c,d); dn=conj(sub(c,d))
    return swap(cn),swap(dn)

random.seed(1)
x=[random.randrange(1000) for _ in range(NW)]
z=[(x[2*p],x[2*p+1]) for p in range(ND)]
T=forward(z)
# check against true DFT with omega = J*v, frequency K = CRT(k3, K2), K2 = kx + W*ky
om=mul(J,v)
ok=all(T[(kx,k3)][ky]==__import__('functools').reduce(add,[mul(z[p],pw(om,p*crt(k3,kx+W*ky))) for p in range(ND)],(0,0))
       for kx in range(W) for k3 in range(3) for ky in range(SH))
print("forward == DFT(omega=J*v) at K=CRT(k3,kx+W*ky):",ok)
# tail: pair (kx,k3,ky) with partner (-K2) in same k3 block; t^2 = J^k3 * v^K2
out={k:[None]*SH for k in T}
for (kx,k3),line in T.items():
    for ky in range(SH):
        K2=kx+W*ky; K2p=(-K2)%L; kxp,kyp=K2p%W,K2p//W
        a=line[ky]; b=T[(kxp,k3)][kyp]
        t2=mul(pw(J,k3),pw(v,K2))
        pa,pb=onePairSq(a,b,t2)
        out[(kx,k3)][ky]=pa
        # consistency: computing from the partner's side must give pb for this element
        pa2,pb2=onePairSq(b,a,mul(pw(J,k3),pw(v,K2p)))
        assert pb2==pa, "partner symmetry"
zq=inverse_from_swapped(out)
y=[]
for p in range(ND): y+= [zq[p][1], zq[p][0]]                # carry does SWAP_XY
ref=[sum(x[i]*x[(n-i)%NW] for i in range(NW))%P for n in range(NW)]
scale=y[0]*pow(ref[0],-1,P)%P
print("output == cyclic convolution * scale:", all(y[n]==ref[n]*scale%P for n in range(NW)))
print("scale = 2^k?", [k for k in range(70) if pow(2,k,P)==scale], " expected log2(2*NWORDS/3) =", (2*NW//3).bit_length()-1)
