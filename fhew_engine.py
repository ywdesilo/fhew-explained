# fhew_engine.py
import torch
import math

class fhew_engine:
    def __init__(self, device=None):
        self.stddev = 3.2
        self.logQ = 27
        self.logQks = 14

        self.N = 1 << 10
        self.Nh = self.N >> 1
        self.Q = 1 << self.logQ
        self.Qks = 1 << self.logQks

        self.n = 571
        self.q = 2*self.N

        self.d = 2
        self.logB = 9

        assert self.d * self.logB <= self.logQ

        self.dks = 2
        self.logBks = 7

        self.Bks = 1 << self.logBks

        assert self.dks * self.logBks <= self.logQks

        # Set devices first.
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device {self.device}")

        self.f_nand = torch.zeros([self.N], dtype=torch.int32, device=self.device).to(self.device)

        for i in range(self.N):
            self.f_nand[i] = self.nand_map(-i)

        self.root_powers = torch.arange(self.Nh, device=self.device).to(torch.complex128)
        self.root_powers = torch.exp((1j * math.pi / self.N)*self.root_powers).to(self.device)

        self.root_powers_inv = torch.arange(0, -self.Nh ,-1).to(torch.complex128)
        self.root_powers_inv = torch.exp((1j * math.pi / self.N)*self.root_powers_inv).to(self.device)

        # parameters for decomposition
        # we will use B-ary decomposition, i.e., cut digits by logB bits

        self.decomp_shift = self.logQ - self.logB * torch.arange(self.d,0,-1).view(self.d,1).to(self.device)
        self.mask = (1 << self.logB) - 1

        self.gvector = 1 << self.decomp_shift

        # for signed decomposition
        self.msbmask = 0
        for i in self.decomp_shift:
            self.msbmask += (1<<(i+self.logB-1))

        # parameters for LWE key switching decomposition
        self.decomp_shift_ks = self.logQks - self.logBks * torch.arange(self.dks,0,-1).to(torch.int32).view(self.dks,1).to(self.device)
        self.mask_ks = torch.tensor([(1 << self.logBks) - 1]).to(torch.int32).to(self.device)

        self.gvector_ks = 1 << self.decomp_shift_ks
        self.decomp_shift_ks = self.decomp_shift_ks.view(self.dks, 1)

        self.alphapoly = self.precompute_alpha()
        self.betapoly = self.precompute_beta()

        self.brk = None
        self.ksk = None

    def create_secret_key(self):
        return self.keygen(self.n)

    """
    @sk: secret key
    returns brk and ksk as tuple
    """
    def create_bootstrap_key(self, sk):
        skN = self.keygen(self.N)

        skNfft = self.negacyclic_fft(skN)
        brk = self.brkgen(sk, skNfft)

        ksk = self.LWEkskGen(sk, skN)
    
        return brk, ksk

    def keygen(self, dim):
        return torch.randint(2, size=(dim,), dtype=torch.int32, device=self.device)

    def errgen(self, dim):
        e = torch.round(self.stddev * torch.randn(dim, device=self.device))
        e = e.squeeze()
        return e.to(torch.int)

    def uniform(self, dim, modulus):
        return torch.randint(modulus, size=(dim,), dtype=torch.int32).to(self.device)

    def negacyclic_fft(self, a):
        acomplex = a.to(torch.complex128).to(self.device)

        a_precomp = (acomplex[..., :self.Nh] + 1j * acomplex[..., self.Nh:]) * self.root_powers

        return torch.fft.fft(a_precomp)

    def negacyclic_ifft(self, A):
        b = torch.fft.ifft(A)
        b *= self.root_powers_inv

        a = torch.cat((b.real, b.imag), dim=-1).to(self.device)
        # Rounding should be more accurate
        # a += 0.5

        aint = a.to(torch.int64).to(torch.int32)
        # only when Q is a power-of-two
        aint &= self.Q - 1

        return aint

    # make an RLWE encryption of message
    def encrypt_to_fft(self, m, sfft):
        ct = torch.stack([self.errgen(self.N), self.uniform(self.N, self.Q)]).to(self.device)
        ctfft = self.negacyclic_fft(ct)

        ctfft[0] += -ctfft[1] * sfft + self.negacyclic_fft(m)

        return ctfft

    def encrypt_bit_to_fft(self, b, sfft):
        ct = torch.stack([self.errgen(self.N), self.uniform(self.N, self.Q)]).to(self.device)
        ctfft = self.negacyclic_fft(ct)

        ctfft[0] += -ctfft[1] * sfft
        ctfft[0] += b

        return ctfft

    def normalize(self, v, logQ):
        # same as follows but no branch
        """
        if v > Q//2:
            v -= Q
        """
        # vmod Q when Q is a power-of-two
        Q = (1 << logQ)
        v &= (Q-1)
        # get msb
        msb = (v & Q//2) >> (logQ - 1)
        v -= (Q) * msb
        return v

    def decrypt_from_fft(self, ctfft, sfft):
        assert len(ctfft.size()) == 2
        # normalization is optional
        return self.normalize(self.negacyclic_ifft(ctfft[0] + ctfft[1]*sfft), self.logQ)
        # return negacyclic_ifft(ctfft[0] + ctfft[1]*sfft, N, Q)

    def decompose(self, a):
        # currently RGSW is only supported
        return (a.unsqueeze(0) >> self.decomp_shift.view(self.d, 1, 1)) & self.mask
        """
        # following code handles both RLWE' and RGSW
        assert len(a.size()) <= 2
        # for RLWE'
        if len(a.size()) == 1:
            res = (a.unsqueeze(0) >> decomp_shift.view(d, 1)) & mask
            return res
        # for RGSW
        elif len(a.size()) == 2:
            res = (a.unsqueeze(0) >> decomp_shift.view(d, 1, 1)) & mask
            return res
        """
        
    # about twice heavier than unsigned decomposition
    # it returns value -B/2 <= * <= B/2, not < B/2, but okay
    def signed_decompose(self, a):
        # carry
        da = self.decompose(a + (a & self.msbmask))
        # -B
        da -= self.decompose((a & self.msbmask))
        return da

    def encrypt_rgsw_fft(self, z, skfft):
        # RGSW has a dimension of d, 2, 2, N
        rgsw = torch.zeros(self.d, 2, 2, self.N, dtype=torch.int32, device=self.device)

        # generate the 'a' part
        # INSECURE: to be fixed later
        rgsw[:, :, 1, :] = torch.randint(self.Q, size = (self.d, 2 , self.N), dtype= torch.int32).to(self.device)

        # add error on b
        # INSECURE: to be fixed later
        rgsw[:, :, 0, :] = torch.round(self.stddev * torch.randn(size = (self.d, 2, self.N))).to(self.device)

        # following is equal to rgsw %= Q, but a faster version
        rgsw = self.normalize(rgsw, self.logQ)

        # do fft for easy a*s
        rgswfft = self.negacyclic_fft(rgsw)

        # now b = -a*sk + e
        rgswfft[:, :, 0, :] -= rgswfft[:, :, 1, :] * skfft.view(1, 1, self.Nh)

        # encrypt (z, z*sk) multiplied by g
        gzfft = self.negacyclic_fft(self.gvector * z)
        rgswfft[:, 0, 0, :] += gzfft
        rgswfft[:, 1, 1, :] += gzfft

        return rgswfft
    
    def encrypt_bit_to_rgsw_fft(self, b, skfft):
        # RGSW has a dimension of d, 2, 2, N
        rgsw = torch.zeros(self.d, 2, 2, self.N, dtype=torch.int32).to(self.device)

        # generate the 'a' part
        # INSECURE: to be fixed later
        rgsw[:, :, 1, :] = torch.randint(self.Q, size = (self.d, 2 , self.N), dtype= torch.int32)

        # add error on b
        # INSECURE: to be fixed later
        rgsw[:, :, 0, :] = torch.round(self.stddev * torch.randn(size = (self.d, 2, self.N)))

        # following is equal to rgsw %= Q, but a faster version
        rgsw = self.normalize(rgsw, self.logQ)

        # do fft for easy a*s
        rgswfft = self.negacyclic_fft(rgsw)

        # now b = -a*sk + e
        rgswfft[:, :, 0, :] -= rgswfft[:, :, 1, :] * skfft.view(1, 1, self.Nh)

        gb = self.gvector * b
        rgswfft[:, 0, 0, :] += gb
        rgswfft[:, 1, 1, :] += gb

        return rgswfft

    def rgswmult(self, ctfft, rgswfft):
        ct = self.negacyclic_ifft(ctfft)
        dct = self.signed_decompose(ct)
        multfft = self.negacyclic_fft(dct).view(self.d, 2, 1, self.Nh) * rgswfft
        
        return torch.sum(multfft, dim = (0,1))
    
    def encryptLWE(self, message, sk):
        ct = self.uniform(self.n + 1, self.q)

        ct[0] = 0

        ct[0] = message * self.q//4 - torch.sum(ct[1:] * sk)
        ct[0] += self.errgen(1)
        ct &= (self.q -1)

        return ct

    def decryptLWE(self, ct, sk, ksk=None):
        if len(ct) > self.n + 1:
            if self.ksk == None and ksk == None:
                print("Cannot key switch to small key. Provide switching key.")
            ct_ms = (ct * (self.Qks/self.Q)).to(torch.int32)
            ct_ks = self.LWEkeySwitch(ct_ms, self.ksk)
            ct_n = (ct_ks * (self.q/self.Qks)).to(torch.int32)
        else:
            ct_n = ct

        m_dec = ct_n[0] + torch.sum(ct_n[1:] * sk)
        m_dec %= self.q

        m_dec = m_dec.to(torch.float)
        m_dec /= self.q/4.
        m_dec = torch.round(m_dec)

        return m_dec.to(torch.int)%4

    def extract(self, ctRLWE):
        beta = ctRLWE[0][0]

        alpha = ctRLWE[1][:]
        alpha[1:] = -alpha[1:].flip(dims = [0])

        return torch.cat((beta.unsqueeze(0), alpha))

    def decompose_ks(self, a):
        
        assert len(a.size()) == 1

        res = (a.unsqueeze(0) >> self.decomp_shift_ks) & self.mask_ks
        return res

    # size: (self.dks, Bks, N, n+1)
    def LWEkskGen(self, sk, skN):    
        ksk = torch.randint(self.Qks, size = (self.dks, self.Bks, self.N, self.n+1), dtype=torch.int32).to(self.device)
        # b <- e
        ksk[..., 0] = torch.round(self.stddev * torch.randn(size = (self.dks, self.Bks, self.N))).to(torch.int32)
        # b <- e - a * s
        ksk[..., 0] -= torch.sum(ksk[:,:,:,1:] * sk, dim = -1)
        # b <- e - a * s + j B^r skN_i
        ksk[..., 0] += (self.gvector_ks * torch.arange(self.Bks, device=self.device)).view(self.dks, self.Bks, 1) * skN
        ksk &= (self.Qks - 1)

        return ksk

    def LWEkeySwitch(self, ctLWE, kskLWE):
        # do decomposition
        alpha = ctLWE[1:]
        dalpha = self.decompose_ks(alpha)
        # do appropriate addition of keys
        switched = torch.zeros(self.n+1, dtype=torch.int32).to(self.device)
        switched[0] = ctLWE[0]

        for r in range(self.dks):
            for i in range(self.N):
                switched += kskLWE[r, dalpha[r, i], i]

        switched &= (self.Qks-1)

        return switched

    def brkgen(self, sk, skNfft):
        zero_poly = torch.zeros([self.N], dtype=torch.int32).to(self.device)

        one_poly = torch.zeros([self.N], dtype=torch.int32).to(self.device)
        one_poly[0] = 1

        brk = torch.zeros(self.n, self.d, 2, 2, self.Nh, dtype=torch.complex128, device=self.device)
        for i in range(self.n):
            if sk[i] == 0:
                brk[i] = self.encrypt_rgsw_fft(zero_poly, skNfft)
            else:
                brk[i] = self.encrypt_rgsw_fft(one_poly, skNfft)

        return brk

    """
    alphapoly[i] = fft(X^i - 1)
    """
    def precompute_alpha(self):
        alphapoly = torch.zeros(self.q, self.Nh, dtype=torch.complex128, device=self.device)

        for i in range(self.q):
            poly = torch.zeros([self.N], dtype=torch.int32, device=self.device)
            poly[0] = -1
            if i < self.N:
                poly[i] += 1
            else:
                poly[i - self.N] += -1
            alphapoly[i] = self.negacyclic_fft(poly)
            
        return alphapoly

    """
    betaapoly[i] = fft(X^i)
    """
    def precompute_beta(self):
        betapoly = torch.zeros(self.q, self.Nh, dtype=torch.complex128, device=self.device)

        for i in range(self.q):
            poly = torch.zeros([self.N], dtype=torch.int32, device=self.device)
            if i < self.N:
                poly[i] = 1
            else:
                poly[i - self.N] = -1
            betapoly[i] = self.negacyclic_fft(poly)
            
        return betapoly

    def nand_map(self, i):
        i += 2 * self.N 
        i %= 2 * self.N
        if 3*(self.q >> 3) <= i < 7*(self.q >> 3): # i \in [3q/8, 7q/8)
            return -(self.Q >> 3)
        else: # i \in [-q/8, 3q/8)
            return self.Q >> 3 

    def gate_bootstrapping(self, ct0, ct1, brk, ksk, gate="NAND"):
        ctsum = ct0 + ct1

        # NAND is supported only
        assert gate == "NAND"
        
        if len(ctsum) > self.n + 1:
            ctsum &= (self.Q  - 1)
            # mod switching 1
            ctsum_ms = (ctsum * (self.Qks/self.Q)).to(torch.int32)

            # key switching
            ctsum_ks = self.LWEkeySwitch(ctsum_ms, ksk)

            # mod switching 2
            ctsum = (ctsum_ks * (self.q/self.Qks)).to(torch.int32)

        ctsum &= (self.q - 1)

        # initialize acc
        acc = torch.zeros([2,self.N], dtype=torch.int32, device=self.device)
        acc[0] = self.f_nand

        accfft = self.negacyclic_fft(acc)

        beta = ctsum[0]
        accfft *= self.betapoly[beta]

        # blind rotation
        alpha = ctsum[1:]
        for i in range(self.n):
            ai = alpha[i]
            accfft += self.alphapoly[ai] * self.rgswmult(accfft, brk[i])

        # extract
        acc = self.negacyclic_ifft(accfft)
        accLWE = self.extract(acc)
        accLWE[0] += (self.Q >> 3)
        accLWE &= (self.Q - 1)
        
        return accLWE

    def setKeySwitchKey(self, ksk):
        self.ksk = ksk
        return self

    def setBlindRotationKey(self, brk):
        self.brk = brk
        return self
    
    def nand(self, ct0, ct1):
        if self.ksk == None or self.brk == None:
            print("Need to set blind rotation key and key switch key")
            print("using setBlindRotationKey and setKeySwitchKey.")
            return
        
        return self.gate_bootstrapping(ct0, ct1, self.brk, self.ksk)
        
