import numpy as np
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

        print("Using numpy")

        self.f_nand = np.zeros([self.N], dtype=np.int32)

        for i in range(self.N):
            self.f_nand[i] = self.nand_map(-i)

        self.root_powers = np.exp((1j * math.pi / self.N) * np.arange(self.Nh))

        self.root_powers_inv = np.exp((1j * math.pi / self.N) * np.arange(0, -self.Nh, -1))

        # parameters for decomposition
        # we will use B-ary decomposition, i.e., cut digits by logB bits
        self.decomp_shift = self.logQ - self.logB * np.arange(self.d, 0, -1).reshape(self.d, 1)
        self.mask = (1 << self.logB) - 1

        self.gvector = 1 << self.decomp_shift

        # for signed decomposition
        self.msbmask = 0
        for shift in self.decomp_shift:
            self.msbmask += (1 << (shift[0] + self.logB - 1))

        # parameters for LWE key switching decomposition
        self.decomp_shift_ks = self.logQks - self.logBks * np.arange(self.dks, 0, -1).reshape(self.dks, 1)
        self.mask_ks = np.array([(1 << self.logBks) - 1], dtype=np.int32)

        self.gvector_ks = 1 << self.decomp_shift_ks
        self.decomp_shift_ks = self.decomp_shift_ks.reshape(self.dks, 1)

        self.alphapoly = self.precompute_alpha()
        self.betapoly = self.precompute_beta()

        self.brk = None
        self.ksk = None

    def create_secret_key(self):
        """ Generate secret key """
        return self.keygen(self.n)

    def create_bootstrap_key(self, sk):
        """ Generate bootstrapping keys, and return a tuple brk (blind rotation keys)
            and ksk (LWE key switching key)

            Keyword arguments:
            sk -- secret key
        """
        skN = self.keygen(self.N)

        skNfft = self.negacyclic_fft(skN)
        brk = self.brkgen(sk, skNfft)

        ksk = self.LWEkskGen(sk, skN)
    
        return brk, ksk

    def keygen(self, dim):
        """ Generate uniform binary secret of size n.
            Note: insecure PRNG is used.
        """
        return np.random.randint(2, size=(dim,), dtype=np.int32)

    def errgen(self, dim):
        """ Generate Gaussian noise of size `dim`.
            Note: insecure PRNG is used.
        """
        e = np.round(self.stddev * np.random.randn(dim))
        e = e.squeeze()
        return e.astype(np.int32)

    def uniform(self, dim, modulus):
        """ Generate Gaussian noise of size `dim` mod `modulus`.
            Note: insecure PRNG is used.
        """
        return np.random.randint(modulus, size=(dim,), dtype=np.int32)

    def negacyclic_fft(self, a):
        """ Negacyclic FFT on the given array a.
        """
        acomplex = a.astype(np.complex128)

        a_precomp = (acomplex[..., :self.Nh] + 1j * acomplex[..., self.Nh:]) * self.root_powers
        return np.fft.fft(a_precomp)

    def negacyclic_ifft(self, A):
        """ Negacyclic inverse FFT on the given array A.
        """
        b = np.fft.ifft(A)
        b *= self.root_powers_inv

        a = np.concatenate((b.real, b.imag), axis=-1).astype(int).astype(np.int32)
        # only when Q is a power-of-two
        a &= self.Q - 1

        return a

    def to_signed(self, v, logQ):
        """ Convert unsigned array to signed array mod Q.

            Keyword arguments:
            v -- unsigned array to convert
            logQ -- log_2 of Q, i.e., Q = 1 << logQ
        """
        Q = (1 << logQ)
        v &= (Q-1)
        msb = (v & Q//2) >> (logQ - 1)
        v -= Q * msb
        return v

    def decompose(self, a):
        """ Unsigned digit (B) decomposition of given array a.
        """
        return (a.reshape(1, *a.shape) >> self.decomp_shift.reshape(self.d, 1, 1)) & self.mask
        
    def signed_decompose(self, a):
        """ Signed decomposition of given array a.
            Uses `decompose` two times.
        """
        da = self.decompose(a + (a & self.msbmask))
        da -= self.decompose((a & self.msbmask))
        return da

    def encrypt_rgsw_fft(self, z, skfft):
        """ Return RGSW_sk(z). """
        rgsw = np.zeros((self.d, 2, 2, self.N), dtype=np.int32)

        # generate the 'a' part
        rgsw[:, :, 1, :] = np.random.randint(self.Q, size=(self.d, 2, self.N), dtype=np.int32)

        # add error on b
        rgsw[:, :, 0, :] = np.round(self.stddev * np.random.randn(self.d, 2, self.N)).astype(np.int32)

        rgsw = self.to_signed(rgsw, self.logQ)

        rgswfft = self.negacyclic_fft(rgsw)

        rgswfft[:, :, 0, :] -= rgswfft[:, :, 1, :] * skfft.reshape(1, 1, self.Nh)

        gzfft = self.negacyclic_fft(self.gvector * z)
        rgswfft[:, 0, 0, :] += gzfft
        rgswfft[:, 1, 1, :] += gzfft

        return rgswfft

    def rgswmult(self, ctfft, rgswfft):
        """ Multiply ctfft (RLWE) and rgswfft (RGSW) and return an RLWE ciphertext. """

        ct = self.negacyclic_ifft(ctfft)
        dct = self.signed_decompose(ct)
        multfft = self.negacyclic_fft(dct).reshape(self.d, 2, 1, self.Nh) * rgswfft
        return np.sum(multfft, axis=(0, 1))
    
    def encryptLWE(self, message, sk):
        """ Encrypt a bit.

        Keyword arguments:
        message -- a bit, 0 or 1
        sk -- secret key to use
        """

        ct = self.uniform(self.n + 1, self.q)

        ct[0] = 0

        ct[0] = message * self.q // 4 - np.sum(ct[1:] * sk)
        ct[0] += self.errgen(1)
        ct &= (self.q - 1)

        return ct

    def decryptLWE(self, ct, sk, ksk=None):
        """ Decrypt LWE ciphertext. Do key switch if needed. """

        if len(ct) > self.n + 1:
            if self.ksk is None and ksk is None:
                print("Cannot key switch to small key. Provide switching key.")
            ct_ms = (ct * (self.Qks/self.Q)).astype(np.int32)
            ct_ks = self.LWEkeySwitch(ct_ms, self.ksk)
            ct_n = (ct_ks * (self.q/self.Qks)).astype(np.int32)
        else:
            ct_n = ct

        m_dec = ct_n[0] + np.sum(ct_n[1:] * sk)
        m_dec %= self.q

        m_dec = np.round(m_dec / (self.q / 4.0)).astype(int)

        return m_dec % 4

    def extract(self, ctRLWE):
        """ Extract an LWE ciphertext from ctRLWE, holding its the constant term as plaintext."""

        beta = ctRLWE[0][0]

        alpha = ctRLWE[1, :]
        alpha[1:] = -alpha[1:][::-1]

        return np.concatenate(([beta], alpha))

    def decompose_ks(self, a):
        """ Decompose LWE ciphertext for LWE key switching.
        """
        assert len(a.shape) == 1
        res = (a.reshape(1, -1) >> self.decomp_shift_ks) & self.mask_ks
        return res

    def LWEkskGen(self, sk, skN):
        """ Generate key switching key from skN to sk. """

        ksk = np.random.randint(self.Qks, size=(self.dks, self.Bks, self.N, self.n + 1), dtype=np.int32)
        ksk[..., 0] = np.round(self.stddev * np.random.randn(self.dks, self.Bks, self.N)).astype(np.int32)
        ksk[..., 0] -= np.sum(ksk[..., 1:] * sk, axis=-1)
        ksk[..., 0] += (self.gvector_ks * np.arange(self.Bks)).reshape(self.dks, self.Bks, 1) * skN
        ksk &= (self.Qks - 1)

        return ksk

    def LWEkeySwitch(self, ctLWE, kskLWE):
        """ Switch key of ctLWE (dimension N) to a smaller key (dimension n). """

        alpha = ctLWE[1:]
        dalpha = self.decompose_ks(alpha)
        switched = np.zeros(self.n + 1, dtype=np.int32)
        switched[0] = ctLWE[0]

        for r in range(self.dks):
            for i in range(self.N):
                switched += kskLWE[r, dalpha[r, i], i]

        switched &= (self.Qks - 1)

        return switched

    def brkgen(self, sk, skNfft):
        """ Generate blind rotation keys. """

        zero_poly = np.zeros([self.N], dtype=np.int32)

        one_poly = np.zeros([self.N], dtype=np.int32)
        one_poly[0] = 1

        brk = np.zeros((self.n, self.d, 2, 2, self.Nh), dtype=np.complex128)
        for i in range(self.n):
            if sk[i] == 0:
                brk[i] = self.encrypt_rgsw_fft(zero_poly, skNfft)
            else:
                brk[i] = self.encrypt_rgsw_fft(one_poly, skNfft)

        return brk

    def precompute_alpha(self):
        """ Precompute fft(X^i - 1), return alphapoly where alphapoly[i] = fft(X^i - 1). """

        alphapoly = np.zeros((self.q, self.Nh), dtype=np.complex128)

        for i in range(self.q):
            poly = np.zeros([self.N], dtype=np.int32)
            poly[0] = -1
            if i < self.N:
                poly[i] += 1
            else:
                poly[i - self.N] += -1
            alphapoly[i] = self.negacyclic_fft(poly)
            
        return alphapoly

    def precompute_beta(self):
        """ Precompute fft(X^i), return alphapoly where betaapoly[i] = fft(X^i). """

        betapoly = np.zeros((self.q, self.Nh), dtype=np.complex128)

        for i in range(self.q):
            poly = np.zeros([self.N], dtype=np.int32)
            if i < self.N:
                poly[i] = 1
            else:
                poly[i - self.N] = -1
            betapoly[i] = self.negacyclic_fft(poly)
            
        return betapoly

    def nand_map(self, i):
        """ Precompute LUT function for NAND gate. """

        i += 2 * self.N 
        i %= 2 * self.N
        if 3 * (self.q >> 3) <= i < 7 * (self.q >> 3):
            return -(self.Q >> 3)
        else:
            return self.Q >> 3 

    def gate_bootstrapping(self, ct0, ct1, brk, ksk, gate="NAND"):
        """ Do a gate bootstrapping for given `ct0` and `ct1`.
        Only NAND is supported currently.

        Keyword arguments:
        ct0, ct1 -- input ciphertexts
        brk -- blind rotation keys to use
        ksk -- LWE key switching keys to use
        gate :String -- gate to perform e.g., "NAND". 
        """

        ctsum = ct0 + ct1

        # NAND is supported only
        assert gate == "NAND"
        
        if len(ctsum) > self.n + 1:
            ctsum &= (self.Q - 1)
            ctsum_ms = (ctsum * (self.Qks/self.Q)).astype(np.int32)
            ctsum_ks = self.LWEkeySwitch(ctsum_ms, ksk)
            ctsum = (ctsum_ks * (self.q/self.Qks)).astype(np.int32)

        ctsum &= (self.q - 1)

        acc = np.zeros([2, self.N], dtype=np.int32)
        acc[0] = self.f_nand

        accfft = self.negacyclic_fft(acc)

        beta = ctsum[0]
        accfft *= self.betapoly[beta]

        alpha = ctsum[1:]
        for i in range(self.n):
            ai = alpha[i]
            accfft += self.alphapoly[ai] * self.rgswmult(accfft, brk[i])

        acc = self.negacyclic_ifft(accfft)
        accLWE = self.extract(acc)
        accLWE[0] += (self.Q >> 3)
        accLWE &= (self.Q - 1)
        
        return accLWE

    def setKeySwitchKey(self, ksk):
        """ Enroll LWE key switching keys to engine for simpler codes. """

        self.ksk = ksk
        return self

    def setBlindRotationKey(self, brk):
        """ Enroll blind rotation keys to engine for simpler codes. """

        self.brk = brk
        return self
    
    def nand(self, ct0, ct1):
        """ Do a NAND gate on two input ciphertexts ct0 and ct1 """

        if self.ksk is None or self.brk is None:
            print("Need to set blind rotation key and key switch key")
            print("using methods setBlindRotationKey and setKeySwitchKey.")
            return
        
        return self.gate_bootstrapping(ct0, ct1, self.brk, self.ksk)
